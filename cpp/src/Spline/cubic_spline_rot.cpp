// Copyright 2019 Alexander Liniger

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#include "Spline/cubic_spline_rot.h"

namespace mpcc{
CubicSplineRot::CubicSplineRot()
: data_set_(false)
{
}

Eigen::Matrix3d getSkewMatrix(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
    result(0,1) = -v(2);
    result(0,2) =  v(1);
    result(1,0) =  v(2);
    result(1,2) = -v(0);
    result(2,0) = -v(1);
    result(2,1) =  v(0);
    return result;
}

Eigen::Vector3d getInverseSkewVector(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    result << R(2,1), R(0,2), R(1,0);
    return result;
}

Eigen::Matrix3d LogMatrix(const Eigen::Matrix3d &R)
{
    Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
    double trace_R = R.trace();
    if (std::fabs(trace_R + 1.0) < 1e-6) 
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(R);
        if (eigensolver.info() != Eigen::Success) 
        {
            std::cerr << "EigenSolver failed" << std::endl;
            result = Eigen::Matrix3d::Zero();
        }

        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
        for (int i = 0; i < eigenvalues.size(); ++i) 
        {
            if (std::fabs(eigenvalues(i) - 1.0) < 1e-4) 
            {
                Eigen::VectorXd unit_eigenvector = eigenvectors.col(i).normalized();
                result = -getSkewMatrix(unit_eigenvector)*M_PI;
            }
        }
    } 
    else if(std::fabs(trace_R - 3.0) < 1e-6)
    {
        result = Eigen::Matrix3d::Zero();
    }
    else 
    {
        double th = acos((R.trace()-1.0)/2.0);
        result = 1.0 /2.0 * th/sin(th) * (R - R.transpose());
    }

    return result;
}

Eigen::Matrix3d ExpMatrix(const Eigen::Matrix3d &sk)
{
    if ((sk.transpose() + sk).norm() >= 1E-8 || sk.diagonal().norm() >=1E-8)
    {
        std::cerr << "Input matrix must be skew-symmetric matrix!!!" << std::endl;
        return Eigen::Matrix3d::Zero();
    }
    Eigen::Vector3d v = getInverseSkewVector(sk);
    double vn = v.norm();
    if(vn <= 1E-8)
    {
        return Eigen::Matrix3d::Identity() + cos(vn)*sk + 1/2*cos(vn)*sk*sk;;
    }
    return Eigen::Matrix3d::Identity() + sin(vn)/vn*sk + (1-cos(vn))/pow(vn,2)*sk*sk;
}

void CubicSplineRot::setRegularData(const Eigen::VectorXd &x_in,const std::vector<Eigen::Matrix3d> &R_in,const double delta_x) {
    //if x and y have same length, store given data in spline data struct
    if(x_in.size() == R_in.size())
    {
        spline_data_.x_data = x_in;
        spline_data_.R_data = R_in;
        spline_data_.n_points = x_in.size();
        spline_data_.is_regular = true;
        spline_data_.delta_x = delta_x;

        data_set_ = true;
    }
    else
    {
        std::cout << "input data does not have the same length" << std::endl;
    }
}

void CubicSplineRot::setData(const Eigen::VectorXd &x_in, const std::vector<Eigen::Matrix3d> &R_in)
{
    if (x_in.size() != R_in.size()) {
        std::cout << "Input data does not have the same length" << std::endl;
        return;
    }

    // Allocate maximum possible size to avoid unnecessary allocations
    int n = x_in.size();
    Eigen::VectorXd x_filtered(n);
    std::vector<Eigen::Matrix3d> R_filtered;
    R_filtered.reserve(n);  // Preallocate to avoid frequent reallocations

    x_filtered(0) = x_in(0);
    R_filtered.push_back(R_in[0]);
    int new_size = 1;  // Counter for valid data points

    // Filter only increasing x values
    for (int i = 1; i < n; i++) {
        if (x_in(i) > x_filtered(new_size - 1)) {  // Keep only strictly increasing x values
            x_filtered(new_size) = x_in(i);
            R_filtered.push_back(R_in[i]);
            new_size++;
        }
    }

    // Ensure there are enough points for interpolation
    if (new_size < 2) {
        std::cout << "Filtered data is too small for spline interpolation" << std::endl;
        data_set_ = false;
        return;
    }

    // Resize vectors to store only valid data
    spline_data_.x_data = x_filtered.head(new_size);
    spline_data_.R_data = std::move(R_filtered);
    spline_data_.n_points = new_size;
    spline_data_.is_regular = false;
    spline_data_.delta_x = 0;

    // Update x_map for fast lookups (using unordered_map)
    spline_data_.x_map.clear();
    for (int i = 0; i < new_size; i++) {
        spline_data_.x_map[spline_data_.x_data(i)] = i;
    }

    data_set_ = true;
}

bool CubicSplineRot::compSplineRotParams()
{
    if(!data_set_)
    {
        return false;
    }
    spline_params_.a.resize(spline_data_.n_points-1);
    spline_params_.b.resize(spline_data_.n_points-1);
    spline_params_.c.resize(spline_data_.n_points-1);
    spline_params_.d.resize(spline_data_.n_points-1);
    for(size_t i=0;i<spline_data_.n_points-1;i++)
    {
        spline_params_.a(i) = 0.0;
        spline_params_.b(i) = 0.0;
        spline_params_.c(i) = 3.0 / pow((spline_data_.x_data(i+1) - spline_data_.x_data(i)), 2);
        spline_params_.d(i) = -2.0 / pow((spline_data_.x_data(i+1) - spline_data_.x_data(i)), 3);
    }                                                                                         
    return true;
}

int CubicSplineRot::getIndex(const double x) const
{
    // given a x value find the closest point in the spline to evalute it
    // special case if x is regularly space
    // assumes wrapped data!

    // if special case of end points
    if(x == spline_data_.x_data(spline_data_.n_points-1))
    {
        return spline_data_.n_points-1;
    }
    // if regular index can be found by rounding
    if(spline_data_.is_regular)
    {
        return int(floor(x/spline_data_.delta_x));
    }
    // if irregular index need to be searched
    else
    {
        auto min_it = spline_data_.x_map.upper_bound(x);
        if(min_it==spline_data_.x_map.end())
            return -1;
        else{
            return min_it->second-1;
        }

    }
}

double CubicSplineRot::unwrapInput(double x) const
{
    double x_max = spline_data_.x_data(spline_data_.n_points-1);
    // return x - x_max*std::floor(x/x_max);
    return std::max(0., std::min(x,x_max));
}

void CubicSplineRot::genSpline(const Eigen::VectorXd &x_in,const std::vector<Eigen::Matrix3d> &R_in,const bool is_regular)
{
    // given x and R data generate spline
    // special case for regular or irregular spaced data points in x
    // if regular the spacing in x is given by deltaX

    // store data in data struct
    if(is_regular)
    {
        double delta_x = x_in(1) - x_in(0);
        setRegularData(x_in,R_in,delta_x);
    }
    else
    {
        setData(x_in,R_in);
    }
    // given data compute spline parameters

    bool succes = compSplineRotParams();

    // TODO if succes is false call exeption
}

Eigen::Matrix3d CubicSplineRot::getPoint(double x) const
{
    // evaluation of spline a x
    int index;
    double x_i;
    double dx,dx2,dx3;
    // wrape input to data -> x data needs start at 0 and contain end point!!!
    x = unwrapInput(x);
    // compute index
    index = getIndex(x);
    if(index == spline_data_.n_points-1) return spline_data_.R_data[spline_data_.n_points-1];

    // access previous points
    x_i = spline_data_.x_data(index);

    // compute diff to point and it's powers
    dx = x-x_i;
    dx2 = dx*dx;
    dx3 = dx*dx2;

    Eigen::Matrix3d log_RR = LogMatrix(spline_data_.R_data[index].transpose() * spline_data_.R_data[index+1]); 
    return spline_data_.R_data[index] * ExpMatrix(log_RR*(spline_params_.a[index] + spline_params_.b[index]*dx + spline_params_.c[index]*dx2 + spline_params_.d[index]*dx3));
}

Eigen::Vector3d CubicSplineRot::getDerivative(double x) const
{
    // evaluate first derivative of spline
    // identical to noram spline with LogMatrix(R_i^T * R_i+1))[b + 2c dx + 3d dx^2])
    int index;
    double x_i;
    double dx,dx2;

    x = unwrapInput(x);
    index = getIndex(x);
    if(index == spline_data_.n_points-1) return Eigen::Vector3d::Zero();

    x_i = spline_data_.x_data(index);

    dx = x-x_i;
    dx2 = dx*dx;

    Eigen::Vector3d Log_RR = getInverseSkewVector(LogMatrix(spline_data_.R_data[index].transpose() * spline_data_.R_data[index+1]));
    return Log_RR * (spline_params_.b[index] + 2.0*spline_params_.c[index]*dx + 3.0*spline_params_.d[index]*dx2);
}
}