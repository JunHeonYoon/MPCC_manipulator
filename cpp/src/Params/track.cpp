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

#include "Params/track.h"
namespace mpcc{
Track::Track(std::string file) 
{
    /////////////////////////////////////////////////////
    // Loading Model and Constraint Parameters //////////
    /////////////////////////////////////////////////////
    std::ifstream iTrack(file);
    json jsonTrack;
    iTrack >> jsonTrack;
    
    // Model Parameters
    std::vector<double> x = jsonTrack["X"];
    X_ = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
    std::vector<double> y = jsonTrack["Y"];
    Y_ = Eigen::Map<Eigen::VectorXd>(y.data(), y.size());
    std::vector<double> z = jsonTrack["Z"];
    Z_ = Eigen::Map<Eigen::VectorXd>(z.data(), z.size());


    std::vector<double> quat_x = jsonTrack["quat_X"];
    std::vector<double> quat_y = jsonTrack["quat_Y"];
    std::vector<double> quat_z = jsonTrack["quat_Z"];
    std::vector<double> quat_w = jsonTrack["quat_W"];


    R_.resize(quat_x.size());

    for(size_t i=0;i<quat_x.size();i++)
    {
        Eigen::Quaterniond q;
        q.x() = quat_x[i];
        q.y() = quat_y[i];
        q.z() = quat_z[i];
        q.w() = quat_w[i];
        R_[i] = q.normalized().toRotationMatrix();
    }
}

void Track::setTrack(const Eigen::VectorXd& X, const Eigen::VectorXd& Y, const Eigen::VectorXd& Z, const std::vector<Eigen::Matrix3d>& R)
{
    X_ = X;
    Y_ = Y;
    Z_ = Z;
    R_ = R;
}

void Track::initTrack(const Eigen::Matrix4d& init_pose)
{
    X_ = X_.array() - X_(0) + init_pose(0,3);
    Y_ = Y_.array() - Y_(0) + init_pose(1,3);
    Z_ = Z_.array() - Z_(0) + init_pose(2,3);

    Eigen::Matrix3d tmp_R = R_[0].transpose() * init_pose.block(0,0,3,3);
    for(size_t i=0; i<R_.size(); i++)
    {
        R_[i] = correctRotationMatrix(R_[i] * tmp_R);
    }
}

TrackPos Track::getTrack(const Eigen::Matrix4d& init_pose)
{
   initTrack(init_pose);

    return {Eigen::Map<Eigen::VectorXd>(X_.data(), X_.size()), 
            Eigen::Map<Eigen::VectorXd>(Y_.data(), Y_.size()),
            Eigen::Map<Eigen::VectorXd>(Z_.data(), Z_.size()),
            R_};
}

Eigen::Matrix3d Track::enforceOrthogonality(const Eigen::Matrix3d& R) 
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    return U * V.transpose();  // Orthogonalized rotation matrix
}

Eigen::Matrix3d Track::normalizeDeterminant(const Eigen::Matrix3d& R) 
{
    double det = R.determinant();
    if (std::abs(det - 1.0) > 1e-10) {
        return R / std::cbrt(det);  // Scale to ensure determinant is 1
    }
    return R;
}

Eigen::Matrix3d Track::correctRotationMatrix(const Eigen::Matrix3d& R) 
{
    Eigen::Matrix3d R_orthogonalized = enforceOrthogonality(R);
    return normalizeDeterminant(R_orthogonalized);
}
}
