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

#ifndef MPCC_CUBIC_SPLINE_ROT_H
#define MPCC_CUBIC_SPLINE_ROT_H

#include "config.h"
#include <map>
#include <vector>

// Original Paper
// Kang, I. G., and F. C. Park.
// "Cubic spline algorithms for orientation interpolation."

namespace mpcc{

/// @brief spline parameter struct R = R_i*ExpMatrix(LogMatrix(R_i^T * R_i+1))[a + b dx + c dx^2 + d dx^3]), caution: x, R is not path, but function param
/// @param a (Eigen::VectorXd) constant
/// @param b (Eigen::VectorXd) first order
/// @param c (Eigen::VectorXd) second order
/// @param d (Eigen::VectorXd) third order
struct SplineRotParams{
    Eigen::VectorXd a;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::VectorXd d;
};

/// @brief input data for spline
/// @param x_data (Eigen::VectorXd) x data, caution: x, R is not path, but function param
/// @param R_data (std::vector<Eigen::Matrix3d>) R data, caution: x, R is not path, but function param
/// @param n_points (int) number of points
/// @param is_regular (bool) regular (True) or irregular (False) spaced points in x direction
/// @param delta_x (double) spacing of regular space points
/// @param x_map (std::map<double,int>) if getting x_data, ouput is index of the x_data
struct SplineRotData{
    Eigen::VectorXd x_data;    
    std::vector<Eigen::Matrix3d> R_data;    
    int n_points;
    bool is_regular;
    double delta_x;
    std::map<double,int> x_map;
};


/// @brief calculate skew symmetric matrix from 3D vector
/// @param v (Eigen::Vector3d) input vector
/// @return (Eigen::Matrix3d) skew symmetric matrix
Eigen::Matrix3d getSkewMatrix(const Eigen::Vector3d &v);

/// @brief calculate inverse of skew symmetric matrix from skew symmetric matrix
/// @param v (Eigen::Matrix3d) input skew symmetric matrix
/// @return (Eigen::Vector3d) 3D vector
Eigen::Vector3d getInverseSkewVector(const Eigen::Matrix3d &R);

/// @brief calculate matrix logarithm
/// @param v (Eigen::Matrix3d) input matrix; rotation matrix
/// @return (Eigen::Matrix3d) log(Matrix); skew symmetric matrix
Eigen::Matrix3d LogMatrix(const Eigen::Matrix3d &R);

/// @brief calculate matrix exponential
/// @param v (Eigen::Matrix3d) input matrix; skew symmetric matrix
/// @return (Eigen::Matrix3d) exp(Matrix); rotation matrix
Eigen::Matrix3d ExpMatrix(const Eigen::Matrix3d &sk);

class CubicSplineRot {
public:
    CubicSplineRot();

    /// @brief compute cubic spline parameters (SplineRotParams) given x, R data, caution: x, R is not path, but function param
    /// @param x_in (Eigen::VectorXd) x data, caution: x, R is not path, but function param
    /// @param R_in (std::vector<Eigen::Matrix3d>) R data, caution: x, R is not path, but function param
    /// @param is_regular (bool) regular (True) or irregular (False) spaced points in x direction
    void genSpline(const Eigen::VectorXd &x_in,const std::vector<Eigen::Matrix3d> &R_in,bool is_regular);

    /// @brief compute spline value R = R_i*ExpMatrix(LogMatrix(R_i^T * R_i+1))[a + b dx + c dx^2 + d dx^3]) given x data
    /// @param x (double) x data, caution: x, R is not path, but function param
    /// @return (Eigen::Matrix3d) R data, caution: x, R is not path, but function param
    Eigen::Matrix3d getPoint(double x) const;

    /// @brief compute first derivative of spline; LogMatrix(R_i^T * R_i+1))[b + 2c dx + 3d dx^2]) given x data
    /// @param x (double) x data, caution: x, R is not path, but function param
    /// @return (Eigen::Vector3d) Rotation derivative wrt x, caution: x, R is not path, but function param
    Eigen::Vector3d getDerivative(double x) const;

private:
    bool data_set_;

    /// @brief set regular data points
    /// @param x_in (Eigen::VectorXd) x data, caution: x, R is not path, but function param
    /// @param R_in (std::vector<Eigen::Matrix3d>) R data, caution: x, R is not path, but function param
    /// @param delta_x (double) spacing of regular space points
    void setRegularData(const Eigen::VectorXd &x_in,const std::vector<Eigen::Matrix3d> &R_in,double delta_x);

    /// @brief set irregular data points
    /// @param x_in (Eigen::VectorXd) x data, caution: x, R is not path, but function param
    /// @param R_in (std::vector<Eigen::Matrix3d>) R data, caution: x, R is not path, but function param
    void setData(const Eigen::VectorXd &x_in,const std::vector<Eigen::Matrix3d> &R_in);

    /// @brief compute cubic spline parameters (SplineRotParams)
    /// @return (bool) success or fail
    bool compSplineRotParams();

    /// @brief find the closest x point in the spline given a x value
    /// @param x (double) x data, caution: x, R is not path, but function param
    /// @return (int) index of x closest point
    int getIndex(double x) const;

    /// @brief if x(arc length) is larger than total arc length, then x become unwrapped. (like angle is -pi ~ pi)
    /// @param x (double)  x data, caution: x, R is not path, but function param
    /// @return (double) unwrapped x data
    double unwrapInput(double x) const;

    SplineRotParams spline_params_;
    SplineRotData spline_data_;
};
}
#endif //MPCC_CUBIC_SPLINE_ROT_H