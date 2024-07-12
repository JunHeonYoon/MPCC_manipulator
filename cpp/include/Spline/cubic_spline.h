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

#ifndef MPCC_CUBIC_SPLINE_H
#define MPCC_CUBIC_SPLINE_H

#include "config.h"
#include <map>

namespace mpcc{
/// @brief spline parameter struct y = a + b dx + c dx^2 + d dx^3, caution: x, y is not path, but function param
/// @param a (Eigen::VectorXd) constant
/// @param b (Eigen::VectorXd) first order
/// @param c (Eigen::VectorXd) second order
/// @param d (Eigen::VectorXd) third order
struct SplineParams{
    Eigen::VectorXd a;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::VectorXd d;
};

/// @brief input data for spline
/// @param x_data (Eigen::VectorXd) x data, caution: x, y is not path, but function param
/// @param y_data (Eigen::VectorXd) y data, caution: x, y is not path, but function param
/// @param n_points (int) number of points
/// @param is_regular (bool) regular (True) or irregular (False) spaced points in x direction
/// @param delta_x (double) spacing of regular space points
/// @param x_map (std::map<double,int>) if getting x_data, ouput is index of the x_data
struct SplineData{
    Eigen::VectorXd x_data;    
    Eigen::VectorXd y_data;    
    int n_points;
    bool is_regular;
    double delta_x;
    std::map<double,int> x_map;
};

class CubicSpline {
public:
    CubicSpline();

    /// @brief compute cubic spline parameters (SplineParams) given x, y data, caution: x, y is not path, but function param
    /// @param x_in (Eigen::VectorXd) x data, caution: x, y is not path, but function param
    /// @param y_in (Eigen::VectorXd) y data, caution: x, y is not path, but function param
    /// @param is_regular (bool) regular (True) or irregular (False) spaced points in x direction
    void genSpline(const Eigen::VectorXd &x_in,const Eigen::VectorXd &y_in,bool is_regular);

    /// @brief compute spline value y = a + b dx + c dx^2 + d dx^3 given x data
    /// @param x (double) x data, caution: x, y is not path, but function param
    /// @return (double) y data, caution: x, y is not path, but function param
    double getPoint(double x) const;

    /// @brief compute first derivative of spline  y' = b + 2 c dx + 3 d dx^2 given x data
    /// @param x (double) x data, caution: x, y is not path, but function param
    /// @return (double) y' data, caution: x, y is not path, but function param
    double getDerivative(double x) const;

    /// @brief compute second derivative of spline  y'' = 2 c + 6 d dx given x data
    /// @param x (double) x data, caution: x, y is not path, but function param
    /// @return (double) y'' data, caution: x, y is not path, but function param
    double getSecondDerivative(double x) const;

private:
    bool data_set_;

    /// @brief set regular data points
    /// @param x_in (Eigen::VectorXd) x data, caution: x, y is not path, but function param
    /// @param y_in (Eigen::VectorXd) y data, caution: x, y is not path, but function param
    /// @param delta_x (double) spacing of regular space points
    void setRegularData(const Eigen::VectorXd &x_in,const Eigen::VectorXd &y_in,double delta_x);

    /// @brief set irregular data points
    /// @param x_in (Eigen::VectorXd) x data, caution: x, y is not path, but function param
    /// @param y_in (Eigen::VectorXd) y data, caution: x, y is not path, but function param
    void setData(const Eigen::VectorXd &x_in,const Eigen::VectorXd &y_in);

    /// @brief compute cubic spline parameters (SplineParams)
    /// @return (bool) success or fail
    bool compSplineParams();

    /// @brief find the closest x point in the spline given a x value
    /// @param x (double) x data, caution: x, y is not path, but function param
    /// @return (bool) index of x closest point
    int getIndex(double x) const;

    /// @brief if x(arc length) is larger than total arc length, then x become unwrapped. (like angle is -pi ~ pi)
    /// @param x (double)  x data, caution: x, y is not path, but function param
    /// @return (double) unwrapped x data
    double unwrapInput(double x) const;

    SplineParams spline_params_;
    SplineData spline_data_;
};
}
#endif //MPCC_CUBIC_SPLINE_H