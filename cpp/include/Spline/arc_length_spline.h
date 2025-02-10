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

#ifndef MPCC_ARC_LENGTH_SPLINE_H
#define MPCC_ARC_LENGTH_SPLINE_H

#include "Model/robot_model.h"
#include "cubic_spline.h"
#include "cubic_spline_rot.h"
#include "types.h"
#include "Params/params.h"
#include <map>
#include <vector>

namespace mpcc{

/// @brief raw path data
/// @param X (Eigen::VectorXd) X position data
/// @param Y (Eigen::VectorXd) Y position data
/// @param Z (Eigen::VectorXd) Z position data
/// @param R (std::vector<Eigen::Matrix3d>) Rotation matrix data
struct RawPath{
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    Eigen::VectorXd Z;
    std::vector<Eigen::Matrix3d> R;
};

/// @brief path data
/// @param X (Eigen::VectorXd) X position data
/// @param Y (Eigen::VectorXd) Y position data
/// @param Z (Eigen::VectorXd) Z position data
/// @param R (std::vector<Eigen::Matrix3d>) Rotation matrix data
/// @param s (Eigen::VectorXd) path parameter
/// @param arc_length (double) total arc length
/// @param n_points (int) number of path points
struct PathData{
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    Eigen::VectorXd Z;
    std::vector<Eigen::Matrix3d> R;
    Eigen::VectorXd s;
    double arc_length;
    int n_points;
};

class ArcLengthSpline {
public:
    ArcLengthSpline();
    ArcLengthSpline(const PathToJson &path);
    ArcLengthSpline(const PathToJson &path,const ParamValue &param_value);

    /// @brief  generate 6-D spline given X-Y-Z position and orientation path data
    /// @param X (Eigen::VectorXd) X position data
    /// @param Y (Eigen::VectorXd) Y position data
    /// @param Z (Eigen::VectorXd) Z position data
    /// @param R (std::vector<Eigen::Matrix3d>) Rotation matrix data
    void gen6DSpline(const Eigen::VectorXd &X,const Eigen::VectorXd &Y,const Eigen::VectorXd &Z,const std::vector<Eigen::Matrix3d> &R);

    /// @brief get X-Y-Z position data given path parameter (s)
    /// @param s (double) path parameter 
    /// @return (Eigen::Vector3d) X-Y-Z position data
    Eigen::Vector3d getPosition(double) const;

    /// @brief get Orientation data given path parameter (s)
    /// @param s (double) path parameter 
    /// @return (Eigen::Matrix3d) Orientation data
    Eigen::Matrix3d getOrientation(double) const;

    /// @brief get X'(s)-Y'(s)-Z'(s) position data derivatived by path parameter (s) given path parameter (s)
    /// @param  (double) path parameter 
    /// @return (Eigen::Vector3d) X'(s)-Y'(s)-Z'(s) position data
    Eigen::Vector3d getDerivative(double) const;

    /// @brief get orientation first derivative data by path parameter (s) given path parameter (s)
    /// @param  (double) path parameter 
    /// @return (Eigen::Vector3d) orientation first derivative data
    Eigen::Vector3d getOrientationDerivative(double) const;

    /// @brief get X''(s)-Y''(s)-Z''(s) position data twice derivatived by path parameter (s) given path parameter (s)
    /// @param  (double) path parameter 
    /// @return (Eigen::Vector3d) X''(s)-Y''(s)-Z''(s) position data
    Eigen::Vector3d getSecondDerivative(double) const;

    /// @brief get total arc length of splined path 
    /// @return (double) total arc length
    double getLength() const;

    /// @brief compute path parameter of projected on splined path which calculated by Newton-Euler method given current state
    /// @param s (double) current path parameter
    /// @param ee_pose (Eigen::Matrix4d) current pose of End-Effector
    /// @return (double) projected arc lenth
    double projectOnSpline(const double &s, const Eigen::Matrix4d ee_pose) const;

    /// @brief get splined path data
    /// @return (PathData) splined path data
    PathData getPathData(){return path_data_;}

private:
    /// @brief set irregular path point data to make path data (PathData)
    /// @param X_in (Eigen::VectorXd) X position data
    /// @param Y_in (Eigen::VectorXd) Y position data
    /// @param Z_in (Eigen::VectorXd) Z position data
    /// @param R_in (std::vector<Eigen::Matrix3d>) Rotation matrix data
    void setData(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in,const Eigen::VectorXd &Z_in,const std::vector<Eigen::Matrix3d> &R_in);

    /// @brief set regular path point data to make path data (PathData)
    /// @param X_in (Eigen::VectorXd) X position data
    /// @param Y_in (Eigen::VectorXd) Y position data
    /// @param Z_in (Eigen::VectorXd) Z position data
    /// @param R_in (std::vector<Eigen::Matrix3d>) Rotation matrix data
    /// @param s_in (Eigen::VectorXd) path parameter
    void setRegularData(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in,const Eigen::VectorXd &Z_in,const std::vector<Eigen::Matrix3d> &R_in,const Eigen::VectorXd &s_in);

    /// @brief compute arc length(position & orientation) given X, Y, Z position and R orientation points
    /// @param X_in (Eigen::VectorXd) X position data
    /// @param Y_in (Eigen::VectorXd) Y position data
    /// @param Z_in (Eigen::VectorXd) Z position data
    /// @param R_in (std::vector<Eigen::MatrixXd>) R orientation data
    /// @return (Eigen::VectorXd) arc length for each point
    Eigen::VectorXd compArcLength(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in,const Eigen::VectorXd &Z_in,const std::vector<Eigen::Matrix3d> &R_in) const;

    /// @brief  re-sample path parameter parametrized X-Y-Z spline path with N_spline data points using equidistant path parameter values
    /// @param initial_spline_x (CubicSpline) X position parameterized by path parameter (S) 
    /// @param initial_spline_y (CubicSpline) Y position parameterized by path parameter (S) 
    /// @param initial_spline_z (CubicSpline) Z position parameterized by path parameter (S) 
    /// @param initial_spline_r (CubicSplineRot) Orientation parameterized by path parameter (S) 
    /// @return (PathData) re-sampled x, y, z position and orientation path data with path parameter (s)
    PathData resamplePath(const CubicSpline &initial_spline_x,const CubicSpline &initial_spline_y,const CubicSpline &initial_spline_z,const CubicSplineRot &initial_spline_r) const;

    /// @brief remove points which are not at all equally spaced, to avoid fitting problems
    /// @param X_original (Eigen::VectorXd) X position data
    /// @param Y_original (Eigen::VectorXd) Y position data
    /// @param Z_original (Eigen::VectorXd) Z position data
    /// @param R_original (std::vector<Eigen::Matrix3d>) Orientation data
    /// @return (RawPath) x, y, z position and Orientation path data 
    RawPath outlierRemoval(const Eigen::VectorXd &X_original,const Eigen::VectorXd &Y_original,const Eigen::VectorXd &Z_original,const std::vector<Eigen::Matrix3d> &R_original) const;

    /// @brief generate cubic-splined X, Y, Z position path points parameterized by path parameter (S)
    /// @param X (Eigen::VectorXd) X position data
    /// @param Y (Eigen::VectorXd) Y position data
    /// @param Z (Eigen::VectorXd) Z position data
    /// @param R (std::vector<Eigen::Matrix3d>) Rotation matrix data
    void fitSpline(const Eigen::VectorXd &X,const Eigen::VectorXd &Y,const Eigen::VectorXd &Z,const std::vector<Eigen::Matrix3d> &R);

    /// @brief if path parameter (s) is larger than total path parameter, then s become unwrapped. (like angle is -pi ~ pi)
    /// @param x (double) path parameter
    /// @return (double) unwrapped path parameter (s) data
    double unwrapInput(double x) const;

    PathData path_data_;
    CubicSpline spline_x_;
    CubicSpline spline_y_;
    CubicSpline spline_z_;
    CubicSplineRot spline_r_;
    Param param_;
};
}
#endif //MPCC_ARC_LENGTH_SPLINE_H