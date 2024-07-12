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

#ifndef MPCC_COST_H
#define MPCC_COST_H

#include "config.h"
#include "types.h"
#include "Spline/arc_length_spline.h"
#include "Model/robot_data.h"
#include <vector>

namespace mpcc{

/// @brief Gradient of Cost wrt state and input
/// @param f_x (Eigen::Matrix<double,NX,1>) Gradient of Cost wrt state
/// @param f_u (Eigen::Matrix<double,NU,1>) Gradient of Cost wrt input
struct CostGrad{
    q_MPC f_x;
    r_MPC f_u;

    void setZero()
    {
        f_x.setZero();
        f_u.setZero();
    }
};

/// @brief Hessian of Cost wrt state and input
/// @param f_xx (Eigen::Matrix<double,NX,NX>) Hessian of Cost wrt state
/// @param f_uu (Eigen::Matrix<double,NU,NU>) Hessian of Cost wrt input
/// @param f_xu (Eigen::Matrix<double,NX,NU>) Hessian of Cost wrt input
struct CostHess{
    Q_MPC f_xx;
    R_MPC f_uu;
    S_MPC f_xu;

    void setZero()
    {
        f_xx.setZero();
        f_uu.setZero();
        f_xu.setZero();
    }
};

/// @brief refernce X-Y-Z path position and its derivates wrt arc length (s)
/// @param x_ref (double) reference X position data
/// @param y_ref (double) reference Y position data
/// @param z_ref (double) reference Z position data
/// @param dx_ref (double) reference X'(s) position data
/// @param dy_ref (double) reference Y'(s) position data
/// @param dz_ref (double) reference Z'(s) position data
/// @param ddx_ref (double) reference X''(s) position data
/// @param ddy_ref (double) reference Y''(s) position data
/// @param ddz_ref (double) reference Z''(s) position data
struct TrackPoint{
    const double x_ref;
    const double y_ref;
    const double z_ref;
    const double dx_ref;
    const double dy_ref;
    const double dz_ref;
    const double ddx_ref;
    const double ddy_ref;
    const double ddz_ref;
};

/// @brief refernce Orientation and its derivates wrt arc length (s)
/// @param R_ref (Eigen::Matrix3d) reference R rotation matrix data
/// @param dR_ref (Eigen::Vector3d) reference R'(s) rotation matrix data
struct TrackOrienatation{
    const Eigen::Matrix3d R_ref;
    const Eigen::Vector3d dR_ref;
};

/// @brief error between reference and X-Y-Z position of the end-effector
/// @param contouring_error (Eigen::Vector3d) contouring error
/// @param lag_error (Eigen::Vector3d) lag error
/// @param d_contouring_error (Eigen::Matrix<double,3,NX>) derivatives of the contouring error with respect to state
/// @param d_lag_error (Eigen::Matrix<double,3,NX>) derivatives of the lag error with respect to state
struct ErrorInfo{
    const Eigen::Vector3d contouring_error;
    const Eigen::Vector3d lag_error;
    const Eigen::Matrix<double,3,NX> d_contouring_error;
    const Eigen::Matrix<double,3,NX> d_lag_error;
};

class Cost {
public:
    Cost(const PathToJson &path);
    Cost();

    /// @brief compute cost for contouring error, heading error, control input given current state
    /// @param track (ArcLengthSpline) reference track
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param obj (*double) exact value of cost
    /// @param grad (*CostGrad) gradient of cost
    /// @param hess (*CostHess) hessian of cost
    void getCost(const ArcLengthSpline &track,const State &x,const Input &u,const RobotData &rb,int k,
                 double* obj,CostGrad* grad,CostHess* hess);

private:
    /// @brief compute all the geometry information of the track at a given current arc length
    /// @param track (ArcLengthSpline) reference track
    /// @param x (State) current state
    /// @return (TrackPoint) reference X-Y-Z path position and its derivates wrt arc length (s)
    TrackPoint getRefPoint(const ArcLengthSpline &track,const State &x);

    /// @brief compute all the geometry information of the track at a given current arc length
    /// @param track (ArcLengthSpline) reference track
    /// @param x (State) current state
    /// @return (TrackPoint) reference Orientation and its derivates wrt arc length (s)
    TrackOrienatation getRefOrientation(const ArcLengthSpline &track,const State &x);

    /// @brief compute error between reference track and X-Y position of the car
    /// @param track (ArcLengthSpline) reference track
    /// @param x (State) current state
    /// @return (ErrorInfo) contouring and lag error and its derivatives wrt state
    ErrorInfo  getErrorInfo(const ArcLengthSpline &track,const State &x,const RobotData &rb);


    /// @brief compute contouring cost given current state
    /// @param track (ArcLengthSpline) reference track
    /// @param x (State) current state
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param obj (*double) exact value of contouring cost
    /// @param grad (*CostGrad) gradient of contouring cost
    /// @param hess (*CostHess) hessian of contouring cost
    void getContouringCost(const ArcLengthSpline &track,const State &x,const RobotData &rb,int k, 
                           double* obj,CostGrad* grad,CostHess* hess);

    /// @brief compute heading angle cost given current state
    /// @param track (ArcLengthSpline) reference track
    /// @param x (State) current state
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param obj (*double) exact value of heading cost
    /// @param grad (*CostGrad) gradient of heading cost
    /// @param grad (*CostHess) hessian of heading cost
    void getHeadingCost(const ArcLengthSpline &track,const State &x,const RobotData &rb,
                        double* obj,CostGrad* grad,CostHess* hess);

    /// @brief compute control input cost
    /// @param track (ArcLengthSpline) reference track
    /// @param u (Input) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param obj (*double) exact value for control input cost
    /// @param grad (*CostGrad) gradient of control input cost
    /// @param grad (*CostHess) hessian of control input cost
    void getInputCost(const ArcLengthSpline &track,const Input &u,const RobotData &rb,int k,
                      double* obj,CostGrad* grad,CostHess* hess);
    

    CostParam cost_param_;
    double contouring_cost_, lag_cost_, heading_cost_;
    Param param_;
};
}
#endif //MPCC_COST_H
