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

#ifndef MPC_COST_H
#define MPC_COST_H

#include "config.h"
#include "types.h"
#include "Spline/arc_length_spline.h"
#include "Model/robot_data.h"
#include <vector>

namespace mpc{

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

class Cost {
public:
    Cost(const PathToJson &path);
    Cost(const PathToJson &path,const ParamValue &param_value);
    Cost();

    /// @brief compute cost for contouring error, heading error, control input given current state
    /// @param ref_posi (Eigen::Vector3d) reference position
    /// @param ref_ori (Eigen::Matrix3d) reference orientation
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param obj (*double) exact value of cost
    /// @param grad (*CostGrad) gradient of cost
    /// @param hess (*CostHess) hessian of cost
    void getCost(const Eigen::Vector3d &ref_posi,const Eigen::Matrix3d &ref_ori,const State &x,const Input &u,const RobotData &rb,int k,
                 double* obj,CostGrad* grad,CostHess* hess);

private:
    /// @brief compute position error cost given current state
    /// @param ref_posi (Eigen::Vector3d) reference position
    /// @param x (State) current state
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param obj (*double) exact value of contouring cost
    /// @param grad (*CostGrad) gradient of contouring cost
    /// @param hess (*CostHess) hessian of contouring cost
    void getPositionCost(const Eigen::Vector3d &ref_posi,const State &x,const RobotData &rb,int k, 
                           double* obj,CostGrad* grad,CostHess* hess);

    /// @brief compute heading angle cost given current state
    /// @param ref_ori (Eigen::Matrix3d) reference orientation
    /// @param x (State) current state
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param obj (*double) exact value of heading cost
    /// @param grad (*CostGrad) gradient of heading cost
    /// @param grad (*CostHess) hessian of heading cost
    void getHeadingCost(const Eigen::Matrix3d &ref_ori,const State &x,const RobotData &rb,
                        double* obj,CostGrad* grad,CostHess* hess);

    /// @brief compute control input cost
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param obj (*double) exact value for control input cost
    /// @param grad (*CostGrad) gradient of control input cost
    /// @param grad (*CostHess) hessian of control input cost
    void getInputCost(const State &x,const Input &u,const RobotData &rb,int k,
                      double* obj,CostGrad* grad,CostHess* hess);
    

    CostParam cost_param_;
    double contouring_cost_, lag_cost_, heading_cost_;
    Param param_;
};
}
#endif //MPCC_COST_H
