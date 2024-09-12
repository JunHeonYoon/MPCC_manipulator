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

#ifndef MPCC_MPC_H
#define MPCC_MPC_H

#include "config.h"
#include "types.h"
#include "Params/params.h"
#include "Spline/arc_length_spline.h"
#include "Model/model.h"
#include "Model/integrator.h"
#include "Cost/cost.h"
#include "Constraints/constraints.h"
#include "Constraints/bounds.h"

#include "Interfaces/solver_interface.h"
#include "Interfaces/osqp_interface.h"

#include <array>
#include <memory>
#include <ctime>
#include <ratio>
#include <chrono>
#include <vector>

namespace mpcc{
/// @brief output of MPC
/// @param u0 (Input) optimal control input
/// @param mpc_horizon (std::vector<OptVariables>) total horizon results (state and control input)
/// @param compute_time (ComputeTime) time to run MPC
struct MPCReturn {
    Input u0;
    std::vector<OptVariables> mpc_horizon;
    ComputeTime compute_time;
    void setZero()
    {
        u0.setZero();
        mpc_horizon.resize(N+1);
        for(size_t i=0;i<=N;i++) mpc_horizon[i].setZero();
        compute_time.setZero();
    }
};

class MPC {
public:
    MPC();
    MPC(double Ts,const PathToJson &path);
    MPC(double Ts,const PathToJson &path,const ParamValue &param_value);

    /// @brief run MPC by sqp given current state
    /// @param (MPCReturn) log for MPC; optimal control input, total horizon results, time to run MPC
    /// @param x0 (State) current state
    /// @param u0 (Input) current control input
    /// @return (bool) whether mpc is solved or not
    bool runMPC(MPCReturn &mpc_return, State &x0, Input &u0);

    /// @brief run MPC by sqp given current state
    /// @param (MPCReturn) log for MPC; optimal control input, total horizon results, time to run MPC
    /// @param x0 (State) current state
    /// @param u0 (Input) current control input
    /// @param voxel (std::vector<float>) Voxel Occupancy grid map
    /// @return (bool) whether mpc is solved or not
    bool runMPC_(MPCReturn &mpc_return, State &x0, Input &u0, const std::vector<float> &voxel);

    /// @brief set track given X-Y-Z-R path data
    /// @param X (Eigen::VectorXd) X path data
    /// @param Y (Eigen::VectorXd) Y path data
    /// @param Z (Eigen::VectorXd) Z path data
    /// @param R (std::vector<Eigen::Matrix3d>) R orientation data
    void setTrack(const Eigen::VectorXd &X, const Eigen::VectorXd &Y,const Eigen::VectorXd &Z,const std::vector<Eigen::Matrix3d> &R);

    /// @brief get total length of track
    /// @return (double) total length of track
    double getTrackLength();

    /// @brief set parameter value
    /// @param param_value (ParamValue) parameter value
    void setParam(const ParamValue &param_value);



    std::unique_ptr<RobotModel> robot_;

private:
    /// @brief unwrapping for initial variables which have phi(yaw) and arc length(s) 
    void unwrapInitialGuess();

    /// @brief to be warmstart, update initial variables for MPC
    /// @param x0 (State) solution of MPC before time step
    void updateInitialGuess(const State &x0);

    /// @brief generate new initial variables for MPC for the first
    /// @param x0 (State) current state
    void generateNewInitialGuess(const State &x0);

    /// @brief print parameter value
    /// @param param_value (ParamValue) parameter value
    void printParamValue(const ParamValue& param_value);

    ArcLengthSpline track_;
    bool valid_initial_guess_;
    std::vector<OptVariables> initial_guess_;
    const double Ts_;
    Integrator integrator_;
    Param param_;
    std::unique_ptr<SolverInterface> solver_interface_;
    PathToJson path_;
    unsigned int num_valid_guess_failed_;
};

}

#endif //MPCC_MPC_H
