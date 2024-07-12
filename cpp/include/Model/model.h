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

#ifndef MPCC_MODEL_H
#define MPCC_MODEL_H

#include "config.h"
#include "types.h"
#include "Params/params.h"
namespace mpcc{

/// @brief Linear model of car: x_(k+1) = A * x_k + B * u_k + g
/// @param A (Eigen::Matrix<double, NX, NX>) Linear part of state
/// @param B (Eigen::Matrix<double, NX, NU>) Linear part of control input
/// @param g (Eigen::Matrix<double, NX, 1>) Constant part
struct LinModelMatrix {
    A_MPC A;
    B_MPC B;
    g_MPC g;
};


class Model {
public:
    Model();
    Model(double Ts,const PathToJson &path);

    /// @brief compute derivative of State wrt time (x_dot) given current state and control input
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @return (Eigen::VectorXd) derivative of State
    StateVector getF(const State &x,const Input &u) const;

    /// @brief compute Discretized Linear model given current state and control input
    /// @param x (State) Current state
    /// @param u (Input) Current control input
    /// @return (LinModelMatrix) Discrete Linear model (A, B, g)
    LinModelMatrix getLinModel(const State &x, const Input &u) const;

private:
    /// @brief compute Linear Model given current state and control input
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @return (LinModelMatrix) Linear model (A, B, g)
    LinModelMatrix getModelJacobian(const State &x, const Input &u) const;

    /// @brief compute Discretized Linear model from continuous Linear model
    /// @param lin_model_c (LinModelMatrix) Continuous  Linear model
    /// @return (LinModelMatrix) Discrete Linear model (A, B, g)
    LinModelMatrix discretizeModel(const LinModelMatrix &lin_model_c) const;

    // Param param_;
    const double Ts_;
};
}
#endif //MPCC_MODEL_H
