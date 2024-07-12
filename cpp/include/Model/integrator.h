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

#ifndef MPCC_INTEGRATOR_H
#define MPCC_INTEGRATOR_H

#include "config.h"
#include "model.h"
#include "types.h"

namespace mpcc{
class Integrator {
public:
    Integrator();
    Integrator(double Ts, const PathToJson &path);

    /// @brief compute very next state given current state, control input and time by RK4
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @param ts (double) time step size
    /// @return (State) next state
    State RK4(const State &x, const Input &u,double ts) const;

    /// @brief compute very next state given current state, control input and time by Euler-Forward
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @param ts (double) time step size
    /// @return (State) next state
    State EF(const State &x, const Input &u,double ts) const;

    /// @brief compute next state given current state, control input and time by RK4
    /// @param x (State) current state
    /// @param u (Input) current control input
    /// @param ts (double) time step size
    /// @return (State) next state
    State simTimeStep(const State &x, const Input &u,double ts) const;


private:
    const double fine_time_step_ = 0.001;

    Model model_;
};
}
#endif //MPCC_INTEGRATOR_H
