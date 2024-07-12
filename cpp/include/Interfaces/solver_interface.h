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

#ifndef MPCC_SOLVER_INTERFACE_H
#define MPCC_SOLVER_INTERFACE_H

#include "config.h"
#include "types.h"
#include "Spline/arc_length_spline.h"
#include <array>

namespace mpcc{
struct OptVariables;
struct ComputeTime;
enum Status
{ 
    SOLVED,    
    MAX_ITER_EXCEEDED,
    QP_DualInfeasibleInaccurate,
    QP_PrimalInfeasibleInaccurate,
    QP_SolvedInaccurate,
    QP_MaxIterReached,
    QP_PrimalInfeasible,
    QP_DualInfeasible,
    Sigint, 
    INVALID_SETTINGS 
};

class SolverInterface {
    public:
        virtual void setTrack(const ArcLengthSpline track) = 0;
        virtual void setInitialGuess(const std::vector<OptVariables> &initial_guess) = 0;
        virtual std::vector<OptVariables> solveOCP(Status *status, ComputeTime *mpc_time) = 0;
        virtual ~SolverInterface(){std::cout << "Deleting Solver Interface" << std::endl;}
};
}

#endif //MPCC_SOLVER_INTERFACE_H
