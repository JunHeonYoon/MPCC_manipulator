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

#ifndef MPCC_CONFIG_H
#define MPCC_CONFIG_H

#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace mpcc{

static const int PANDA_DOF = 7;

static const int NX = 9;  // [joint angle, s(path param), vs]
static const int NU = 8;  // [djoint angle, dVs]

// static const int NPC = 2 + 9; // number of polytopic constraints: Self collision, Singularity, Env collision
static const int NPC = 1;

static constexpr int N = 10;
static constexpr double INF = 1E30;
static constexpr int N_SPLINE = 100;

/// @brief Index of State, Control input and soft constraints
struct StateInputIndex{
    // Index of State
    int q1 = 0;
    int q2 = 1;
    int q3 = 2;
    int q4 = 3;
    int q5 = 4;
    int q6 = 5;
    int q7 = 6;
    int s  = 7;
    int vs = 8;

    // Index of control input
    int dq1 = 0;
    int dq2 = 1;
    int dq3 = 2;
    int dq4 = 3;
    int dq5 = 4;
    int dq6 = 5;
    int dq7 = 6;
    int dVs = 7;

    // Index of constraints
    int con_sing = 0;     // singularity
    int con_selcol = 1;   // self collision 
    int con_envcol1 = 2;  // env collision (link 0)
    int con_envcol2 = 3;  // env collision (link 1)
    int con_envcol3 = 4;  // env collision (link 2)
    int con_envcol4 = 5;  // env collision (link 3)
    int con_envcol5 = 6;  // env collision (link 4)
    int con_envcol6 = 7;  // env collision (link 5)
    int con_envcol7 = 8;  // env collision (link 6)
    int con_envcol8 = 9;  // env collision (link 7)
    int con_envcol9 = 10; // env collision (hand)
};

static const StateInputIndex si_index;

static const std::string pkg_path =  std::string(BUILD_DIRECTORY) + "/";


}
#endif //MPCC_CONFIG_H
