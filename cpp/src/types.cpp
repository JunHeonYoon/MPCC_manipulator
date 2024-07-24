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

#include "types.h"
namespace mpcc{

StateVector stateToVector(const State &x)
{
    StateVector xk;
    xk(0) = x.q1;
    xk(1) = x.q2;
    xk(2) = x.q3;
    xk(3) = x.q4;
    xk(4) = x.q5;
    xk(5) = x.q6;
    xk(6) = x.q7;
    xk(7) = x.dq1;
    xk(8) = x.dq2;
    xk(9) = x.dq3;
    xk(10) = x.dq4;
    xk(11) = x.dq5;
    xk(12) = x.dq6;
    xk(13) = x.dq7;
    xk(14) = x.s;
    xk(15) = x.vs;
    return xk;
}

JointVector stateToJointVector(const State &x)
{
    JointVector xk;
    xk(0) = x.q1;
    xk(1) = x.q2;
    xk(2) = x.q3;
    xk(3) = x.q4;
    xk(4) = x.q5;
    xk(5) = x.q6;
    xk(6) = x.q7;
    return xk;
}

dJointVector stateTodJointVector(const State &x)
{
    dJointVector xk;
    xk(0) = x.dq1;
    xk(1) = x.dq2;
    xk(2) = x.dq3;
    xk(3) = x.dq4;
    xk(4) = x.dq5;
    xk(5) = x.dq6;
    xk(6) = x.dq7;
    return xk;
}

ddJointVector inputToddJointVector(const Input &u)
{
    ddJointVector uk;
    uk(0) = u.ddq1;
    uk(1) = u.ddq2;
    uk(2) = u.ddq3;
    uk(3) = u.ddq4;
    uk(4) = u.ddq5;
    uk(5) = u.ddq6;
    uk(6) = u.ddq7;
    return uk;
}

InputVector inputToVector(const Input &u)
{
    InputVector uk;
    uk(0) = u.ddq1;
    uk(1) = u.ddq2;
    uk(2) = u.ddq3;
    uk(3) = u.ddq4;
    uk(4) = u.ddq5;
    uk(5) = u.ddq6;
    uk(6) = u.ddq7;
    uk(7) = u.dVs;
    return uk;
}

State vectorToState(const StateVector &xk)
{
    State x;
    x.q1 = xk(0);
    x.q2 = xk(1);
    x.q3 = xk(2);
    x.q4 = xk(3);
    x.q5 = xk(4);
    x.q6 = xk(5);
    x.q7 = xk(6);
    x.dq1 = xk(7);
    x.dq2 = xk(8);
    x.dq3 = xk(9);
    x.dq4 = xk(10);
    x.dq5 = xk(11);
    x.dq6 = xk(12);
    x.dq7 = xk(13);
    x.s  = xk(14);
    x.vs = xk(15);

    return x;
}

Input vectorToInput(const InputVector &uk)
{
    Input u;
    u.ddq1 = uk(0);
    u.ddq2 = uk(1);
    u.ddq3 = uk(2);
    u.ddq4 = uk(3);
    u.ddq5 = uk(4);
    u.ddq6 = uk(5);
    u.ddq7 = uk(6);
    u.dVs = uk(7);

    return u;
}

State arrayToState(double *xk)
{
    State x;
    x.q1 = xk[0];
    x.q2 = xk[1];
    x.q3 = xk[2];
    x.q4 = xk[3];
    x.q5 = xk[4];
    x.q6 = xk[5];
    x.q7 = xk[6];
    x.dq1 = xk[7];
    x.dq2 = xk[8];
    x.dq3 = xk[9];
    x.dq4 = xk[10];
    x.dq5 = xk[11];
    x.dq6 = xk[12];
    x.dq7 = xk[13];
    x.s  = xk[14];
    x.vs = xk[15];

    return x;
}

Input arrayToInput(double *uk)
{
    Input u;
    u.ddq1 = uk[0];
    u.ddq2 = uk[1];
    u.ddq3 = uk[2];
    u.ddq4 = uk[3];
    u.ddq5 = uk[4];
    u.ddq6 = uk[5];
    u.ddq7 = uk[6];
    u.dVs = uk[7];

    return u;
}
}