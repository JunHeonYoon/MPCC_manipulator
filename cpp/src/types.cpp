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
    xk(7) = x.q8;
    xk(8) = x.q9;
    xk(9) = x.q10;
    xk(10) = x.q11;
    xk(11) = x.s;
    xk(12) = x.vs;
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
    xk(7) = x.q8;
    xk(8) = x.q9;
    xk(9) = x.q10;
    xk(10) = x.q11;
    return xk;
}

dJointVector inputTodJointVector(const Input &u)
{
    dJointVector uk;
    uk(0) = u.dq1;
    uk(1) = u.dq2;
    uk(2) = u.dq3;
    uk(3) = u.dq4;
    uk(4) = u.dq5;
    uk(5) = u.dq6;
    uk(6) = u.dq7;
    uk(7) = u.dq8;
    uk(8) = u.dq9;
    uk(9) = u.dq10;
    uk(10) = u.dq11;
    return uk;
}

InputVector inputToVector(const Input &u)
{
    InputVector uk;
    uk(0) = u.dq1;
    uk(1) = u.dq2;
    uk(2) = u.dq3;
    uk(3) = u.dq4;
    uk(4) = u.dq5;
    uk(5) = u.dq6;
    uk(6) = u.dq7;
    uk(7) = u.dq8;
    uk(8) = u.dq9;
    uk(9) = u.dq10;
    uk(10) = u.dq11;
    uk(11) = u.dVs;
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
    x.q8 = xk(7);
    x.q9 = xk(8);
    x.q10 = xk(9);
    x.q11 = xk(10);
    x.s  = xk(11);
    x.vs = xk(12);

    return x;
}

Input vectorToInput(const InputVector &uk)
{
    Input u;
    u.dq1 = uk(0);
    u.dq2 = uk(1);
    u.dq3 = uk(2);
    u.dq4 = uk(3);
    u.dq5 = uk(4);
    u.dq6 = uk(5);
    u.dq7 = uk(6);
    u.dq8 = uk(7);
    u.dq9 = uk(8);
    u.dq10 = uk(9);
    u.dq11 = uk(10);
    u.dVs = uk(11);

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
    x.q8 = xk[7];
    x.q9 = xk[8];
    x.q10 = xk[9];
    x.q11 = xk[10];
    x.s  = xk[11];
    x.vs = xk[12];

    return x;
}

Input arrayToInput(double *uk)
{
    Input u;
    u.dq1 = uk[0];
    u.dq2 = uk[1];
    u.dq3 = uk[2];
    u.dq4 = uk[3];
    u.dq5 = uk[4];
    u.dq6 = uk[5];
    u.dq7 = uk[6];
    u.dq8 = uk[7];
    u.dq9 = uk[8];
    u.dq10 = uk[9];
    u.dq11 = uk[10];
    u.dVs = uk[11];

    return u;
}
}