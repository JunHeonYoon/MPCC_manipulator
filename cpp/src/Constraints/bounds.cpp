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

#include "Constraints/bounds.h"
namespace mpc{
Bounds::Bounds()
{
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

Bounds::Bounds(BoundsParam bounds_param,Param param):
param_(param)
{
    l_bounds_x_(0) = bounds_param.lower_state_bounds.q1_l;
    l_bounds_x_(1) = bounds_param.lower_state_bounds.q2_l;
    l_bounds_x_(2) = bounds_param.lower_state_bounds.q3_l;
    l_bounds_x_(3) = bounds_param.lower_state_bounds.q4_l;
    l_bounds_x_(4) = bounds_param.lower_state_bounds.q5_l;
    l_bounds_x_(5) = bounds_param.lower_state_bounds.q6_l;
    l_bounds_x_(6) = bounds_param.lower_state_bounds.q7_l;

    u_bounds_x_(0) = bounds_param.upper_state_bounds.q1_u;
    u_bounds_x_(1) = bounds_param.upper_state_bounds.q2_u;
    u_bounds_x_(2) = bounds_param.upper_state_bounds.q3_u;
    u_bounds_x_(3) = bounds_param.upper_state_bounds.q4_u;
    u_bounds_x_(4) = bounds_param.upper_state_bounds.q5_u;
    u_bounds_x_(5) = bounds_param.upper_state_bounds.q6_u;
    u_bounds_x_(6) = bounds_param.upper_state_bounds.q7_u;

    l_bounds_u_(0) = bounds_param.lower_input_bounds.dq1_l;
    l_bounds_u_(1) = bounds_param.lower_input_bounds.dq2_l;
    l_bounds_u_(2) = bounds_param.lower_input_bounds.dq3_l;
    l_bounds_u_(3) = bounds_param.lower_input_bounds.dq4_l;
    l_bounds_u_(4) = bounds_param.lower_input_bounds.dq5_l;
    l_bounds_u_(5) = bounds_param.lower_input_bounds.dq6_l;
    l_bounds_u_(6) = bounds_param.lower_input_bounds.dq7_l;

    u_bounds_u_(0) = bounds_param.upper_input_bounds.dq1_u;
    u_bounds_u_(1) = bounds_param.upper_input_bounds.dq2_u;
    u_bounds_u_(2) = bounds_param.upper_input_bounds.dq3_u;
    u_bounds_u_(3) = bounds_param.upper_input_bounds.dq4_u;
    u_bounds_u_(4) = bounds_param.upper_input_bounds.dq5_u;
    u_bounds_u_(5) = bounds_param.upper_input_bounds.dq6_u;
    u_bounds_u_(6) = bounds_param.upper_input_bounds.dq7_u;


    std::cout << "bounds initialized" << std::endl;
}

Bounds_x Bounds::getBoundsLX() const
{
    return  l_bounds_x_;
}

Bounds_x Bounds::getBoundsUX() const
{
    return  u_bounds_x_;
}

Bounds_u Bounds::getBoundsLU() const
{
    return  l_bounds_u_;
}

Bounds_u Bounds::getBoundsUU() const
{
    return  u_bounds_u_;
}

}