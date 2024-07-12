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

#ifndef MPCC_BOUNDS_H
#define MPCC_BOUNDS_H

#include "config.h"
#include "types.h"
#include "Params/params.h"

namespace mpcc{
class Bounds {
public:
    Bounds();
    Bounds(BoundsParam bounds_param, Param param);

    /// @brief get lower bound for state
    /// @param x (State) current state
    /// @return (Eigen::VectorXd) lower bound for state
    Bounds_x getBoundsLX(const State &x) const;
    Bounds_x getBoundsLX() const;

    /// @brief get upper bound for state
    /// @param x (State) current state
    /// @param track_length (double) track length
    /// @return (Eigen::VectorXd) upper bound for state
    Bounds_x getBoundsUX(const State &x,const double &track_length) const;
    Bounds_x getBoundsUX() const;

    /// @brief get lower bound for control input
    /// @return (Eigen::VectorXd) lower bound for control input
    Bounds_u getBoundsLU() const;

    /// @brief get upper bound for control input
    /// @return (Eigen::VectorXd) upper bound for control input
    Bounds_u getBoundsUU() const;

private:

    Bounds_x u_bounds_x_;
    Bounds_x l_bounds_x_;

    Bounds_u u_bounds_u_;
    Bounds_u l_bounds_u_;

    Param param_;
};
}
#endif //MPCC_BOUNDS_H
