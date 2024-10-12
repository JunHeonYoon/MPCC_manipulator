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

#ifndef MPC_TRACK_H
#define MPC_TRACK_H

#include "config.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

namespace mpc {
//used namespace
using json = nlohmann::json;

/// @brief Tracking waypoints
/// @param X   (const Eigen::VectorXd) X waypoints
/// @param Y   (const Eigen::VectorXd) y waypoints
/// @param Z   (const Eigen::VectorXd) z waypoints
/// @param R   (const std::vector<Eigen::Matrix3d>) R for rotation matrix
struct TrackPos {
    // for position
    const Eigen::VectorXd X;
    const Eigen::VectorXd Y;
    const Eigen::VectorXd Z;

    // for orientation
    const std::vector<Eigen::Matrix3d> R;
};

class Track {
public:
    Track(std::string file);

    /// @brief get Track waypoints 
    /// @param init_position (Eigen::Vector3d) initial End-Effector position 
    /// @return (TrackPos) Tracking waypoints about x, y, z-axis and rotation matrix(not yet)
    TrackPos getTrack(Eigen::Vector3d init_position = Eigen::Vector3d::Zero());

private:
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    Eigen::VectorXd Z;
    std::vector<Eigen::Matrix3d> R;
};
};

#endif //MPCC_TRACK_H
