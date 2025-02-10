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

#ifndef MPCC_TRACK_H
#define MPCC_TRACK_H

#include "config.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

namespace mpcc {
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

    /// @brief Set Path
    /// @param X (Eigen::VectorXd) x position
    /// @param Y (Eigen::VectorXd) y position
    /// @param Z (Eigen::VectorXd) z position
    /// @param R (std::vector<Eigen::Matrix3d>) rotation
    void setTrack(const Eigen::VectorXd& X, const Eigen::VectorXd& Y, const Eigen::VectorXd& Z, const std::vector<Eigen::Matrix3d>& R);

    /// @brief get Track waypoints 
    /// @param init_pose (Eigen::Matrix4d) initial End-Effector pose 
    /// @return (TrackPos) Tracking waypoints about x, y, z-axis and rotation matrix
    TrackPos getTrack(const Eigen::Matrix4d& init_pose);

private:
    /// @brief initialize Path w.r.t initial EE pose
    /// @param init_pose (Eigen::Matrix4d) initial End-Effector pose 
    void initTrack(const Eigen::Matrix4d& init_pose);

    Eigen::VectorXd X_;
    Eigen::VectorXd Y_;
    Eigen::VectorXd Z_;
    std::vector<Eigen::Matrix3d> R_;

    Eigen::Matrix3d enforceOrthogonality(const Eigen::Matrix3d &R);
    Eigen::Matrix3d normalizeDeterminant(const Eigen::Matrix3d &R);
    Eigen::Matrix3d correctRotationMatrix(const Eigen::Matrix3d &R);
};
};

#endif //MPCC_TRACK_H
