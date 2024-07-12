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

#include "Params/track.h"
namespace mpcc{
Track::Track(std::string file) 
{
    /////////////////////////////////////////////////////
    // Loading Model and Constraint Parameters //////////
    /////////////////////////////////////////////////////
    std::ifstream iTrack(file);
    json jsonTrack;
    iTrack >> jsonTrack;
    
    // Model Parameters
    std::vector<double> x = jsonTrack["X"];
    X = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
    std::vector<double> y = jsonTrack["Y"];
    Y = Eigen::Map<Eigen::VectorXd>(y.data(), y.size());
    std::vector<double> z = jsonTrack["Z"];
    Z = Eigen::Map<Eigen::VectorXd>(z.data(), z.size());


    std::vector<double> quat_x = jsonTrack["quat_X"];
    std::vector<double> quat_y = jsonTrack["quat_Y"];
    std::vector<double> quat_z = jsonTrack["quat_Z"];
    std::vector<double> quat_w = jsonTrack["quat_W"];


    R.resize(quat_x.size());

    for(size_t i=0;i<quat_x.size();i++)
    {
        Eigen::Quaterniond q;
        q.x() = quat_x[i];
        q.y() = quat_y[i];
        q.z() = quat_z[i];
        q.w() = quat_w[i];
        R[i] = q.normalized().toRotationMatrix();
    }
}

TrackPos Track::getTrack(Eigen::Vector3d init_position)
{
    X = X.array() - X(0) + init_position(0);
    Y = Y.array() - Y(0) + init_position(1);
    Z = Z.array() - Z(0) + init_position(2);
    return {Eigen::Map<Eigen::VectorXd>(X.data(), X.size()), 
            Eigen::Map<Eigen::VectorXd>(Y.data(), Y.size()),
            Eigen::Map<Eigen::VectorXd>(Z.data(), Z.size()),
            R
            };
}
}
