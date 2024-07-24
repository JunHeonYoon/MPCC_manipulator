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

#ifndef MPCC_TYPES_H
#define MPCC_TYPES_H

#include "config.h"
#include <map>
namespace mpcc{
/// @brief State of manipulator system
/// @param q1  (double) joint angle 
/// @param q2  (double) joint angle 
/// @param q3  (double) joint angle 
/// @param q4  (double) joint angle 
/// @param q5  (double) joint angle 
/// @param q6  (double) joint angle 
/// @param q7  (double) joint angle 
/// @param dq1 (double) velocity of joint angle
/// @param dq2 (double) velocity of joint angle
/// @param dq3 (double) velocity of joint angle
/// @param dq4 (double) velocity of joint angle
/// @param dq5 (double) velocity of joint angle
/// @param dq6 (double) velocity of joint angle
/// @param dq7 (double) velocity of joint angle
/// @param s   (double) path parameter, arc length 
/// @param vs  (double) velocity of path parameter 
struct State{ 
    double q1;  // joint angle
    double q2;  // joint angle
    double q3;  // joint angle 
    double q4;  // joint angle
    double q5;  // joint angle
    double q6;  // joint angle
    double q7;  // joint angle
    double dq1; // joint velocity
    double dq2; // joint velocity
    double dq3; // joint velocity 
    double dq4; // joint velocity
    double dq5; // joint velocity
    double dq6; // joint velocity
    double dq7; // joint velocity
    double s;   // path parameter (arc length)
    double vs;  // velocity of path parameter

    void setZero()
    {
        q1 = 0.0;
        q2 = 0.0;
        q3 = 0.0;
        q4 = 0.0;
        q5 = 0.0;
        q6 = 0.0;
        q7 = 0.0;
        dq1 = 0.0;
        dq2 = 0.0;
        dq3 = 0.0;
        dq4 = 0.0;
        dq5 = 0.0;
        dq6 = 0.0;
        dq7 = 0.0;
        s = 0.0;
        vs = 0.0;
    }
    /// @brief mapping s to [0, track_length]
    void unwrap(double track_length)
    {
        s = std::max(0.,std::min(track_length,s));
    }

    bool operator==(const State& other) const 
    {
        return q1 == other.q1 
            && q2 == other.q2
            && q3 == other.q3
            && q4 == other.q4
            && q5 == other.q5
            && q6 == other.q6
            && q7 == other.q7
            && dq1 == other.dq1
            && dq2 == other.dq2
            && dq3 == other.dq3
            && dq4 == other.dq4
            && dq5 == other.dq5
            && dq6 == other.dq6
            && dq7 == other.dq7
            && s == other.s
            && vs == other.vs;
    }
};

/// @brief Control input of manipulator system
/// @param ddq1 (double) acceleration of joint angle
/// @param ddq2 (double) acceleration of joint angle
/// @param ddq3 (double) acceleration of joint angle
/// @param ddq4 (double) acceleration of joint angle
/// @param ddq5 (double) acceleration of joint angle
/// @param ddq6 (double) acceleration of joint angle
/// @param ddq7 (double) acceleration of joint angle
/// @param dVs (double) change of velocity of path parameter 
struct Input{
    double ddq1;
    double ddq2;
    double ddq3;
    double ddq4;
    double ddq5;
    double ddq6;
    double ddq7;
    double dVs;

    void setZero()
    {
        ddq1 = 0.0;
        ddq2 = 0.0;
        ddq3 = 0.0;
        ddq4 = 0.0;
        ddq5 = 0.0;
        ddq6 = 0.0;
        ddq7 = 0.0;
        dVs = 0.0;
    }

    bool operator==(const Input& other) const 
    {
        return ddq1 == other.ddq1 
            && ddq2 == other.ddq2
            && ddq3 == other.ddq3
            && ddq4 == other.ddq4
            && ddq5 == other.ddq5
            && ddq6 == other.ddq6
            && ddq7 == other.ddq7
            && dVs == other.dVs;
    }
};

/// @brief path of JSON files
/// @param param_path         (std::string) path of model parameter 
/// @param cost_path          (std::string) path of cost parameter
/// @param bounds_path        (std::string) path of bound parameter 
/// @param track_path         (std::string) path of track 
/// @param normalization_path (std::string) path of normalization parameter 
/// @param sqo_path           (std::string) path of sqp parameter 
struct PathToJson{
    std::string param_path;
    std::string cost_path;
    std::string bounds_path;
    std::string track_path;
    std::string normalization_path;
    std::string sqp_path;
};

/// @brief Parameter value
/// @param param (std::map<std::string, double>) model parameter
/// @param cost (std::map<std::string, double>) cost parameter
/// @param bounds (std::map<std::string, double>) bounds parameter
/// @param track (std::map<std::string, double>) track parameter
/// @param normalization (std::map<std::string, double>) normalization parameter
/// @param sqp (std::map<std::string, double>) sqp parameter
struct ParamValue{
    std::map<std::string, double> param;
    std::map<std::string, double> cost;
    std::map<std::string, double> bounds;
    std::map<std::string, double> track;
    std::map<std::string, double> normalization;
    std::map<std::string, double> sqp;
};

typedef Eigen::Matrix<double,NX,1> StateVector;
typedef Eigen::Matrix<double,PANDA_DOF,1> JointVector;
typedef Eigen::Matrix<double,PANDA_DOF,1> dJointVector;
typedef Eigen::Matrix<double,PANDA_DOF,1> ddJointVector;
typedef Eigen::Matrix<double,NU,1> InputVector;

// x_(k+1) = Ax + Bu + g
typedef Eigen::Matrix<double,NX,NX> A_MPC; 
typedef Eigen::Matrix<double,NX,NU> B_MPC;
typedef Eigen::Matrix<double,NX,1> g_MPC;

typedef Eigen::Matrix<double,NX,NX> Q_MPC;
typedef Eigen::Matrix<double,NU,NU> R_MPC;
typedef Eigen::Matrix<double,NX,NU> S_MPC;

typedef Eigen::Matrix<double,NX,1> q_MPC;
typedef Eigen::Matrix<double,NU,1> r_MPC;

typedef Eigen::Matrix<double,NPC,NX> C_MPC;
typedef Eigen::Matrix<double,1,NX> C_i_MPC;
typedef Eigen::Matrix<double,NPC,NU> D_MPC;
typedef Eigen::Matrix<double,1,NU> D_i_MPC;
typedef Eigen::Matrix<double,NPC,1> d_MPC;

typedef Eigen::Matrix<double,NX,NX> TX_MPC;
typedef Eigen::Matrix<double,NU,NU> TU_MPC;

typedef Eigen::Matrix<double,NX,1> Bounds_x;
typedef Eigen::Matrix<double,NU,1> Bounds_u;

StateVector stateToVector(const State &x);
JointVector stateToJointVector(const State &x);
dJointVector stateTodJointVector(const State &x);
ddJointVector inputToddJointVector(const Input &u);
InputVector inputToVector(const Input &u);

State vectorToState(const StateVector &xk);
Input vectorToInput(const InputVector &uk);

State arrayToState(double *xk);
Input arrayToInput(double *uk);
}
#endif //MPCC_TYPES_H
