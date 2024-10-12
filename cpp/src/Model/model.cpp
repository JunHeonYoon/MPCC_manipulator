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

#include "Model/model.h"
namespace mpc{
Model::Model()
:Ts_(1.0)
{
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

Model::Model(double Ts,const PathToJson &path)
// :Ts_(Ts),param_(Param(path.param_path))
:Ts_(Ts)
{
}

StateVector Model::getF(const State &x,const Input &u) const
{
    StateVector f;
    f(0) = u.dq1;
    f(1) = u.dq2;
    f(2) = u.dq3;
    f(3) = u.dq4;
    f(4) = u.dq5;
    f(5) = u.dq6;
    f(6) = u.dq7;

    return f;
}

LinModelMatrix Model::getModelJacobian(const State &x, const Input &u) const
{
    // compute jacobian of the model

    // LinModelMatrix lin_model_c;
    A_MPC A_c = A_MPC::Zero();
    B_MPC B_c = B_MPC::Zero();
    g_MPC g_c = g_MPC::Zero();

    // Jacobians
    // Matrix A

    // Matrix B
    B_c.block(si_index.q1,si_index.dq1,PANDA_DOF,PANDA_DOF).setIdentity();

    return {A_c,B_c,g_c};
}

LinModelMatrix Model::discretizeModel(const LinModelMatrix &lin_model_c) const
{
    // https://en.wikipedia.org/wiki/Discretization
    // disctetize the continuous time linear model \dot x = A x + B u + g using ZHO
    Eigen::Matrix<double,NX+NU+1,NX+NU+1> temp = Eigen::Matrix<double,NX+NU+1,NX+NU+1>::Zero();
    // building matrix necessary for expm
    // temp = Ts*[A,B,g;zeros]
    temp.block<NX,NX>(0,0) = lin_model_c.A;
    temp.block<NX,NU>(0,NX) = lin_model_c.B;
    temp.block<NX,1>(0,NX+NU) = lin_model_c.g;
    temp = temp*Ts_;
    // take the matrix exponential of temp
    const Eigen::Matrix<double,NX+NU+1,NX+NU+1> temp_res = temp.exp();
    // extract dynamics out of big matrix
    // x_{k+1} = Ad x_k + Bd u_k + gd
    //temp_res = [Ad,Bd,gd;zeros]
    const A_MPC A_d = temp_res.block<NX,NX>(0,0);
    const B_MPC B_d = temp_res.block<NX,NU>(0,NX);
    const g_MPC g_d = temp_res.block<NX,1>(0,NX+NU);
    // const A_MPC A_d = (lin_model_c.A*Ts_).exp();
    // const B_MPC B_d = lin_model_c.A.inverse() * (A_d - Eigen::MatrixXd::Identity(NX,NX)) * lin_model_c.B;
    // const g_MPC g_d = lin_model_c.g * Ts_;

    return {A_d,B_d,g_d};
}

//LinModelMatrix Model::discretizeModel(const LinModelMatrix &lin_model_c) const
//{
//    // disctetize the continuous time linear model \dot x = A x + B u + g using ZHO
//    Eigen::Matrix<double,NX+NU+1,NX+NU+1> temp = Eigen::Matrix<double,NX+NU+1,NX+NU+1>::Zero();
//    // building matrix necessary for expm
//    // temp = Ts*[A,B,g;zeros]
//    temp.block<NX,NX>(0,0) = lin_model_c.A;
//    temp.block<NX,NU>(0,NX) = lin_model_c.B;
//    temp.block<NX,1>(0,NX+NU) = lin_model_c.g;
//    temp = temp*TS;
//    Eigen::Matrix<double,NX+NU+1,NX+NU+1> eye;
//    eye.setIdentity();
//    const Eigen::Matrix<double,NX+NU+1,NX+NU+1> temp_mult = temp * temp;
//
//    const Eigen::Matrix<double,NX+NU+1,NX+NU+1> temp_res = eye + temp + 1./2.0 * temp_mult + 1./6.0 * temp_mult * temp;
//
//    // x_{k+1} = Ad x_k + Bd u_k + gd
//    const A_MPC A_d = temp_res.block<NX,NX>(0,0);
//    const B_MPC B_d = temp_res.block<NX,NU>(0,NX);
//    const g_MPC g_d = temp_res.block<NX,1>(0,NX+NU);
//
//    return {A_d,B_d,g_d};
//
//}

LinModelMatrix Model::getLinModel(const State &x, const Input &u) const
{
    // compute linearized and discretized model
    const LinModelMatrix lin_model_c = getModelJacobian(x,u);
    // discretize the system
    return discretizeModel(lin_model_c);
}
}