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
#ifndef MPCC_COST_TEST_H
#define MPCC_COST_TEST_H

#include "Cost/cost.h"
#include "constraints_test.h"
#include "Constraints/bounds.h"
#include "Model/robot_data.h"
#include "gtest/gtest.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

TEST(TestCost, TestSPD)
{
    std::ifstream iConfig(mpcc::pkg_path + "Params/config.json");
    json jsonConfig;
    iConfig >> jsonConfig;

    mpcc::PathToJson json_paths {mpcc::pkg_path + std::string(jsonConfig["model_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["cost_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["bounds_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["track_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["normalization_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["sqp_path"])};
    mpcc::Bounds bound = mpcc::Bounds(mpcc::BoundsParam(json_paths.bounds_path), mpcc::Param(json_paths.param_path));
    mpcc::Cost cost = mpcc::Cost(json_paths);
    std::unique_ptr<mpcc::RobotModel> robot = std::make_unique<mpcc::RobotModel>();
    std::unique_ptr<mpcc::SelCollNNmodel> selcolNN = std::make_unique<mpcc::SelCollNNmodel>();
    mpcc::ArcLengthSpline track = mpcc::ArcLengthSpline(json_paths);

    Eigen::Vector3d sel_col_n_hidden;
    sel_col_n_hidden << 128, 64, 32;
    selcolNN->setNeuralNetwork(mpcc::PANDA_DOF, 1, sel_col_n_hidden, true);

    genRoundTrack(track);

    mpcc::Bounds_x x_LB = bound.getBoundsLX();
    mpcc::Bounds_x x_UB = bound.getBoundsUX();
    mpcc::Bounds_x x_range = x_UB - x_LB;
    mpcc::Bounds_u u_LB = bound.getBoundsLU();
    mpcc::Bounds_u u_UB = bound.getBoundsUU();
    mpcc::Bounds_u u_range = u_UB - u_LB;

    mpcc::StateVector xk_vec;
    xk_vec = mpcc::StateVector::Random();
    xk_vec = (xk_vec + mpcc::StateVector::Ones()) / 2;
    xk_vec = (xk_vec.array() * x_range.array()).matrix() + x_LB; 
    mpcc::State xk = mpcc::vectorToState(xk_vec);

    mpcc::InputVector uk_vec;
    uk_vec = mpcc::InputVector::Random();
    uk_vec = (uk_vec + mpcc::InputVector::Ones()) / 2;
    uk_vec = (uk_vec.array() * u_range.array()).matrix() + u_LB; 
    mpcc::Input uk = mpcc::vectorToInput(uk_vec);

    mpcc::RobotData rbk;
    rbk.update(xk_vec.head(mpcc::PANDA_DOF),robot,selcolNN);

    // calculate cost matrix
    double temp_obj;
    mpcc::CostGrad cost_grad;
    mpcc::CostHess cost_hess;
    cost.getCost(track,xk,uk,rbk,1,&temp_obj,&cost_grad,&cost_hess);

    bool is_symQ = (cost_hess.f_xx.transpose() - cost_hess.f_xx).norm() < 1e-5;
    bool is_symR = (cost_hess.f_uu.transpose() - cost_hess.f_uu).norm() < 1e-5;
    std::cout<< "Is symmetric(Q, R): " << is_symQ << is_symR  << std::endl;

    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverQ, eigensolverR;
    eigensolverQ.compute(cost_hess.f_xx); 
    eigensolverR.compute(cost_hess.f_uu); 

    Eigen::VectorXd eigen_valuesQ = eigensolverQ.eigenvalues().real();
    Eigen::VectorXd eigen_valuesR = eigensolverR.eigenvalues().real();

    bool eigen_positiveQ = eigen_valuesQ.minCoeff() > 0;
    bool eigen_positiveR = eigen_valuesR.minCoeff() > 0;

    bool is_spdQ = is_symQ && eigen_positiveQ;
    bool is_spdR = is_symR && eigen_positiveR;

    std::cout<< "Is semi positive definite(Q, R): " << is_spdQ << is_spdR << std::endl;
    if(!eigen_positiveQ) std::cout << "min eigenvalue Q: " << eigen_valuesQ.minCoeff() << std::endl;
    if(!eigen_positiveR) std::cout << "min eigenvalue R: " << eigen_valuesR.minCoeff() << std::endl;

    
    EXPECT_TRUE(is_symQ && is_symR && is_spdQ && is_spdR);
}

TEST(TestCost, TestLinearization)
{
    std::ifstream iConfig(mpcc::pkg_path + "Params/config.json");
    json jsonConfig;
    iConfig >> jsonConfig;

    mpcc::PathToJson json_paths {mpcc::pkg_path + std::string(jsonConfig["model_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["cost_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["bounds_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["track_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["normalization_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["sqp_path"])};
    mpcc::Bounds bound = mpcc::Bounds(mpcc::BoundsParam(json_paths.bounds_path), mpcc::Param(json_paths.param_path));
    std::unique_ptr<mpcc::RobotModel> robot = std::make_unique<mpcc::RobotModel>();
    std::unique_ptr<mpcc::SelCollNNmodel> selcolNN = std::make_unique<mpcc::SelCollNNmodel>();
    mpcc::Cost cost = mpcc::Cost(json_paths);
    mpcc::CostParam cost_param = mpcc::CostParam(json_paths.cost_path);
    mpcc::Param param = mpcc::Param(json_paths.param_path);
    mpcc::ArcLengthSpline track = mpcc::ArcLengthSpline(json_paths);

    genRoundTrack(track);

    Eigen::Vector3d sel_col_n_hidden;
    sel_col_n_hidden << 128, 64, 32;
    selcolNN->setNeuralNetwork(mpcc::PANDA_DOF, 1, sel_col_n_hidden, true);

    mpcc::Bounds_x x_LB = bound.getBoundsLX();
    mpcc::Bounds_x x_UB = bound.getBoundsUX();
    mpcc::Bounds_x x_range = x_UB - x_LB;
    mpcc::Bounds_u u_LB = bound.getBoundsLU();
    mpcc::Bounds_u u_UB = bound.getBoundsUU();
    mpcc::Bounds_u u_range = u_UB - u_LB;

    mpcc::StateVector xk_vec, d_xk_vec, xk1_vec;
    xk_vec = mpcc::StateVector::Random();
    xk_vec = (xk_vec + mpcc::StateVector::Ones()) / 2;
    xk_vec = (xk_vec.array() * x_range.array()).matrix() + x_LB; 
    d_xk_vec = mpcc::StateVector::Constant(0.01);
    xk1_vec = xk_vec + d_xk_vec;

    mpcc::InputVector uk_vec, d_uk_vec, uk1_vec;
    uk_vec = mpcc::InputVector::Random();
    uk_vec = (uk_vec + mpcc::InputVector::Ones()) / 2;
    uk_vec = (uk_vec.array() * u_range.array()).matrix() + u_LB; 
    d_uk_vec = mpcc::InputVector::Constant(0.01);
    uk1_vec = uk_vec + d_uk_vec;

    mpcc::State xk = mpcc::vectorToState(xk_vec);
    mpcc::State xk1 = mpcc::vectorToState(xk1_vec);
    mpcc::Input uk = mpcc::vectorToInput(uk_vec);
    mpcc::Input uk1 = mpcc::vectorToInput(uk1_vec);

    mpcc::RobotData rbk, rbk1;
    rbk.update(xk_vec.head(mpcc::PANDA_DOF),robot,selcolNN);
    rbk1.update(xk1_vec.head(mpcc::PANDA_DOF),robot,selcolNN);

    double obj, obj1;
    mpcc::CostGrad cost_grad;
    mpcc::CostHess cost_hess;

    // caluculate real cost at xk, uk
    cost.getCost(track,xk,uk,rbk,1,&obj,&cost_grad,&cost_hess);
    cost.getCost(track,xk1,uk1,rbk1,1,&obj1,NULL,NULL);


    // calculate linearized cost at xk, uk
    double obj_lin1 = (0.5*d_xk_vec.transpose()*cost_hess.f_xx*d_xk_vec + 
                       0.5*d_uk_vec.transpose()*cost_hess.f_uu*d_uk_vec + 
                       d_xk_vec.transpose()*cost_hess.f_xu*d_uk_vec +  
                       cost_grad.f_x.transpose()*d_xk_vec + 
                       cost_grad.f_u.transpose()*d_uk_vec).value() + obj;

    
    std::cout << "X0\t: " << xk_vec.transpose()  <<"\t||\t U0\t:" << uk_vec.transpose() << std::endl;
    std::cout << "X1\t: " << xk1_vec.transpose() <<"\t||\t U1\t:" << uk1_vec.transpose() << std::endl;
    std::cout << "real cost on X0     : " << obj << std::endl;
    std::cout << "real cost on X0 + dX: " << obj1 << std::endl;
    std::cout << "lin  cost on X0 + dX: " << obj_lin1 << std::endl;
    std::cout << "Error[%]: " << fabs((obj1 - obj_lin1) / obj1)*100 <<std::endl;

    EXPECT_TRUE(fabs((obj1 - obj_lin1) / obj1) <= 1e-2);
}

#endif //MPCC_COST_TEST_H
