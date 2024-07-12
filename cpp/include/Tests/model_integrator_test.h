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

#ifndef MPCC_MODEL_INTEGRATOR_TEST_H
#define MPCC_MODEL_INTEGRATOR_TEST_H

#include "Model/model.h"
#include "Model/integrator.h"
#include "gtest/gtest.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

TEST(TestIntegrator, TestIntegrator)
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
    double Ts = 0.02;
    const mpcc::Integrator integrator = mpcc::Integrator(Ts,json_paths);

    // test integrator by comparing Euler forward to RK4
    // 3 differnet test points, hand picked, going straight and random

    //Integrator integrator;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Hand picked x and u
    mpcc::StateVector error1;
    mpcc::State xk1 = {0,0,0,2,0.1,-0.3,0.1,0.2,-0.1};
    mpcc::Input uk1 = {0.2,-0.1,0,-0.3,0.5,0.7,0,1};

    error1 = stateToVector(integrator.EF(xk1,uk1,Ts)) - stateToVector(integrator.RK4(xk1,uk1,Ts));
    std::cout << "hand picked point RK4 - EF error = " << error1.norm() << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // x and u corresponding to going straight with 0.1 rad/s at init configuration
    mpcc::StateVector error2;
    mpcc::State xk2 = {0, 0, 0, -1.0471, 0, 1.0471, 0.7854, 0, 0};
    mpcc::Input uk2 = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,1};

    error2 = stateToVector(integrator.EF(xk2,uk2,Ts)) - stateToVector(integrator.RK4(xk2,uk2,Ts));
    std::cout << "straight RK4 - EF error = " << error2.norm() << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // random x and u
    mpcc::StateVector error3;
    mpcc::StateVector xkr = mpcc::StateVector::Random();
    mpcc::InputVector ukr = mpcc::InputVector::Random();
    mpcc::State xk3 = mpcc::vectorToState(xkr);
    mpcc::Input uk3 = mpcc::vectorToInput(ukr);

    error3 = stateToVector(integrator.EF(xk3,uk3,Ts)) - stateToVector(integrator.RK4(xk3,uk3,Ts));
    std::cout << "random RK4 - EF error = " <<  error3.norm() << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // test how good fit is
    EXPECT_TRUE((error1.norm()/10.0 <= 0.3 && error2.norm()/10.0 <= 0.3 && error3.norm()/10.0 <= 0.3 ));

}

TEST(TestIntegrator, TestLinModel)
{
    // test Liniear model by comparing it to RK4
    // 3 differnet test cases, hand picked, going straight and test how good linear model generalizes
    std::ifstream iConfig(mpcc::pkg_path + "Params/config.json");
    json jsonConfig;
    iConfig >> jsonConfig;

    mpcc::PathToJson json_paths {mpcc::pkg_path + std::string(jsonConfig["model_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["cost_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["bounds_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["track_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["normalization_path"]),
                                 mpcc::pkg_path + std::string(jsonConfig["sqp_path"])};
    double Ts = 0.02;
    const mpcc::Integrator integrator = mpcc::Integrator(0.02,json_paths);
    const mpcc::Model model = mpcc::Model(0.02,json_paths);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Hand picked x and u
    mpcc::StateVector error1;
    mpcc::State xk1 = {0,0,0,2,0.1,-0.3,0.1,0.2,-0.1};
    mpcc::Input uk1 = {0.2,-0.1,0,-0.3,0.5,0.7,0,1};
    mpcc::StateVector xk1_vec = mpcc::stateToVector(xk1);
    mpcc::InputVector uk1_vec = mpcc::inputToVector(uk1);

    const mpcc::LinModelMatrix lin_model_d1 = model.getLinModel(xk1,uk1);

    error1 = (lin_model_d1.A*xk1_vec + lin_model_d1.B*uk1_vec + lin_model_d1.g)  - stateToVector(integrator.RK4(xk1,uk1,Ts));
    std::cout << "hand picked point RK4 - lin error = " << error1.norm() << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // x and u corresponding to going straight with 0.1 rad/s at init configuration
    mpcc::StateVector error2;
    mpcc::State xk2 = {0, 0, 0, -1.0471, 0, 1.0471, 0.7854, 0, 0};
    mpcc::Input uk2 = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,1};
    mpcc::StateVector xk2_vec = mpcc::stateToVector(xk2);
    mpcc::InputVector uk2_vec = mpcc::inputToVector(uk2);

    const mpcc::LinModelMatrix lin_model_d2 = model.getLinModel(xk2,uk2);

    error2 = (lin_model_d2.A*xk2_vec + lin_model_d2.B*uk2_vec + lin_model_d2.g)  - stateToVector(integrator.RK4(xk2,uk2,Ts));
    std::cout << "straight RK4 - lin error = " << error2.norm() << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // generalization test
    // perturbe xk1 slightly, however us the model linearized around xk1 and uk1
    mpcc::StateVector error3;
    mpcc::State xk3;
    // xk3 is slightly perturbed version of xk1
    xk3 = xk1;
    xk3.q3 += 0.2;  //q3
    xk3.q6 += 0.05; //q6
    xk3.q7 += 0.8;  //q7

    mpcc::Input uk3;
    uk3 = uk1;
    //still linearize around xk1 and uk1
    mpcc::StateVector xk3_vec = stateToVector(xk3);
    mpcc::InputVector uk3_vec = inputToVector(uk3);
    error3 = (lin_model_d1.A*xk3_vec + lin_model_d1.B*uk3_vec + lin_model_d1.g)  - stateToVector(integrator.RK4(xk3,uk3,Ts));
//    std::cout << error3 << std::endl;
    std::cout << "generalization test RK4 - lin error = " << error3.norm() << std::endl;

    EXPECT_TRUE((error1.norm()/10.0 <= 0.03 && error2.norm()/10.0 <= 0.03));
}
#endif //MPCC_MODEL_INTEGRATOR_TEST_H
