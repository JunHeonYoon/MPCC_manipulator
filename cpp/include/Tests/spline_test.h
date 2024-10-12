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

#ifndef MPC_SPLINE_TEST_H
#define MPC_SPLINE_TEST_H


#include "config.h"
#include "Spline/cubic_spline.h"
#include "Spline/arc_length_spline.h"
#include "Model/robot_model.h"
#include "gtest/gtest.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
using json = nlohmann::json;

TEST(TestSpilne, TestSpline)
{
    //Fit spline to a cos and test if the spine interpolation is accurate
    //Also test if first and second derivatives are accurate

    mpc::CubicSpline spline;

    Eigen::VectorXd x;
    Eigen::VectorXd y;

    int NT = 50;    //number of "training" points
    int NV = 100;   //number of validation points
    x.setLinSpaced(NT,0,M_PI);
    y.setZero(NT,1);

    //set y ot cos(x)
    for(int i=0;i<x.size();i++){
        y(i) = std::cos(x(i));
    }
    //give data to spline class and compute spline parameters
    spline.genSpline(x,y,true);

    //spline test outputs and true outputs -> initialize to zero
    Eigen::VectorXd xt;
    Eigen::VectorXd yt,ytt;
    Eigen::VectorXd dyt,dytt;
    Eigen::VectorXd ddyt,ddytt;

    xt.setLinSpaced(NV,0,M_PI); //test x points
    yt.setZero(NV,1);
    ytt.setZero(NV,1);

    dyt.setZero(NV,1);
    dytt.setZero(NV,1);

    ddyt.setZero(NV,1);
    ddytt.setZero(NV,1);

    //compute spline outputs and true outputs given the y = cos(x)
    for(int i=0;i<xt.size();i++){
        yt(i) = spline.getPoint(xt(i));
        ytt(i) = std::cos(xt(i));

        dyt(i) = spline.getDerivative(xt(i));
        dytt(i) = -std::sin(xt(i));

        ddyt(i) = spline.getSecondDerivative(xt(i));
        ddytt(i) = -std::cos(xt(i));
    }

    // output how good the fit is
    std::cout << "mean fit error: " << (yt-ytt).norm()/NV << std::endl;

    std::cout << "mean fit derivative error: " << (dyt-dytt).norm()/NV << std::endl;

    std::cout << "mean fit second derivative error: " << (ddyt-ddytt).norm()/NV << std::endl;

    //return true if fit is good enough
    EXPECT_TRUE((yt-ytt).norm()/NV <= 1e-4 && (dyt-dytt).norm()/NV <= 1e-3 && (ddyt-ddytt).norm()/NV <= 1e-1);
}

TEST(TestSpline, TestSplineRot)
{
    mpc::CubicSplineRot spline_rot;

    // generate sample rotation
    Eigen::VectorXd x(7);
    std::vector<Eigen::Matrix3d> R;
    Eigen::MatrixXd rpy;
    x << 0,1,2,3,4,5,6;
    R.resize(x.size());
    rpy.resize(3,x.size()); rpy.setZero();
    for(size_t i=1;i<R.size();i++)
    {
        if(i<4) 
        {
            rpy.col(i)(0) = rpy.col(i-1)(0) + 60;
            rpy.col(i)(1) = rpy.col(i-1)(1) + 0;
            rpy.col(i)(2) = rpy.col(i-1)(2) + 0;
        }
        else
        {
            rpy.col(i)(0) = rpy.col(i-1)(0) - 60;
            rpy.col(i)(1) = rpy.col(i-1)(1) - 0;
            rpy.col(i)(2) = rpy.col(i-1)(2) - 0;
        } 
    }

    for(size_t i=0;i<R.size();i++)
    {   
        R[i] = (Eigen::AngleAxisd(rpy(2,i)*M_PI/180, Vector3d::UnitZ())
              * Eigen::AngleAxisd(rpy(1,i)*M_PI/180, Vector3d::UnitY())
              * Eigen::AngleAxisd(rpy(0,i)*M_PI/180, Vector3d::UnitX())).matrix();
    }
    std::cout<<"RPY: "<<std::endl;
    std::cout<<x.transpose() <<std::endl;
    std::cout<<rpy <<std::endl;

    //give data to spline class and compute spline parameters
    spline_rot.genSpline(x,R,false);

    // get splined rotation
    Eigen::VectorXd xt;
    std::vector<Eigen::Matrix3d> Rt;
    Eigen::MatrixXd rpy_t;
    double n_sample =  (x.size()-1)*3+x.size();
    xt.setLinSpaced(n_sample,x(0),x(x.size()-1));
    Rt.resize(n_sample);
    rpy_t.resize(3,n_sample);
    for(size_t i=0;i<n_sample;i++)
    {
        Rt[i] = spline_rot.getPoint(xt(i));
        rpy_t.col(i) = Rt[i].eulerAngles(0,1,2)*180/M_PI;
    }
    std::cout<<"splined RPY: "<<std::endl;
    std::cout<<xt.transpose() <<std::endl;
    std::cout<<rpy_t <<std::endl;

    //----------------------------------------
    double d_xt = 0.01;
    std::vector<Eigen::Vector3d> d_Rt;    d_Rt.resize(n_sample);
    std::vector<Eigen::Matrix3d> Rt_real; Rt_real.resize(n_sample);
    std::vector<Eigen::Matrix3d> Rt_est;  Rt_est.resize(n_sample);
    double error = 0;
    for(size_t i=0;i<n_sample;i++)
    {
        Rt_real[i] = spline_rot.getPoint(xt(i)+d_xt);
        d_Rt[i] = spline_rot.getDerivative(xt(i));
        Rt_est[i] = mpc::ExpMatrix(mpc::getSkewMatrix(d_Rt[i]*d_xt))*spline_rot.getPoint(xt(i));
        error += mpc::getInverseSkewVector(mpc::LogMatrix(Rt_real[i].transpose()* Rt_est[i])).norm();
    }
    std::cout << "start: \n"<<Rt[3]<<std::endl;
    std::cout << "real:  \n"<<Rt_real[3]<<std::endl;
    std::cout << "esti:  \n"<<Rt_est[3]<<std::endl;
    std::cout << "error: "<< error <<std::endl;


    EXPECT_TRUE(error < 1e-2);
}


TEST(TestSpline, TestArcLengthSpline)
{
    std::ifstream iConfig(mpc::pkg_path + "Params/config.json");
    json jsonConfig;
    iConfig >> jsonConfig;

    mpc::PathToJson json_paths {mpc::pkg_path + std::string(jsonConfig["model_path"]),
                                 mpc::pkg_path + std::string(jsonConfig["cost_path"]),
                                 mpc::pkg_path + std::string(jsonConfig["bounds_path"]),
                                 mpc::pkg_path + std::string(jsonConfig["track_path"]),
                                 mpc::pkg_path + std::string(jsonConfig["normalization_path"]),
                                 mpc::pkg_path + std::string(jsonConfig["sqp_path"])};

    // test 6-D arc length spline approach
    // given a circle with randomly distributed points
    mpc::ArcLengthSpline sixDspline = mpc::ArcLengthSpline(json_paths);

    int NT = 50;    //number of "training" points
    int NV = 200;   //number of validation points points

    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    Eigen::VectorXd Z;
    std::vector<Eigen::Matrix3d> R;
    Eigen::VectorXd phiR;

    X.setZero(NT);
    Y.setZero(NT);
    Z.setZero(NT);
    R.resize(NT);
    // randomly distribute training points around half-circle
    // generate random angles between [0,pi]
    phiR.setRandom(NT);
    phiR = (phiR/2.0+0.5*Eigen::VectorXd::Ones(NT))*M_PI;
    // sort training points
    std::sort(phiR.data(), phiR.data()+phiR.size());
    // fix start and end point
    phiR(0) = 0;
    phiR(NT-1) = M_PI;
    // compute circle points
    for(int i=0;i<NT;i++){
        X(i) = 0.;
        Y(i) = std::cos(phiR(i));
        Z(i) = std::sin(phiR(i));
        R[i] = Matrix3d::Identity();
    }

    // give points to arc length based 3-D curve fitting
    sixDspline.gen6DSpline(X,Y,Z,R,0.01);
    //validation points
    Eigen::VectorXd phiv;
    //error between true circle and spline approximation
    Eigen::VectorXd error;
    // uniformly space validaiton points between [0,pi]
    phiv.setLinSpaced(NV,0,M_PI);

    error.setZero(NV);

    Eigen::Vector3d pos;

    for(int i=0;i<NV;i++){
        pos = sixDspline.getPosition(phiv(i));
        error(i) = std::sqrt(std::pow(pos(0) - 0.,2) + std::pow(pos(1) - std::cos(phiv(i)),2) + std::pow(pos(2) -std::sin(phiv(i)),2));
    }
    std::cout << "norm of error = " << error.norm() << std::endl;

    EXPECT_TRUE(error.norm()/(double)NV <= 0.03);
}
#endif