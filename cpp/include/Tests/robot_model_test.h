#ifndef MPC_ROBOT_MODEL_TEST_H
#define MPC_ROBOT_MODEL_TEST_H

#include "Model/robot_model.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include "gtest/gtest.h"

typedef std::chrono::high_resolution_clock hd_clock;
typedef std::chrono::duration<double, std::ratio<1> > second;
std::chrono::time_point<hd_clock> beg_;

TEST(TestRobotModel, TestGetEEPosition)
{
    std::unique_ptr<mpc::RobotModel> robot;
    robot = std::make_unique<mpc::RobotModel>();
    mpc::JointVector q0;
    // q0 <<  -0.002, -0.001,  0.002, -1.574,  0.006,  1.584,  0.789;
    q0 << 0, 0, 0, -M_PI/2, 0, M_PI/2, M_PI/4;
    Vector3d x0 = Vector3d::Zero();
    bool result;
    try
    {
        x0 = robot->getEEPosition(q0);
        std::cout << "EE position: " << std::fixed << std::setprecision(5) <<
         x0.transpose() << std::endl;
        // real robot: 
        // 0.557 0.001 0.522
        result = true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        result = false;
    }

    EXPECT_TRUE(result);
}

TEST(TestRobotModel, TestGetEEOrientation)
{
    std::unique_ptr<mpc::RobotModel> robot;
    robot = std::make_unique<mpc::RobotModel>();
    mpc::JointVector q0;
    q0.setZero();
    q0 << 0, 0, 0, -M_PI/2, 0, M_PI/2, M_PI/4;
    Matrix3d r0 = Matrix3d::Zero();
    bool result;
    try
    {
        r0 = robot->getEEOrientation(q0);
        std::cout << "EE ori: \n" << std::fixed << std::setprecision(3) <<
         r0 << std::endl;
        result = true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        result = false;
    }
    EXPECT_TRUE(result);
}

TEST(TestRobotModel, TestGetJacobianv)
{
    std::unique_ptr<mpc::RobotModel> robot;
    robot = std::make_unique<mpc::RobotModel>();
    mpc::JointVector q0;
    q0 <<  -0.002, -0.001,  0.002, -1.574,  0.006,  1.584,  0.789;
    Matrix<double, 3, mpc::PANDA_DOF> J0;
    J0.setZero();
    bool result;
    try
    {
        J0 = robot->getJacobianv(q0);
        // std::cout << "EE jacobian : \n" << std::fixed << std::setprecision(3) 
        // << J0 << std::endl;
        // real robot:
        // 0.001 0.189 -0.001 0.128 0.000 0.209 0.000
        // 0.557 -0.000 0.557 -0.000 0.209 -0.001 -0.000
        // 0.000 -0.557 -0.000 0.474 0.001 0.090 -0.000
        result = true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        result = false;
    }
    EXPECT_TRUE(result);
}

TEST(TestRobotModel, TestManipulability)
{
    std::unique_ptr<mpc::RobotModel> robot;
    robot = std::make_unique<mpc::RobotModel>();
    mpc::JointVector q0, dq, q1;
    q0 <<  0, 0, 0, 0.1, 0, M_PI/2, M_PI/4;
    dq = mpc::JointVector::Ones()*0.01;
    q1 = q0 + dq;
    double mani0 = 0, mani1 = 0, mani_est = 0;
    VectorXd d_mani = VectorXd::Zero(mpc::PANDA_DOF);
    bool result;
    try
    {
        beg_ = hd_clock::now();
        mani0 = robot->getManipulability(q0);
        mani1 = robot->getManipulability(q1);
        d_mani = robot->getDManipulability(q0);
        mani_est = mani0 + d_mani.dot(dq);
        auto duration = std::chrono::duration_cast<second>(hd_clock::now() - beg_).count();
        // std::cout << "Manipulability: \n" << mani << std::endl;
        // std::cout << "Gradient of Manipulability: \n" << d_mani.transpose() << std::endl;
        std::cout << "Time to get manipulability index and its gradient: " << duration*1e+3 << " [ms]!"<<std::endl;
        std::cout << std::fixed << std::setprecision(5) << "start     : "<< mani0 << std::endl; 
        std::cout << std::fixed << std::setprecision(5) << "real      : "<< mani1 << std::endl; 
        std::cout << std::fixed << std::setprecision(5) << "est       : "<< mani_est << std::endl; 
        std::cout << std::fixed << std::setprecision(5) << "error[%]  : "<< fabs((mani_est-mani1)/mani1)*100 << std::endl; 

        if(fabs((mani_est-mani1)/mani1)*100 < 5) result = true;
        else result = false;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        result = false;
    }
    EXPECT_TRUE(result);
}

#endif