#ifndef MPCC_SELF_COLLISION_TEST_H
#define MPCC_SELF_COLLISION_TEST_H

#include "Constraints/SelfCollision/SelfCollisionModel.h"
#include "types.h"
#include "gtest/gtest.h"
#include <chrono>


typedef std::chrono::high_resolution_clock hd_clock;
typedef std::chrono::duration<double, std::ratio<1> > second;

TEST(TestSelfCollision, TestCalculateMLPOutput)
{
    std::chrono::time_point<hd_clock> beg;
    std::unique_ptr<mpcc::SelCollNNmodel> selcol;
    selcol = std::make_unique<mpcc::SelCollNNmodel>();
    mpcc::JointVector q0, dq, q1;
    q0 <<  0, 0, 0, -M_PI/2, 0,  M_PI/2,  M_PI/4;
    dq = mpcc::JointVector::Ones()*0.01;
    q1 = q0 + dq;
    bool result;
    try
    {
        Eigen::VectorXd n_hidden(3);
        n_hidden << 128, 64, 32;
        selcol->setNeuralNetwork(mpcc::PANDA_DOF, 1, n_hidden, true);
        beg = hd_clock::now();
        auto pred0 = selcol->calculateMlpOutput(q0, false);
        double min_dist0 = pred0.first.value();
        Eigen::MatrixXd min_dist0_jac = pred0.second;
        auto duration = std::chrono::duration_cast<second>(hd_clock::now() - beg).count(); 

        auto pred1 = selcol->calculateMlpOutput(q1,false);
        double min_dist1 = pred1.first.value();

        double min_est = min_dist0 + (min_dist0_jac*dq).value();

        std::cout << "Time to get self collision SDF and its gradient: " << duration*1e+3 << " [ms]!"<<std::endl;
        std::cout << std::fixed << std::setprecision(5) << "start     : "<< min_dist0 << std::endl; 
        std::cout << std::fixed << std::setprecision(5) << "real      : "<< min_dist1 << std::endl; 
        std::cout << std::fixed << std::setprecision(5) << "est       : "<< min_est << std::endl; 
        std::cout << std::fixed << std::setprecision(5) << "error[%]  : "<< fabs((min_est-min_dist1)/min_dist1)*100 << std::endl; 

        if(fabs((min_est-min_dist1)/min_dist1)*100 < 5) result = true;
        else result = false;

        // q0 << -0.46122822, -0.83015843, 1.45945742, -1.69096399, 1.58827731, 0.74419064, 0.21002424;
        // auto pred = selcol->calculateMlpOutput(q0, false);
        // std::cout << "min dist: " << pred.first.value() << std::endl; // python: 11.353057
        // std::cout << "min jac : " << pred.second << std::endl; // python: 0.1930, -0.5562,  1.0094, -0.3212, -1.4554, 10.2682, -0.0309
        // result = true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        result = false;
    }

    EXPECT_TRUE(result);
}
#endif