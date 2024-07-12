#include "gtest/gtest.h"
#include "Tests/model_integrator_test.h"
#include "Tests/robot_model_test.h"
#include "Tests/self_collision_test.h"
#include "Tests/spline_test.h"
#include "Tests/constraints_test.h"
#include "Tests/cost_test.h"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}