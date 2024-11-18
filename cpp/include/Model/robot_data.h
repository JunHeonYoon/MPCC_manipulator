#ifndef MPC_ROBOT_DATA_H
#define MPC_ROBOT_DATA_H

#include "Model/robot_model.h"
#include "Constraints/SelfCollision/SelfCollisionModel.h"
#include "Constraints/EnvCollision/EnvCollisionModel.h"

namespace mpc
{
// Data containing kinematic of robot wrt given joint angle
struct RobotData
{
    Eigen::Matrix<double,PANDA_DOF,1> q_;           // joint angle
    // Eigen::Matrix<double,PANDA_DOF,1> q_dot;       // joint velocity
    
    Eigen::Vector3d EE_position_;                   // End-Effector position
    Eigen::Matrix3d EE_orientation_;                // End-Effector orientation

    Eigen::Matrix<double,6,PANDA_DOF> J_;           // End-Effector Jacobian
    Eigen::Matrix<double,3,PANDA_DOF> Jv_;          // End-Effector translation Jacobian
    Eigen::Matrix<double,3,PANDA_DOF> Jw_;          // End-Effector rotation Jacobian

    double manipul_;                                // Manipullabilty
    Eigen::Matrix<double,PANDA_DOF,1> d_manipul_;   // Gradient of Manipullabilty wrt q

    double sel_min_dist_;                               // Minimum distance between robot links
    Eigen::Matrix<double,PANDA_DOF,1> d_sel_min_dist_;  // Jacobian of minimum distance between robot links

    double obs_radius_;                                               // Radius of external sphere obstacle
    Eigen::Matrix<double, PANDA_NUM_LINKS, 1> env_min_dist_;          // Minimum distance between robot links and enviornment
    Eigen::Matrix<double,PANDA_NUM_LINKS,PANDA_DOF> d_env_min_dist_;  // Jacobian of minimum distance between robot links and enviornment

    bool is_data_valid;
    bool is_env_data_valid;

    void setZero()
    {
        q_.setZero();
        // q_dot.setZero();
        EE_position_.setZero();
        EE_orientation_.setZero();
        J_.setZero();
        Jv_.setZero();
        Jw_.setZero();
        manipul_ = 0;
        d_manipul_.setZero();
        sel_min_dist_ = 0;
        d_sel_min_dist_.setZero();
        env_min_dist_.setZero();
        d_env_min_dist_.setZero();
        is_data_valid = false;
        is_env_data_valid = false;
    }

    void update(Eigen::Matrix<double,PANDA_DOF,1> q_input, const std::unique_ptr<RobotModel> &robot_model, const std::unique_ptr<SelCollNNmodel> &selcol_model)
    {
        q_ = q_input;
        EE_position_ = robot_model->getEEPosition(q_);
        EE_orientation_ = robot_model->getEEOrientation(q_);
        J_ = robot_model->getJacobian(q_);
        Jv_ = J_.block(0,0,3,PANDA_DOF);
        Jw_ = J_.block(3,0,3,PANDA_DOF);
        manipul_ = robot_model->getManipulability(q_);
        d_manipul_ = robot_model->getDManipulability(q_);

        auto pred = selcol_model->calculateMlpOutput(q_, false);
        sel_min_dist_ = pred.first.value();
        d_sel_min_dist_ = pred.second.transpose();

        is_data_valid = true;
    }

    // void updateEnv(double env_min_dist, Eigen::Matrix<double,PANDA_DOF,1> d_env_min_dist)
    void updateEnv(const Eigen::Vector3d &obs_position, const double &obs_radius, const std::unique_ptr<EnvCollNNmodel> &envcol_model)
    {
        assert(is_data_valid == true);

        obs_radius_ = obs_radius;

        Eigen::VectorXd input(PANDA_DOF + obs_position.size());
        input << q_, obs_position;

        auto pred = envcol_model->calculateMlpOutput(input, false);
        env_min_dist_ = pred.first;
        d_env_min_dist_ = pred.second.block(0, 0, PANDA_NUM_LINKS, PANDA_DOF);

        is_env_data_valid = true;
    }

    bool isUpdated()
    {
        return (is_data_valid && is_env_data_valid);
    }
};
}
#endif // MPCC_ROBOT_DATA