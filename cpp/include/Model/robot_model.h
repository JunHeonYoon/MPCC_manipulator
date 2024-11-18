#ifndef MPCC_ROBOT_MODEL_H
#define MPCC_ROBOT_MODEL_H

#include <rbdl/rbdl.h>
#include "config.h"
#include "types.h"

using namespace RigidBodyDynamics;
using namespace Eigen;
using namespace std;

// #define PANDA_NUM_LINKS 10 // with hand
#define PANDA_NUM_LINKS 9 // with hand

namespace mpcc
{   
    class RobotModel
    {
        public:
            RobotModel();
            ~RobotModel();

            void getUpdateKinematics(const VectorXd & q, const VectorXd & qdot);
            const unsigned int & getNumq() {return nq_;}
            const unsigned int & getNumv() {return nv_;}
            const unsigned int & getNumu() {return nu_;}
            const MatrixXd & getJacobian(const int & frame_id) 
            {
				Jacobian(frame_id);
				return j_;
			}
            const MatrixXd & getJacobian(const VectorXd &q) 
            {
				Jacobian(PANDA_NUM_LINKS, q);
				return j_;
			}
            const MatrixXd & getJacobian(const VectorXd &q, const int & frame_id) 
            {
				Jacobian(frame_id, q);
				return j_;
			}
            const MatrixXd & getJacobianv(const VectorXd &q) 
            {
				Jacobian(PANDA_NUM_LINKS, q);
				return j_v_;
			}
            const MatrixXd & getJacobianv(const VectorXd &q, const int & frame_id) 
            {
				Jacobian(frame_id, q);
				return j_v_;
			}
            const MatrixXd & getJacobianw(const VectorXd &q, const int & frame_id) 
            {
				Jacobian(frame_id, q);
				return j_w_;
			}
            const MatrixXd & getJacobianw(const VectorXd &q) 
            {
				Jacobian(PANDA_NUM_LINKS, q);
				return j_w_;
			}
            const Vector3d & getPosition(const int & frame_id) 
            {
				Position(frame_id);
				return x_;
			}
            const Vector3d & getEEPosition() 
            {
				Position(PANDA_NUM_LINKS);
				return x_;
			}
            const Vector3d & getEEPosition(const VectorXd &q) 
            {
				Position(PANDA_NUM_LINKS, q);
				return x_;
			}      
			const Matrix3d & getOrientation(const int & frame_id) 
            {
				Orientation(frame_id);
				return rotation_;
			}
            const Matrix3d & getEEOrientation() 
            {
				Orientation(PANDA_NUM_LINKS);
				return rotation_;
			}
            const Matrix3d & getEEOrientation(const VectorXd &q) 
            {
				Orientation(PANDA_NUM_LINKS, q);

				return rotation_;
			}
            const Affine3d & getTransformation(const int & frame_id) 
            {
				Transformation(frame_id);
				return trans_;
			}
            const Affine3d & getEETransformation() 
            {
				Transformation(PANDA_NUM_LINKS);
				return trans_;
			}
            const Affine3d & getEETransformation(const VectorXd &q) 
            {
				Transformation(PANDA_NUM_LINKS, q);
				return trans_;
			}
            const VectorXd & getJointPosition() 
            {
				return q_rbdl_;
            }
			const MatrixXd & getMassMatrix()
			{
				MassMatrix();
				return m_;
			}
			const VectorXd & getNonlinearEffect()
			{
				NonlinearEffect();
				return nle_;
			}
            const double & getManipulability(const VectorXd &q)
            {
                Manipulability(PANDA_NUM_LINKS,q);
                return mani_;
            }
            const double & getManipulability(const VectorXd &q, const int & frame_id)
            {
                Manipulability(frame_id,q);
                return mani_;
            }
            const VectorXd & getDManipulability(const VectorXd &q)
            {
                dManipulability(PANDA_NUM_LINKS,q);
                return d_mani_;
            }
            const VectorXd & getDManipulability(const VectorXd &q, const int & frame_id)
            {
                dManipulability(frame_id,q);
                return d_mani_;
            }
            
        private:
            void setRobot();
            void setPanda(unsigned int base_id, Vector3d joint_position, Matrix3d joint_rotataion);
            void setHusky();
            void Jacobian(const int & frame_id);
            void Jacobian(const int & frame_id, const VectorXd &q);
			void Position(const int & frame_id);
			void Position(const int & frame_id, const VectorXd &q);
			void Orientation(const int & frame_id);
			void Orientation(const int & frame_id, const VectorXd &q);
			void Transformation(const int & frame_id);
			void Transformation(const int & frame_id, const VectorXd &q);
			void MassMatrix();
			void NonlinearEffect();
            void Manipulability(const int & frame_id, const VectorXd &q); 
            void dManipulability(const int & frame_id, const VectorXd &q); 

            std::shared_ptr<Model> model_;
            unsigned int base_id_;     // for mobile base
            unsigned int body_id_[PANDA_NUM_LINKS];  // only for manipulator (link0~7, hand)
            // Vector3d link_position_[panda_num_links]; // only for manipulator (link0~7, hand)

            // Current state
            VectorXd q_rbdl_;
			VectorXd qdot_rbdl_;

            // Task space (End Effector)
            Vector3d x_;
            Matrix3d rotation_;
            Affine3d trans_;
			MatrixXd j_;           // Full basic Jacobian matrix
            MatrixXd j_v_;	       // Linear velocity Jacobian matrix
            MatrixXd j_w_;	       // Angular veolicty Jacobain matrix
            MatrixXd j_tmp_;
            double mani_;          // manipulability index
            VectorXd d_mani_;      // gradient of manipulability index wrt q

            // Dynamics
            MatrixXd m_;         // Mass matrix
            MatrixXd m_inverse_; // Inverse of mass matrix
            VectorXd nle_;       // Nonlinear Effect (Gravity torque + Coriolis torque)
            MatrixXd m_tmp_;
            VectorXd nle_tmp_;

        protected:
			unsigned int nq_; // size of state
			unsigned int nv_; // size of joint velocity
			unsigned int nu_; // size of control input
            unsigned int ee_dof = 6;
    };
}

#endif