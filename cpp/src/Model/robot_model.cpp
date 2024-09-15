#include "Model/robot_model.h"
#include <iomanip>
#include <iostream>

mpcc::RobotModel::RobotModel()
{

	nq_ = TOCABI_DOF; 
	nv_ = TOCABI_DOF; 
	nu_ = TOCABI_DOF; 

	q_rbdl_.resize(nq_);
	qdot_rbdl_.resize(nq_);
	qdot_rbdl_.resize(nq_);
	
	q_rbdl_.setZero();
	qdot_rbdl_.setZero();
	qdot_rbdl_.setZero();

	j_.resize(ee_dof, nv_);
	j_v_.resize(int(ee_dof/2), nv_);
	j_w_.resize(int(ee_dof/2), nv_);
	j_tmp_.resize(ee_dof, nq_);
	d_mani_.resize(nq_);

	x_.setZero();
	rotation_.setZero();
	j_.setZero();
	j_v_.setZero();
	j_w_.setZero();
	j_tmp_.setZero();
	mani_ = 0;
	d_mani_.setZero();

	nle_.resize(nv_);
	m_.resize(nv_, nv_);
	m_inverse_.resize(nv_, nv_);
	m_tmp_.resize(nq_, nq_);
	nle_tmp_.resize(nq_);

	nle_.setZero();
	m_.setZero();
	m_inverse_.setZero();
	m_tmp_.setZero();
	nle_tmp_.setZero();

   setRobot(pkg_path+"/urdf/tocabi.urdf");
   std::cout << "Tocabi Robot Model is loaded!" << std::endl;
}

mpcc::RobotModel::~RobotModel()
{
	std::cout << "Tocabi Robot Model is removed!" << std::endl;
}

void mpcc::RobotModel::setRobot(const std::string &urdf_file_path = pkg_path+"/urdf/tocabi.urdf")
{
	model_ = std::make_shared<Model>();    
    model_->gravity = Math::Vector3d(0., 0, -9.81);

	Addons::URDFReadFromFile(urdf_file_path.c_str(), model_.get(), false, false);

	std::vector<std::string> links_name;
	links_name = {"Pelvis_Link", "Waist1_Link", "Waist2_Link", "Upperbody_Link", "R_Shoulder1_Link",
				  "R_Shoulder2_Link", "R_Shoulder3_Link", "R_Armlink_Link", "R_Elbow_Link", 
				  "R_Forearm_Link", "R_Wrist1_Link", "R_Wrist2_Link", "palm"};
	for(size_t i=0; i<links_name.size(); i++)
	{
		body_id_[i] = model_->GetBodyId(links_name[i].c_str());
	}
}

void mpcc::RobotModel::Jacobian(const int &frame_id)
{
	j_tmp_.setZero();

	CalcPointJacobian6D(*model_, q_rbdl_, body_id_[frame_id - 1], Math::Vector3d::Zero(), j_tmp_, true);

	j_w_ = j_tmp_.block(0, 0, 3, nv_);
	j_v_ = j_tmp_.block(3, 0, 3, nv_);

	j_ << j_v_, j_w_;
}

void mpcc::RobotModel::Jacobian(const int &frame_id, const VectorXd &q)
{
	j_tmp_.setZero();

	CalcPointJacobian6D(*model_, q, body_id_[frame_id - 1], Math::Vector3d::Zero(), j_tmp_, true);

	j_w_ = j_tmp_.block(0, 0, 3, nv_);
	j_v_ = j_tmp_.block(3, 0, 3, nv_);

	j_ << j_v_, j_w_;

}

void mpcc::RobotModel::Position(const int &frame_id)
{
	x_ = CalcBodyToBaseCoordinates(*model_, q_rbdl_, body_id_[frame_id - 1], Math::Vector3d::Zero(), true);
}

void mpcc::RobotModel::Position(const int &frame_id, const VectorXd &q)
{
	x_ = CalcBodyToBaseCoordinates(*model_, q, body_id_[frame_id - 1], Math::Vector3d::Zero(), true);
}

void mpcc::RobotModel::Orientation(const int &frame_id)
{
	rotation_ = CalcBodyWorldOrientation(*model_, q_rbdl_, body_id_[frame_id - 1], true).transpose();
}

void mpcc::RobotModel::Orientation(const int &frame_id, const VectorXd &q)
{
	// std::cout<<"body name: " << model_->GetBodyName(body_id_[frame_id - 1]) <<std::endl;
	rotation_ = CalcBodyWorldOrientation(*model_, q, body_id_[frame_id - 1], true).transpose();
}

void mpcc::RobotModel::Transformation(const int &frame_id)
{
	Position(frame_id);
	Orientation(frame_id);
	trans_.linear() = rotation_;
	trans_.translation() = x_;
}

void mpcc::RobotModel::Transformation(const int &frame_id, const VectorXd &q)
{
	Position(frame_id, q);
	Orientation(frame_id, q);
	trans_.linear() = rotation_;
	trans_.translation() = x_;
}

void mpcc::RobotModel::MassMatrix()
{
	m_tmp_.setZero();
	CompositeRigidBodyAlgorithm(*model_, q_rbdl_, m_tmp_, true);
	m_ = m_tmp_;
}

void mpcc::RobotModel::NonlinearEffect()
{
	nle_tmp_.setZero();
	MassMatrix();
	NonlinearEffects(*model_, q_rbdl_, qdot_rbdl_, nle_tmp_);
	nle_ = nle_tmp_;
}

void mpcc::RobotModel::Manipulability(const int &frame_id, const VectorXd &q)
{
	Jacobian(frame_id, q);
	mani_ = sqrt((j_*j_.transpose()).determinant());
}

void mpcc::RobotModel::dManipulability(const int &frame_id, const VectorXd &q)
{
	double delta = 1e-4;
	for(size_t i=0;i<nq_;i++)
	{
		VectorXd delta_q = VectorXd::Zero(nq_);
		delta_q(i) = delta;
		Manipulability(frame_id, q+delta_q);
		double m_1 = mani_;
		Manipulability(frame_id, q-delta_q);
		double m_2 = mani_;
		d_mani_(i) = (m_1 - m_2) / (2*delta);
	}
}

void mpcc::RobotModel::getUpdateKinematics(const VectorXd &q, const VectorXd &qdot)
{
	q_rbdl_ = q;
	qdot_rbdl_ = qdot;
	UpdateKinematicsCustom(*model_, &q_rbdl_, &qdot_rbdl_, NULL);

}
