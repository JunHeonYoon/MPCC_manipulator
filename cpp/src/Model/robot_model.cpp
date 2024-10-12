#include "Model/robot_model.h"
#include <iomanip>
#include <iostream>

namespace mpc
{
RobotModel::RobotModel()
{

	nq_ = PANDA_DOF; 
	nv_ = PANDA_DOF; 
	nu_ = PANDA_DOF; 

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

   setRobot();
   std::cout << "Franka Panda Robot Model is loaded!" << std::endl;
}

RobotModel::~RobotModel()
{
	std::cout << "Franka Panda Robot Model is removed!" << std::endl;
}

void RobotModel::setRobot()
{
	model_ = std::make_shared<Model>();    
    model_->gravity = Math::Vector3d(0., 0, -9.81);
	Body root_link = Body();
	Joint root_joint(JointTypeFixed);
	Math::SpatialTransform root_joint_frame = Math::SpatialTransform();
	base_id_ = model_->AppendBody(root_joint_frame, root_joint, root_link, "world");
	setPanda(base_id_, Vector3d::Zero(), Matrix3d::Identity());
	
}

void RobotModel::setPanda(unsigned int base_id, Vector3d base_position, Matrix3d base_rotataion)
{
	Math::Vector3d com_position[13];
	com_position[0] = Vector3d(-0.041018, -0.00014, 0.049974);        // panda_link0
	com_position[1] = Vector3d(0.003875, 0.002081, -0.04762);         // panda_link1
	com_position[2] = Vector3d(-0.003141, -0.02872, 0.003495);        // panda_link2
	com_position[3] = Vector3d(2.7518e-02, 3.9252e-02, -6.6502e-02);  // panda_link3
	com_position[4] = Vector3d(-5.317e-02, 1.04419e-01, 2.7454e-02);  // panda_link4
	com_position[5] = Vector3d(-1.1953e-02, 4.1065e-02, -3.8437e-02); // panda_link5
	com_position[6] = Vector3d(6.0149e-02, -1.4117e-02, -1.0517e-02); // panda_link6
	com_position[7] = Vector3d(1.0517e-02, -4.252e-03, 6.1597e-02);   // panda_link7
	com_position[8] = Vector3d(0.0, 0.0, 0.0);                        // panda_link8
	com_position[9] = Vector3d(-0.01, 0.0, 0.03);                     // panda_hand
	com_position[10] = Vector3d(0.0, 0.0, 0.0);                       // panda_leftfinger
	com_position[11] = Vector3d(0.0, 0.0, 0.0);                       // panda_rightfinger
	com_position[12] = Vector3d(0.0, 0.0, 0.0);                       // panda_hand tcp


	double mass[13];
	mass[0] = 0.629769; // link0
	mass[1] = 4.97068;  // link1
	mass[2] = 0.646926; // link2
	mass[3] = 3.2286;   // link3
	mass[4] = 3.5879;   // link4
	mass[5] = 1.22595;  // link5
	mass[6] = 1.66656;  // link6
	mass[7] = 0.735522; // link7
	mass[8] = 0.;       // link8
	mass[9] = 0.73;     // hand
	mass[10] = 0.015;   // left finger
	mass[11] = 0.015;   // right finger
	mass[11] = 0.;      // hand tcp

	Math::Matrix3d inertia[13];
	// link0
	inertia[0] = (Matrix3d() << 0.00315, 8.2904e-07, 0.00015,
								8.2904e-07, 0.00388, 8.2299e-06,
								0.00015, 8.2299e-06, 0.004285).finished();

	// link1
	inertia[1] = (Matrix3d() << 0.70337, -0.000139, 0.006772,
								-0.000139, 0.70661, 0.019169,
								0.006772, 0.019169, 0.009117).finished();

	// link2
	inertia[2] = (Matrix3d() << 0.007962, -0.003925, 0.010254,
								-0.003925, 0.02811, 0.000704,
								0.010254, 0.000704, 0.025995).finished();

	// link3
	inertia[3] = (Matrix3d() << 0.037242, -0.004761, -0.011396,
								-0.004761, 0.036155, -0.012805,
								-0.011396, -0.012805, 0.01083).finished();

	// link4
	inertia[4] = (Matrix3d() << 0.025853, 0.007796, -0.001332,
								0.007796, 0.019552, 0.008641,
								-0.001332, 0.008641, 0.028323).finished();

	// link5
	inertia[5] = (Matrix3d() << 0.035549, -0.002117, -0.004037,
								-0.002117, 0.029474, 0.000229,
								-0.004037, 0.000229, 0.008627).finished();

	// link6
	inertia[6] = (Matrix3d() << 0.001964, 0.000109, -0.001158,
								0.000109, 0.004354, 0.000341,
								-0.001158, 0.000341, 0.005433).finished();

	// link7
	inertia[7] = (Matrix3d() << 0.012516, -0.000428, -0.001196,
								-0.000428, 0.010027, -0.000741,
								-0.001196, -0.000741, 0.004815).finished();

	// link8
	inertia[8] = Matrix3d::Zero();

	// hand
	inertia[9] = (Matrix3d() << 0.001, 0, 0,
								0, 0.0025, 0,
								0, 0, 0.0017).finished();

	// left finger
	inertia[10] = (Matrix3d() << 2.375e-06, 0, 0,
								0, 2.375e-06, 0,
								0, 0, 7.5e-07).finished();

	// right finger
	inertia[11] = inertia[10];

	// hand-tcp
	inertia[12] = Matrix3d::Zero();

	Math::Vector3d axis[13];
	axis[0] = Vector3d::Zero();    // world to link0 (dummy: fixed)
	axis[1] = Vector3d::UnitZ();   // link0 to link1
	axis[2] = Vector3d::UnitZ();   // link1 to link2
	axis[3] = Vector3d::UnitZ();   // link2 to link3
	axis[4] = Vector3d::UnitZ();   // link3 to link4
	axis[5] = Vector3d::UnitZ();   // link4 to link5
	axis[6] = Vector3d::UnitZ();   // link5 to link6
	axis[7] = Vector3d::UnitZ();   // link6 to link7
	axis[8] = Vector3d::Zero();    // link7 to link8 (dummy: fixed)
	axis[9] = Vector3d::Zero();    // link8 to hand (dummy: fixed)
	axis[10] = Vector3d::UnitY();  // hand to left finger
	axis[11] = -Vector3d::UnitY(); // hand to right finger
	axis[12] = Vector3d::Zero();   // hand to hand-tcp (dummy: fixed)

	Math::Vector3d joint_position[13];
	joint_position[0] = base_position;                // world to link0
	joint_position[1] = Vector3d(0, 0, 0.333);        // link0 to link1
	joint_position[2] = Vector3d(0, 0, 0);            // link1 to link2
	joint_position[3] = Vector3d(0, -0.316, 0);       // link2 to link3
	joint_position[4] = Vector3d(0.0825, 0, 0);       // link3 to link4
	joint_position[5] = Vector3d(-0.0825, 0.384, 0);  // link4 to link5
	joint_position[6] = Vector3d(0, 0, 0);            // link5 to link6
	joint_position[7] = Vector3d(0.088, 0, 0);        // link6 to link7
	joint_position[8] = Vector3d(0, 0, 0.107);        // link7 to link8
	joint_position[9] = Vector3d(0, 0, 0);            // link8 to hand
	joint_position[10] = Vector3d(0, 0, 0.0584);      // hand to leftfinger
	joint_position[11] = Vector3d(0, 0, 0.0584);      // hand to rightfinger
	joint_position[12] = Vector3d(0, 0, 0.1034);      // hand to hand-tcp

	Math::Matrix3d joint_rotation[13];
	// world to link0
	joint_rotation[0] = base_rotataion;

	// link0 to link1
	joint_rotation[1] = (Matrix3d() <<
		1., 0., 0.,
		0., 1., 0.,
		0., 0., 1.
	).finished();

	// link1 to link2
	joint_rotation[2] = (Matrix3d() <<
		1., 0., 0.,
		0., 0., -1.,
		0., 1., 0.
	).finished();

	// link2 to link3
	joint_rotation[3] = (Matrix3d() <<
		1., 0., 0.,
		0., 0., 1.,
		0., -1., 0.
	).finished();

	// link3 to link4
	joint_rotation[4] = (Matrix3d() <<
		1., 0., 0.,
		0., 0., 1.,
		0., -1., 0.
	).finished();

	// link4 to link5
	joint_rotation[5] = (Matrix3d() <<
		1., 0., 0.,
		0., 0., -1.,
		0., 1., 0.
	).finished();

	// link5 to link6
	joint_rotation[6] = (Matrix3d() <<
		1., 0., 0.,
		0., 0., 1.,
		0., -1., 0.
	).finished();

	// link6 to link7
	joint_rotation[7] = (Matrix3d() <<
		1., 0., 0.,
		0., 0., 1.,
		0., -1., 0.
	).finished();

	// link7 to link8
	joint_rotation[8] = (Matrix3d() <<
		1., 0., 0.,
		0., 1., 0.,
		0., 0., 1.
	).finished();

	// link8 to hand
	joint_rotation[9] = (Matrix3d() <<
		0.707107, -0.707107, 0.,
	    0.707107, 0.707107, 0.,
		0., 0., 1.
	).finished();

	// hand to leftfinger
	joint_rotation[10] = (Matrix3d() <<
		1., 0., 0.,
		0., 1., 0.,
		0., 0., 1.
	).finished();

	// hand to rightfinger
	joint_rotation[11] = (Matrix3d() <<
		1., 0., 0.,
		0., 1., 0.,
		0., 0., 1.
	).finished();

	// hand to hand-tcp
	joint_rotation[12] = (Matrix3d() <<
		1., 0., 0.,
		0., 1., 0.,
		0., 0., 1.
	).finished();

	std::string link_name[13];
	link_name[0] = "panda_link0";
	link_name[1] = "panda_link1";
	link_name[2] = "panda_link2";
	link_name[3] = "panda_link3";
	link_name[4] = "panda_link4";
	link_name[5] = "panda_link5";
	link_name[6] = "panda_link6";
	link_name[7] = "panda_link7";
	link_name[8] = "panda_link8";
	link_name[9] = "panda_hand";
	link_name[10] = "panda_leftfinger";
	link_name[11] = "panda_rightfinger";
	link_name[12] = "panda_hand_tcp";

	Body body[13];
	Joint joint[13];
	unsigned int body_id[13]; 
	for(size_t i = 0; i < 13; i++)
	{
		body[i] = Body(mass[i], com_position[i], inertia[i]);
		if(i == 0) // body: link0, joint: world to link0
		{
			joint[i] = Joint(JointTypeFixed);
			body_id[i] = model_->AddBody(base_id, Math::SpatialTransform(joint_rotation[i], joint_position[i]), joint[i], body[i], link_name[i].c_str());
		}
		else if(i == 8 || i == 9) // body: link8, hand, joint: link7 to link8, link8 to hand
		{
			joint[i] = Joint(JointTypeFixed);
			body_id[i] = model_->AddBody(body_id[i - 1], Math::SpatialTransform(joint_rotation[i], joint_position[i]), joint[i], body[i], link_name[i].c_str());
		}
		else if(i == 10 || i == 11) // body: finger, joint: hand to finger
		{
			// joint[i] = Joint(JointTypePrismatic, axis[i]);
			joint[i] = Joint(JointTypeFixed);
			body_id[i] = model_->AddBody(body_id[9], Math::SpatialTransform(joint_rotation[i], joint_position[i]), joint[i], body[i], link_name[i].c_str());
		}
		else if(i == 12) // body: hand-tcp, joint: hand to hand-tcp
		{
			joint[i] = Joint(JointTypeFixed);
			body_id[i] = model_->AddBody(body_id[9], Math::SpatialTransform(joint_rotation[i], joint_position[i]), joint[i], body[i], link_name[i].c_str());
		}
		else
		{
			joint[i] = Joint(JointTypeRevolute, axis[i]);
			body_id[i] = model_->AddBody(body_id[i - 1], Math::SpatialTransform(joint_rotation[i], joint_position[i]), joint[i], body[i], link_name[i].c_str());
		}
	}

	std::string tmp_name = "panda_";
	for(size_t i=0; i<PANDA_NUM_LINKS-1; ++i) 
	{
		body_id_[i] = model_->GetBodyId(("panda_link" + std::to_string(i)).c_str());
	}
	body_id_[PANDA_NUM_LINKS-1] = model_->GetBodyId("panda_hand_tcp");
}

void RobotModel::setHusky()
{
	// for base_footprint
	Body virtual_body[5];
	Joint virtual_joint[5];
	virtual_body[0] = Body(0.0, Math::Vector3d(0.0, 0.0, 0.0), Math::Matrix3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
	virtual_body[1] = Body(0.0, Math::Vector3d(0.0, 0.0, 0.0), Math::Matrix3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
	virtual_body[2] = Body(0.0, Math::Vector3d(0.0, 0.0, 0.0), Math::Matrix3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
	virtual_joint[0] = Joint(JointTypePrismatic, Vector3d::UnitX());
	virtual_joint[1] = Joint(JointTypePrismatic, Vector3d::UnitY());
	virtual_joint[2] = Joint(Math::SpatialVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0));

	unsigned int virtual_body_id[3];
	virtual_body_id[0] = model_->AddBody(0, Math::Xtrans(Math::Vector3d(0.0, 0.0, 0.0)), virtual_joint[0], virtual_body[0], "husky_x");
	virtual_body_id[1] = model_->AddBody(virtual_body_id[0], Math::Xtrans(Math::Vector3d(0.0, 0.0, 0.0)), virtual_joint[1], virtual_body[1], "husky_y");
	virtual_body_id[2] = model_->AddBody(virtual_body_id[1], Math::Xtrans(Math::Vector3d(0.0, 0.0, 0.0)), virtual_joint[2], virtual_body[2], "husky_th");

	// for floating base
	Body base;
	double mass = 50.0;
	base = Body(mass, Math::Vector3d(0.0, 0.0, 0.0), Math::Vector3d(0.5, 100.0, 100.0));
	base_id_ = model_->AddBody(virtual_body_id[2], Math::Xtrans(Math::Vector3d(0.0, 0.0, 0.0)), Joint(JointTypeFixed), base, "husky_base");

	//// for wheel
	virtual_body[3] = Body(2.6 *2.0, Math::Vector3d(0.0, 0.0, 0.0), Math::Matrix3d(0.001, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1));
	virtual_body[4] = Body(2.6 *2.0, Math::Vector3d(0.0, 0.0, 0.0), Math::Matrix3d(0.001, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1));
	virtual_joint[3] = Joint(JointTypeRevolute, -Vector3d::UnitY());
	virtual_joint[4] = Joint(JointTypeRevolute,  Vector3d::UnitY());

	model_->AddBody(base_id_, Math::Xtrans(Math::Vector3d(0.0, 0.25, 0.165)), virtual_joint[3], virtual_body[3], "husky_leftwheel");
	model_->AddBody(base_id_, Math::Xtrans(Math::Vector3d(0.0, -0.25, 0.165)), virtual_joint[4], virtual_body[4], "husky_rightwheel");
}

void RobotModel::Jacobian(const int &frame_id)
{
	j_tmp_.setZero();

	CalcPointJacobian6D(*model_, q_rbdl_, body_id_[frame_id - 1], Math::Vector3d::Zero(), j_tmp_, true);

	j_w_ = j_tmp_.block(0, 0, 3, nv_);
	j_v_ = j_tmp_.block(3, 0, 3, nv_);

	j_ << j_v_, j_w_;
}

void RobotModel::Jacobian(const int &frame_id, const VectorXd &q)
{
	j_tmp_.setZero();

	CalcPointJacobian6D(*model_, q, body_id_[frame_id - 1], Math::Vector3d::Zero(), j_tmp_, true);

	j_w_ = j_tmp_.block(0, 0, 3, nv_);
	j_v_ = j_tmp_.block(3, 0, 3, nv_);

	j_ << j_v_, j_w_;

}

void RobotModel::Position(const int &frame_id)
{
	x_ = CalcBodyToBaseCoordinates(*model_, q_rbdl_, body_id_[frame_id - 1], Math::Vector3d::Zero(), true);
}

void RobotModel::Position(const int &frame_id, const VectorXd &q)
{
	x_ = CalcBodyToBaseCoordinates(*model_, q, body_id_[frame_id - 1], Math::Vector3d::Zero(), true);
}

void RobotModel::Orientation(const int &frame_id)
{
	rotation_ = CalcBodyWorldOrientation(*model_, q_rbdl_, body_id_[frame_id - 1], true).transpose();
}

void RobotModel::Orientation(const int &frame_id, const VectorXd &q)
{
	// std::cout<<"body name: " << model_->GetBodyName(body_id_[frame_id - 1]) <<std::endl;
	rotation_ = CalcBodyWorldOrientation(*model_, q, body_id_[frame_id - 1], true).transpose();
}

void RobotModel::Transformation(const int &frame_id)
{
	Position(frame_id);
	Orientation(frame_id);
	trans_.linear() = rotation_;
	trans_.translation() = x_;
}

void RobotModel::Transformation(const int &frame_id, const VectorXd &q)
{
	Position(frame_id, q);
	Orientation(frame_id, q);
	trans_.linear() = rotation_;
	trans_.translation() = x_;
}

void RobotModel::MassMatrix()
{
	m_tmp_.setZero();
	CompositeRigidBodyAlgorithm(*model_, q_rbdl_, m_tmp_, true);
	m_ = m_tmp_;
}

void RobotModel::NonlinearEffect()
{
	nle_tmp_.setZero();
	MassMatrix();
	NonlinearEffects(*model_, q_rbdl_, qdot_rbdl_, nle_tmp_);
	nle_ = nle_tmp_;
}

void RobotModel::Manipulability(const int &frame_id, const VectorXd &q)
{
	Jacobian(frame_id, q);
	mani_ = sqrt((j_*j_.transpose()).determinant());
}

void RobotModel::dManipulability(const int &frame_id, const VectorXd &q)
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

void RobotModel::getUpdateKinematics(const VectorXd &q, const VectorXd &qdot)
{
	q_rbdl_ = q;
	qdot_rbdl_ = qdot;
	UpdateKinematicsCustom(*model_, &q_rbdl_, &qdot_rbdl_, NULL);

}
}