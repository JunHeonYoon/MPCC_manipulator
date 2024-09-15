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

#include "Constraints/constraints.h"
namespace mpcc{
Constraints::Constraints()
{   
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

Constraints::Constraints(double Ts,const PathToJson &path) 
:param_(Param(path.param_path))
{
}

Constraints::Constraints(double Ts,const PathToJson &path,const ParamValue &param_value)
:param_(Param(path.param_path,param_value.param))
{
}

double getRBF(double delta, double h)
{
    // Grandia, Ruben, et al. 
    // "Feedback mpc for torque-controlled legged robots." 
    // 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019.
    double result;
    if (h >= delta) result = -log(h+1);
    else            result = -log(delta+1) - 1/(delta+1) * (h-delta) + 1/(2*pow(delta+1,2)) * pow(h-delta,2);
    return result;
}

double getDRBF(double delta, double h)
{
    // Grandia, Ruben, et al. 
    // "Feedback mpc for torque-controlled legged robots." 
    // 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019.
    double result;
    if (h >= delta) result = -1/(h+1);
    else            result = -1/(delta+1) + 1/(pow(delta+1,2)) * (h-delta);
    return result;
}

void Constraints::getSelcollConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                                       OneDConstraintInfo *constraint, OneDConstraintsJac* Jac)
{
    // compute self-collision constraints
    // -∇_q Γ(q)^T * q_dot <= -RBF(Γ(q) - r), where r is buffer
    const JointVector q = stateToJointVector(x);
    const dJointVector dq = inputTodJointVector(u);

    // compute minimum distance between each links and its derivative
    double min_dist = 0.01*rb.sel_min_dist_; // unit: [cm]
    Eigen::VectorXd d_min_dist = 0.01*rb.d_sel_min_dist_;

    // compute RBF value of minimum distance and its derivative
    double r = param_.tol_selcol*0.01; // buffer [cm]
    double delta = -0.5; // switching point of RBF
    double RBF = getRBF(delta, min_dist - r);

    if(constraint)
    {
        constraint->setZero();
        if(k != N)
        {
            constraint->c_l = -INF;
            constraint->c_u = 0.0;
            constraint->c = -d_min_dist.dot(dq) + RBF;
        }
    }
    if(Jac)
    {
        Jac->setZero();
        if(k != N)
        {
            Eigen::Matrix<double, TOCABI_DOF, TOCABI_DOF> dd_min_dist = d_min_dist * d_min_dist.transpose(); // hessian matrix (approximation)
            double d_RBF = getDRBF(delta, min_dist - r);
            Jac->c_x_i.block(0,si_index.q1,1,TOCABI_DOF) = (-dd_min_dist*dq + d_RBF*d_min_dist).transpose();
            Jac->c_u_i.block(0,si_index.dq1,1,TOCABI_DOF) = -d_min_dist.transpose();
        }
    }
    return;
}

void Constraints::getSingularConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                                        OneDConstraintInfo *constraint, OneDConstraintsJac* Jac)
{
    // compute singularity constraints
    // -∇_q μ(q)^T * q_dot <= -RBF(μ(q) - ɛ), where ɛ is buffer
    const dJointVector dq = inputTodJointVector(u);

    //  compute manipulability and its derivative
    double manipulability = rb.manipul_; 
    Eigen::VectorXd d_manipulability = rb.d_manipul_;

    // compute RBF value of manipulability and its derivative
    double eps = param_.tol_sing;    // buffer
    double delta = -0.5;  // switching point of RBF
    double RBF = getRBF(delta, manipulability - eps);

    if(constraint)
    {
        constraint->setZero();
        if(k!=N)
        {
            constraint->c_l = -INF;
            constraint->c_u = 0.0;
            constraint->c = -d_manipulability.dot(dq) + RBF;
        }
    }
    if(Jac)
    {
        Jac->setZero();
        if(k!=N)
        {
            Eigen::Matrix<double, TOCABI_DOF, TOCABI_DOF> dd_manipulability = d_manipulability * d_manipulability.transpose(); // hessian matrix (approximation)
            double d_RBF = getDRBF(delta, manipulability - eps);
            Jac->c_x_i.block(0,si_index.q1,1,TOCABI_DOF) = (-dd_manipulability*dq + d_RBF*d_manipulability).transpose();
            Jac->c_u_i.block(0,si_index.dq1,1,TOCABI_DOF) = -d_manipulability.transpose();
        }
    }
    return;
}

void Constraints::getEnvcollConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                                       OneDConstraintInfo *constraint, OneDConstraintsJac* Jac)
{
    // compute environment-collision constraints
    // -∇_q Γ(q)^T * q_dot <= -RBF(Γ(q) - r), where r is buffer
    const JointVector q = stateToJointVector(x);
    const dJointVector dq = inputTodJointVector(u);

    // compute minimum distance between each links and its derivative
    double min_dist = 0.01*rb.env_min_dist_; // unit: [cm]
    Eigen::VectorXd d_min_dist = 0.01*rb.d_env_min_dist_;

    // compute RBF value of minimum distance and its derivative
    double r = param_.tol_envcol*0.01; // buffer [cm]
    double delta = -0.5; // switching point of RBF
    double RBF = getRBF(delta, min_dist - r);

    if(constraint)
    {
        constraint->setZero();
        if(k != N)
        {
            constraint->c_l = -INF;
            constraint->c_u = 0.0;
            constraint->c = -d_min_dist.dot(dq) + RBF;
        }
    }
    if(Jac)
    {
        Jac->setZero();
        if(k != N)
        {
            Eigen::Matrix<double, TOCABI_DOF, TOCABI_DOF> dd_min_dist = d_min_dist * d_min_dist.transpose(); // hessian matrix (approximation)
            double d_RBF = getDRBF(delta, min_dist - r);
            Jac->c_x_i.block(0,si_index.q1,1,TOCABI_DOF) = (-dd_min_dist*dq + d_RBF*d_min_dist).transpose();
            Jac->c_u_i.block(0,si_index.dq1,1,TOCABI_DOF) = -d_min_dist.transpose();
        }
    }
    return;
}

void Constraints::getConstraints(const State &x,const Input &u,const RobotData &rb,int k,
                                 ConstraintsInfo *constraint, ConstraintsJac* Jac)
{
    // compute all the polytopic state constraints
    // compute the three constraints
    OneDConstraintInfo constraint_selcol, constraint_sing, constraint_envcol;
    OneDConstraintsJac jac_selcol, jac_sing, jac_envcol;

    if(Jac)
    {
        getSelcollConstraint(x, u, rb, k, &constraint_selcol, &jac_selcol);
        getSingularConstraint(x, u, rb, k, &constraint_sing, &jac_sing);
        // getEnvcollConstraint(x, u, rb, k, &constraint_envcol, &jac_envcol);
    }
    else
    {
        getSelcollConstraint(x, u, rb, k, &constraint_selcol, NULL);
        getSingularConstraint(x, u, rb, k, &constraint_sing, NULL);
        // getEnvcollConstraint(x, u, rb, k, &constraint_envcol, NULL);
    }

    if(constraint)
    {
        constraint->setZero();
        constraint->c_vec(si_index.con_selcol) = constraint_selcol.c;
        constraint->c_vec(si_index.con_sing) = constraint_sing.c;
        // constraint->c_vec(si_index.con_envcol) = constraint_envcol.c;

        constraint->c_lvec(si_index.con_selcol) = constraint_selcol.c_l;
        constraint->c_lvec(si_index.con_sing) = constraint_sing.c_l;
        // constraint->c_lvec(si_index.con_envcol) = constraint_envcol.c_l;

        constraint->c_uvec(si_index.con_selcol) = constraint_selcol.c_u;
        constraint->c_uvec(si_index.con_sing) = constraint_sing.c_u;
        // constraint->c_uvec(si_index.con_envcol) = constraint_envcol.c_u;
    }
    
    if(Jac)
    {
        Jac->setZero();
        Jac->c_x.row(si_index.con_selcol) = jac_selcol.c_x_i;
        Jac->c_x.row(si_index.con_sing) = jac_sing.c_x_i;
        // Jac->c_x.row(si_index.con_envcol) = jac_envcol.c_x_i;
        Jac->c_u.row(si_index.con_selcol) = jac_selcol.c_u_i;
        Jac->c_u.row(si_index.con_sing) = jac_sing.c_u_i;
        // Jac->c_u.row(si_index.con_envcol) = jac_envcol.c_u_i;
    }
    return;
}
}