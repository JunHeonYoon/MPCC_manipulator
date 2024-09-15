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

#include "Cost/cost.h"
namespace mpcc{
Cost::Cost() 
{
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

Cost::Cost(const PathToJson &path) 
:cost_param_(path.cost_path),
param_(path.param_path)
{
}

Cost::Cost(const PathToJson &path,const ParamValue &param_value) 
:cost_param_(path.cost_path,param_value.cost),
param_(path.param_path,param_value.param)
{
}

double CubicSpline(double x, double x_0, double x_f, double y_0, double y_f)
{
    double x_temp =  (x - x_0) / (x_f - x_0);
    double x_temp2 = pow(x_temp, 2);
    double x_temp3 = pow(x_temp, 3);

    return y_0 + (y_f - y_0) * (3 * x_temp2 - 2 * x_temp3);
}


TrackPoint Cost::getRefPoint(const ArcLengthSpline &track,const State &x)
{
    // compute all the geometry information of the track at a given arc length
    const double s = x.s;

    // X-Y-Z postion of the reference at s
    const Eigen::Vector3d pos_ref = track.getPosition(s);
    const double x_ref = pos_ref(0);
    const double y_ref = pos_ref(1);
    const double z_ref = pos_ref(2);
    // reference path derivatives
    const Eigen::Vector3d dpos_ref = track.getDerivative(s);
    const double dx_ref = dpos_ref(0);
    const double dy_ref = dpos_ref(1);
    const double dz_ref = dpos_ref(2);
    // second order derivatives
    Eigen::Vector3d ddpos_ref = track.getSecondDerivative(s);
    const double ddx_ref = ddpos_ref(0);
    const double ddy_ref = ddpos_ref(1);
    const double ddz_ref = ddpos_ref(1);

    return {x_ref,y_ref,z_ref,dx_ref,dy_ref,dz_ref,ddx_ref,ddy_ref,ddz_ref};
}

TrackOrienatation Cost::getRefOrientation(const ArcLengthSpline &track,const State &x)
{
    // compute all the geometry information of the track at a given arc length
    const double s = x.s;

    // orientation of the reference path
    const Eigen::Matrix3d ori_ref = track.getOrientation(s);
    const Eigen::Vector3d dori_ref = track.getOrientationDerivative(s);

    return {ori_ref,dori_ref};
}

ErrorInfo Cost::getErrorInfo(const ArcLengthSpline &track,const State &x,const RobotData &rb)
{
    ErrorInfo error_info;
    // compute error between reference and X-Y-Z position of the robot EE
    const Eigen::Vector3d pos = rb.EE_position_;
    const TrackPoint track_point = getRefPoint(track,x);
    const Eigen::Vector3d total_error = pos - Eigen::Vector3d(track_point.x_ref,track_point.y_ref, track_point.z_ref);
    
    // lag error
    Eigen::Vector3d Tangent = Eigen::Vector3d(track_point.dx_ref, track_point.dy_ref, track_point.dz_ref); // tangent vector for ref point
    Eigen::Vector3d lag_error = (Tangent.dot(total_error)) * Tangent; // e_l = <T, e> * T

    // contouring error
    Eigen::Vector3d contouring_error = total_error - lag_error;

    // jacobian of the total error with respect to state X
    Eigen::Matrix<double,3,NX> d_total_error;
    d_total_error.setZero();
    d_total_error.block(0,si_index.q1,3,TOCABI_DOF) = rb.Jv_;
    d_total_error.block(0,si_index.s,3,1) = -Tangent;

    // jacobian of the lag error with respect to state X
    Eigen::Matrix<double,3,NX> d_Tangent;
    d_Tangent.setZero();
    d_Tangent.block(0,si_index.s,3,1) = Eigen::Vector3d(track_point.ddx_ref, track_point.ddy_ref, track_point.ddz_ref); // normal vector for ref point
    Eigen::Matrix<double,3,NX> d_lag_error;
    d_lag_error.setZero();
    d_lag_error = (Tangent*Tangent.transpose()) * d_total_error + (Tangent*total_error.transpose() + lag_error.norm()*Eigen::MatrixXd::Identity(3,3)) * d_Tangent;
    
    // jacobian of the contouring error with respect to state X
    Eigen::Matrix<double,3,NX> d_contouring_error;
    d_contouring_error.setZero();
    d_contouring_error = d_total_error - d_lag_error;

    return {contouring_error,lag_error,d_contouring_error,d_lag_error};
}

void Cost::getContouringCost(const ArcLengthSpline &track,const State &x,const RobotData &rb,int k, 
                             double* obj,CostGrad* grad,CostHess* hess)
{
    // compute state cost, formed by contouring error cost + cost on progress of path parameter
    // compute error and jacobian of error
    const ErrorInfo error_info = getErrorInfo(track,x,rb);
    // contouring cost matrix
    Eigen::Vector2d ContouringCost;
    ContouringCost.setZero(2);
    ContouringCost(0) = k < N ? contouring_cost_ : cost_param_.q_c_N_mult * contouring_cost_; // for contouring error
    ContouringCost(1) = lag_cost_; // for lag error


    // progress maximization part
    double s_max = track.getLength();
    double desired_ee_vel = (x.s < s_max * param_.deacc_ratio) ? param_.desired_ee_velocity : -param_.desired_ee_velocity / (s_max * param_.deacc_ratio) * (x.s - s_max);

    // Exact Contouring error cost
    if(obj)
    {
        (*obj) = ContouringCost(0) * error_info.contouring_error.squaredNorm() +
                 ContouringCost(1) * error_info.lag_error.squaredNorm() +
                 cost_param_.q_vs * pow(x.vs - desired_ee_vel,2);
    }

    // Gradient of Contouring error cost
    if(grad)
    {
        grad->setZero();
        grad->f_x = 2.0*ContouringCost(0)*error_info.d_contouring_error.transpose()*error_info.contouring_error +
                    2.0*ContouringCost(1)*error_info.d_lag_error.transpose()*error_info.lag_error;
        grad->f_x(si_index.vs) += 2.0 * cost_param_.q_vs * (x.vs - desired_ee_vel);
    }

    // Hessian of Contouring error cost
    if(hess)
    {
        hess->setZero();
        hess->f_xx = 2.0*ContouringCost(0)*error_info.d_contouring_error.transpose()*error_info.d_contouring_error +
                     2.0*ContouringCost(1)*error_info.d_lag_error.transpose()*error_info.d_lag_error;
        hess->f_xx(si_index.vs,si_index.vs) += 2.0 * cost_param_.q_vs;
    }
    return;
}

void Cost::getHeadingCost(const ArcLengthSpline &track,const State &x,const RobotData &rb,
                          double* obj,CostGrad* grad,CostHess* hess)
{
     // compute heading orientation error cost
    const TrackOrienatation ref_ori = getRefOrientation(track,x);
    const Eigen::Matrix3d ref_R = ref_ori.R_ref;
    const Eigen::Matrix3d cur_R = rb.EE_orientation_;

    const Eigen::Matrix3d R_bar = ref_R.transpose() * cur_R;
    const Eigen::Vector3d Log_R_bar = getInverseSkewVector(LogMatrix(R_bar));
    Eigen::Matrix<double,3,NX> d_Log_R_bar = Eigen::MatrixXd::Zero(3,NX);

    // Exact Heading error cost
    if(obj)
    {
        (*obj) = heading_cost_ * Log_R_bar.squaredNorm();
    }


    if(grad || hess)
    {
        // Linearize heading error function by jacobian. 
        Eigen::Matrix3d J_r_inv;
        if(Log_R_bar.norm() < 1e-8) J_r_inv = Eigen::Matrix3d::Identity();
        else J_r_inv = Eigen::Matrix3d::Identity() + 1./2.*getSkewMatrix(Log_R_bar) + ( 1. / Log_R_bar.squaredNorm() + ( 1. + cos(Log_R_bar.norm()) ) / ( 2. * Log_R_bar.norm() * sin(Log_R_bar.norm()) ) )*getSkewMatrix(Log_R_bar)*getSkewMatrix(Log_R_bar);
        d_Log_R_bar.block(0,si_index.q1,3,TOCABI_DOF) = J_r_inv * cur_R.transpose() * rb.Jw_;
        d_Log_R_bar.block(0,si_index.s,3,1) = -J_r_inv * cur_R.transpose() * ref_ori.dR_ref;
    }

    // Gradient of Heading error cost
    if(grad)
    {
        grad->setZero();
        grad->f_x = 2.0 * heading_cost_ * d_Log_R_bar.transpose() * Log_R_bar;
    }

    // Hessian of Heading error cost
    if(hess)
    {
        hess->setZero();
        hess->f_xx = 2.0 * heading_cost_ * d_Log_R_bar.transpose() * d_Log_R_bar;
    }
    return;
}

void Cost::getInputCost(const ArcLengthSpline &track,const State &x,const Input &u,const RobotData &rb,int k,
                        double* obj,CostGrad* grad,CostHess* hess)
{
    // compute control input cost, formed by joint velocity, joint acceleration, acceleration of path parameter 
    dJointVector dq = inputTodJointVector(u);
    Eigen::Matrix<double, 6, TOCABI_DOF> J = rb.J_;
    Eigen::Matrix<double, 3, TOCABI_DOF> Jv = rb.Jv_;
    TrackPoint track_point = getRefPoint(track, x);
    Eigen::Vector3d Tangent = Eigen::Vector3d(track_point.dx_ref,track_point.dy_ref, track_point.dz_ref);
    Eigen::Vector3d Normal = Eigen::Vector3d(track_point.ddx_ref,track_point.ddy_ref, track_point.ddz_ref);
    
    double path_vel = Tangent.dot(Jv * dq);
    double vel_error = path_vel - x.vs;
    double NJq = Normal.dot(Jv * dq);

    Eigen::VectorXd weighted_rdq = cost_param_.r_dq * Eigen::VectorXd::Ones(TOCABI_DOF);
    weighted_rdq.segment(si_index.dq1, 3) *= cost_param_.r_dq_W_mult; // 3 is number of tocabi waist
    Eigen::MatrixXd rdq_mat = weighted_rdq.asDiagonal();


    // Exact Input cost
    if(obj)
    {
        (*obj) = 0;
        // if(k != N) (*obj) = cost_param_.r_dq * dq.squaredNorm() +
        if(k != N) (*obj) = dq.dot(rdq_mat * dq) +
                            cost_param_.r_dVs * pow(u.dVs,2);
                            // cost_param_.r_Vee * (J * dq).squaredNorm();
                            // cost_param_.r_Vee * pow(vel_error, 2);
    }


    // Gradient of Input cost
    if(grad)
    {
        grad->setZero();
        // grad->f_x(si_index.s) = 2.0 * cost_param_.r_Vee * vel_error * NJq;
        // grad->f_x(si_index.vs) = 2.0 * cost_param_.r_Vee * vel_error * (-1);
        if(k != N)
        {
            // grad->f_u.segment(si_index.dq1,TOCABI_DOF) = 2.0 * cost_param_.r_dq * dq;
            grad->f_u.segment(si_index.dq1,TOCABI_DOF) = 2.0 * rdq_mat * dq;
                                                        // 2.0 * cost_param_.r_Vee * (J.transpose() * J * dq);
                                                        // 2.0 * cost_param_.r_Vee * vel_error * (Jv.transpose() * Tangent);
            grad->f_u(si_index.dVs) = 2.0 * cost_param_.r_dVs*u.dVs;
        }
    }
    
    if(hess)
    {
        hess->setZero();
        // hess->f_xx(si_index.s,si_index.s) = 2.0 * cost_param_.r_Vee * NJq * NJq;
        // hess->f_xx(si_index.vs,si_index.vs) = 2.0 * cost_param_.r_Vee;
        // hess->f_xx(si_index.vs,si_index.s) = 2.0 * cost_param_.r_Vee * (-1) * NJq;
        // hess->f_xx(si_index.s,si_index.vs) = hess->f_xx(si_index.vs,si_index.s);
        if(k != N)
        {
            // hess->f_uu.block(si_index.dq1,si_index.dq1,TOCABI_DOF,TOCABI_DOF) = 2.0 * cost_param_.r_dq * Eigen::MatrixXd::Identity(TOCABI_DOF,TOCABI_DOF);
            hess->f_uu.block(si_index.dq1,si_index.dq1,TOCABI_DOF,TOCABI_DOF) = 2.0 * rdq_mat;
                                                                            //   2.0 * cost_param_.r_Vee * (J.transpose() * J);
                                                                            //   2.0 * cost_param_.r_Vee * (Jv.transpose() * Tangent) * (Tangent.transpose() * Jv);
            hess->f_uu(si_index.dVs,si_index.dVs) = 2.0 * cost_param_.r_dVs;
            // hess->f_xu.block(si_index.s,si_index.dq1,1,TOCABI_DOF) = 2.0 * cost_param_.r_Vee * (NJq * (Tangent.transpose() * Jv) + vel_error * (Normal.transpose() * Jv));
            // hess->f_xu.block(si_index.vs,si_index.dq1,1,TOCABI_DOF) = 2.0 * cost_param_.r_Vee * (-1) * Tangent.transpose() * Jv;
        }
    }

    return;
}

void Cost::getCost(const ArcLengthSpline &track,const State &x,const Input &u,const RobotData &rb,int k,
                   double* obj,CostGrad* grad,CostHess* hess)
{
    double ratio = std::min(rb.sel_min_dist_ / (param_.tol_selcol*2.0), rb.manipul_ / (param_.tol_sing*2.0));
    if(ratio <= 1.0)
    {
        // contouring_cost_ = cost_param_.q_c * ((1.0 - cost_param_.q_c_red_ratio) / (1.0 - 0.5) * (ratio - 0.5) + cost_param_.q_c_red_ratio);
        // lag_cost_ = cost_param_.q_l * ((1.0 - cost_param_.q_l_inc_ratio) / (1.0 - 0.5) * (ratio - 0.5) + cost_param_.q_l_inc_ratio);
        // heading_cost_ = cost_param_.q_ori * ((1.0 - cost_param_.q_ori_red_ratio) / (1.0 - 0.5) * (ratio - 0.5) + cost_param_.q_ori_red_ratio);
        contouring_cost_ = cost_param_.q_c * CubicSpline(ratio, 0.5, 1.0, cost_param_.q_c_red_ratio, 1.0);
        lag_cost_ = cost_param_.q_l * CubicSpline(ratio, 0.5, 1.0, cost_param_.q_l_inc_ratio, 1.0);
        heading_cost_ = cost_param_.q_ori * CubicSpline(ratio, 0.5, 1.0, cost_param_.q_ori_red_ratio, 1.0);
    }
    else
    {
        contouring_cost_ = cost_param_.q_c;
        lag_cost_ = cost_param_.q_l;
        heading_cost_ = cost_param_.q_ori;
    }
    double obj_contouring, obj_heading, obj_input;
    CostGrad grad_contouring, grad_heading, grad_input;
    CostHess hess_contouring, hess_heading, hess_input;

    if(obj && !grad && !hess)
    {
        getContouringCost(track, x, rb, k, &obj_contouring, NULL, NULL);
        getHeadingCost(track, x, rb, &obj_heading, NULL, NULL);
        getInputCost(track, x, u, rb, k, &obj_input, NULL, NULL);

    }
    else if(obj && grad && !hess)
    {
        getContouringCost(track, x, rb, k, &obj_contouring, &grad_contouring, NULL);
        getHeadingCost(track, x, rb, &obj_heading, &grad_heading, NULL);
        getInputCost(track, x, u, rb, k, &obj_input, &grad_input, NULL);
    }
    else if(obj && grad && hess)
    {
        getContouringCost(track, x, rb, k, &obj_contouring, &grad_contouring, &hess_contouring);
        getHeadingCost(track, x, rb, &obj_heading, &grad_heading, &hess_heading);
        getInputCost(track, x, u, rb, k, &obj_input, &grad_input, &hess_input);
    }

    if(obj)
    {
        (*obj) =  obj_contouring + obj_heading + obj_input;
    }

    if(grad)
    {
        grad->f_x = grad_contouring.f_x + grad_heading.f_x + grad_input.f_x;
        grad->f_u = grad_contouring.f_u + grad_heading.f_u + grad_input.f_u;
    }

    if(hess)
    {
        hess->f_xx = hess_contouring.f_xx + hess_heading.f_xx + hess_input.f_xx;
        hess->f_uu = hess_contouring.f_uu + hess_heading.f_uu + hess_input.f_uu;
        hess->f_xu = hess_contouring.f_xu + hess_heading.f_xu + hess_input.f_xu;

        hess->f_xx += Q_MPC::Identity()*1e-6;
        hess->f_uu += R_MPC::Identity()*1e-6;
    }
    return;
}
}