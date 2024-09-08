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

#include "Interfaces/osqp_interface.h"
#include <chrono>

namespace mpcc{
OsqpInterface::OsqpInterface(double Ts,const PathToJson &path)
:cost_(path),
model_(Ts,path),
constraints_(Ts,path),
bounds_(BoundsParam(path.bounds_path),Param(path.param_path)),
normalization_param_(path.normalization_path),
sqp_param_(path.sqp_path),
path_(path),
Ts_(Ts),
device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{   
    robot_ = std::make_unique<RobotModel>();

    selcolNN_ = std::make_unique<SelCollNNmodel>();
    Eigen::VectorXd sel_col_n_hidden(2);
    sel_col_n_hidden << 256, 64;
    selcolNN_->setNeuralNetwork(PANDA_DOF, 1, sel_col_n_hidden, true);

    // envcolNN_ = std::make_unique<EnvCollNNmodel>();
    // envcolNN_->setNeuralNetwork(std::vector<int>{36, 36, 36}, PANDA_DOF);
    // envcolNN_.setNeuralNetwork(std::vector<int>{36, 36, 36}, PANDA_DOF);
    env_model_ = torch::jit::load(pkg_path + "NNmodel/env_collision.pt");
    env_model_.to(device_);

    initial_guess_.resize(N+1);
    rb_.resize(N+1);
    for(size_t i=0; i<rb_.size(); i++) rb_[i].setZero();
}

OsqpInterface::OsqpInterface(double Ts,const PathToJson &path,const ParamValue &param_value)
:cost_(path, param_value),
model_(Ts,path),
constraints_(Ts,path,param_value),
bounds_(BoundsParam(path.bounds_path),Param(path.param_path,param_value.param)),
normalization_param_(path.normalization_path,param_value.normalization),
sqp_param_(path.sqp_path,param_value.sqp),
path_(path),
Ts_(Ts),
device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{   
    robot_ = std::make_unique<RobotModel>();

    selcolNN_ = std::make_unique<SelCollNNmodel>();
    Eigen::VectorXd sel_col_n_hidden(2);
    sel_col_n_hidden << 256, 64;
    selcolNN_->setNeuralNetwork(PANDA_DOF, 1, sel_col_n_hidden, true);

    // envcolNN_ = std::make_unique<EnvCollNNmodel>();
    // envcolNN_->setNeuralNetwork(std::vector<int>{36, 36, 36}, PANDA_DOF);
    // envcolNN_.setNeuralNetwork(std::vector<int>{36, 36, 36}, PANDA_DOF);
    env_model_ = torch::jit::load(pkg_path + "NNmodel/env_collision.pt");
    env_model_.to(device_);


    initial_guess_.resize(N+1);
    rb_.resize(N+1);
    for(size_t i=0; i<rb_.size(); i++) rb_[i].setZero();
}

void OsqpInterface::setTrack(const ArcLengthSpline track)
{
    track_ = track;
}

void OsqpInterface::setParam(const ParamValue &param_value)
{
    cost_ = Cost(path_, param_value);
    constraints_ = Constraints(Ts_,path_,param_value);
    bounds_ = Bounds(BoundsParam(path_.bounds_path),Param(path_.param_path,param_value.param));
}

void OsqpInterface::setEnvData(const std::vector<float> &voxel)
{
    std::vector<float> x_occ;
    std::vector<float> x_q;

    x_occ.resize(voxel.size()*(N+1));
    x_q.resize(PANDA_DOF*(N+1));
    for(size_t i=0; i<=N; i++)
    {
        assert(rb_[i].is_data_valid == true);
        std::copy(voxel.begin(), voxel.end(), x_occ.begin() + i * voxel.size());
        std::transform(rb_[i].q_.data(), rb_[i].q_.data() + rb_[i].q_.size(),
                       x_q.begin() + i * rb_[i].q_.size(),
                       [](double val) { return static_cast<float>(val); });
    }
    // auto env_pred = envcolNN_->forward(x_occ, x_q);
    // auto env_pred = envcolNN_.forward(x_occ, x_q);


    at::Tensor x_q_ten = torch::from_blob(x_q.data(), {N+1, PANDA_DOF}, torch::kFloat32).clone().to(device_);
    at::Tensor x_occ_ten = torch::from_blob(x_occ.data(), {N+1, 1, 36,36,36}, torch::kFloat32).clone().to(device_);
    // std::cout << x_q << std::endl;
    // std::cout << x_occ << std::endl;

    // Set requires_grad_ to true for x_q to enable gradient computation
    x_q_ten.set_requires_grad(true);

    // Forward pass through the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_q_ten);
    inputs.push_back(x_occ_ten);

    // for(auto& val : inputs) std::cout << inputs <<std::endl;

    torch::Tensor output = env_model_.forward(inputs).toTensor().cpu();


    // Convert the output tensor to a vector for easy handling
    Eigen::VectorXf env_min_dist_pred(output.numel());
    std::memcpy(env_min_dist_pred.data(), output.data_ptr<float>(), output.numel() * sizeof(float));

    // Calculate the Jacobian
    int batch_size = output.size(0);
    Eigen::MatrixXf jacobian_pred_T(x_q_ten.size(1), batch_size);
    for (int i=0; i<batch_size; ++i) 
    {
        at::Tensor scalar_output = output[i][0];  // output[i] is a scalar (since output is [10, 1])
    
        // Perform backward pass for each scalar output to compute the gradient
        scalar_output.backward(torch::Tensor(), true);

        // Ensure the gradient is contiguous
        at::Tensor grad = x_q_ten.grad()[i].cpu();

        // Use memcpy to copy the gradient into the Eigen matrix (1D memory block)
        std::memcpy(jacobian_pred_T.col(i).data(), grad.data_ptr<float>(), 7 * sizeof(float));

        // Reset gradients for the next iteration
        x_q_ten.grad().zero_();
    }

    auto env_pred = std::make_pair(env_min_dist_pred.cast<double>(), jacobian_pred_T.transpose().cast<double>());
    // std::cout << "Model inference completed\n";
    // std::cout << "Ans: " << env_pred.first.size() << std:: endl;
    // std::cout << "Jac: " << env_pred.second.rows() << ", "<< env_pred.second.cols() << std:: endl;



    for(size_t i=0; i<=N; i++)
    {
        rb_[i].updateEnv(env_pred.first(i), env_pred.second.row(i).transpose());
    }
}

void OsqpInterface::setInitialGuess(const std::vector<OptVariables> &initial_guess)
{
    for(size_t i=0; i<rb_.size(); i++) rb_[i].setZero();
    initial_guess_ = initial_guess;
    initial_guess_vec_.setZero(N_var);
    for(size_t i=0;i<=N; i++)
    {
        initial_guess_vec_.segment(NX*i, NX) = stateToVector(initial_guess[i].xk);
        if(i != N) initial_guess_vec_.segment(NX*(N+1) + NU*i, NU) = inputToVector(initial_guess[i].uk);

        rb_[i].update(stateToJointVector(initial_guess[i].xk), robot_, selcolNN_);
    }
}

void OsqpInterface::setCost(const std::vector<OptVariables> &initial_guess, 
                            double *obj, Eigen::VectorXd *grad_obj, Eigen::MatrixXd *hess_obj)
{
    if(obj) (*obj) = 0;
    if(grad_obj) grad_obj->setZero(N_var);
    if(hess_obj) hess_obj->setZero(N_var,N_var);
    for(size_t i=0; i<=N; i++)
    {
        // RobotData rbk;
        // rbk.update(stateToJointVector(initial_guess[i].xk), robot_, selcolNN_);
        assert(rb_[i].isUpdated() == true);

        double obj_k;
        CostGrad grad_cost_k;
        CostHess hess_cost_k;
        
        if(obj && !grad_obj && !hess_obj)
        {
            // cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rbk,i, &obj_k, NULL,NULL);
            cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rb_[i],i, &obj_k, NULL,NULL);
        }
        else if(obj && grad_obj && !hess_obj)
        {
            // cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rbk,i, &obj_k, &grad_cost_k,NULL);
            cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rb_[i],i, &obj_k, &grad_cost_k,NULL);
        }
        else if(obj && grad_obj && hess_obj)
        {
            // cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rbk,i, &obj_k, &grad_cost_k,&hess_cost_k);
            cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rb_[i],i, &obj_k, &grad_cost_k,&hess_cost_k);
        }
        if(obj) (*obj) += obj_k;
        if(grad_obj) grad_obj->segment(NX*i, NX) = normalization_param_.T_x*grad_cost_k.f_x;
        if(hess_obj) hess_obj->block(NX*i,NX*i,NX,NX) = normalization_param_.T_x*hess_cost_k.f_xx*normalization_param_.T_x;
        if(i != N)
        {
            if(grad_obj) grad_obj->segment(NX*(N+1)+NU*i, NU) = normalization_param_.T_u*grad_cost_k.f_u;
            if(hess_obj) hess_obj->block(NX*(N+1)+NU*i,NX*(N+1)+NU*i,NU,NU) = normalization_param_.T_u*hess_cost_k.f_uu*normalization_param_.T_u;
            if(hess_obj) hess_obj->block(NX*i,NX*(N+1)+NU*i,NX,NU) = normalization_param_.T_x*hess_cost_k.f_xu*normalization_param_.T_u;
            if(hess_obj) hess_obj->block(NX*(N+1)+NU*i,NX*i,NU,NX) = (normalization_param_.T_x*hess_cost_k.f_xu*normalization_param_.T_u).transpose();
        }
    }
}

void OsqpInterface::setDynamics(const std::vector<OptVariables> &initial_guess,
                                Eigen::MatrixXd *jac_constr_eq, Eigen::VectorXd *constr_eq, Eigen::VectorXd *l_eq, Eigen::VectorXd *u_eq)
{
    if(jac_constr_eq) jac_constr_eq->setZero(N_eq,N_var);
    if(constr_eq) constr_eq->setZero(N_eq);
    if(l_eq) l_eq->setZero(N_eq);
    if(u_eq) u_eq->setZero(N_eq);

    for(size_t i=0;i<=N;i++)
    {
        if(i == 0)
        {
            if(jac_constr_eq) jac_constr_eq->block(0,0,NX,NX) = MatrixXd::Identity(NX,NX);
            if(constr_eq) constr_eq->segment(0,NX) = VectorXd::Zero(NX);
            if(l_eq) l_eq->segment(0,NX) = VectorXd::Zero(NX);
            if(u_eq) u_eq->segment(0,NX) = VectorXd::Zero(NX);
        }
        else
        {
            LinModelMatrix lin_model_prev = model_.getLinModel(initial_guess[i-1].xk,initial_guess[i-1].uk);
            if(jac_constr_eq)
            {
                jac_constr_eq->block(NX*i, NX*(i-1), NX, NX) = -normalization_param_.T_x_inv*lin_model_prev.A*normalization_param_.T_x;
                jac_constr_eq->block(NX*i, NX*i, NX, NX) = MatrixXd::Identity(NX,NX);
                jac_constr_eq->block(NX*i, NX*(N+1) + NU*(i-1), NX, NU) = -normalization_param_.T_x_inv*lin_model_prev.B*normalization_param_.T_u;
            }
            if(constr_eq) constr_eq->segment(NX*i, NX) = normalization_param_.T_x_inv * (stateToVector(initial_guess[i].xk) - (lin_model_prev.A*stateToVector(initial_guess[i-1].xk) + lin_model_prev.B*inputToVector(initial_guess[i-1].uk) +  lin_model_prev.g));
            if(l_eq) l_eq->segment(NX*i, NX) = VectorXd::Zero(NX);
            if(u_eq) u_eq->segment(NX*i, NX) = VectorXd::Zero(NX);
        }
    }
}

void OsqpInterface::setBounds(const std::vector<OptVariables> &initial_guess,
                              Eigen::MatrixXd *jac_constr_ineqb, Eigen::VectorXd *constr_ineqb, Eigen::VectorXd *l_ineqb, Eigen::VectorXd *u_ineqb)
{
    if(jac_constr_ineqb) jac_constr_ineqb->setZero(N_ineqb,N_var);
    if(constr_ineqb) constr_ineqb->setZero(N_ineqb);
    if(l_ineqb) l_ineqb->setZero(N_ineqb);
    if(u_ineqb) u_ineqb->setZero(N_ineqb);

    for(size_t i=0;i<=N;i++)
    {
        if(jac_constr_ineqb) jac_constr_ineqb->block(NX*i, NX*i, NX, NX).setIdentity();
        if(constr_ineqb) constr_ineqb->segment(NX*i, NX) = normalization_param_.T_x_inv*stateToVector(initial_guess[i].xk);
        if(l_ineqb) l_ineqb->segment(NX*i, NX) = normalization_param_.T_x_inv*bounds_.getBoundsLX(initial_guess[i].xk);
        if(u_ineqb) u_ineqb->segment(NX*i, NX) = normalization_param_.T_x_inv*bounds_.getBoundsUX(initial_guess[i].xk,track_.getLength());
        if(i != N)
        {
            if(jac_constr_ineqb) jac_constr_ineqb->block(NX*(N+1) + NU*i, NU*i, NU, NU).setIdentity();
            if(constr_ineqb) constr_ineqb->segment(NX*(N+1) + NU*i, NU) = normalization_param_.T_u_inv*inputToVector(initial_guess[i].uk);
            if(l_ineqb) l_ineqb->segment(NX*(N+1) + NU*i, NU) = normalization_param_.T_u_inv*bounds_.getBoundsLU();
            if(u_ineqb) u_ineqb->segment(NX*(N+1) + NU*i, NU) = normalization_param_.T_u_inv*bounds_.getBoundsUU();
        }
    }
}

void OsqpInterface::setPolytopicConstraints(const std::vector<OptVariables> &initial_guess,
                                            Eigen::MatrixXd *jac_constr_ineqp, Eigen::VectorXd *constr_ineqp, Eigen::VectorXd *l_ineqp, Eigen::VectorXd *u_ineqp)
{
    if(jac_constr_ineqp) jac_constr_ineqp->setZero(N_ineqp,N_var);
    if(constr_ineqp) constr_ineqp->setZero(N_ineqp);
    if(l_ineqp) l_ineqp->setZero(N_ineqp);
    if(u_ineqp) u_ineqp->setZero(N_ineqp);

    for(size_t i=0;i<=N;i++)
    {
        // RobotData rbk;
        // rbk.update(stateToJointVector(initial_guess[i].xk), robot_, selcolNN_);
        assert(rb_[i].isUpdated() == true);

        ConstraintsInfo constr_info_k;
        ConstraintsJac jac_constr_k;
        if(constr_ineqp && !jac_constr_ineqp)
        {
            // constraints_.getConstraints(initial_guess[i].xk,initial_guess[i].uk,rbk,i,&constr_info_k,NULL);
            constraints_.getConstraints(initial_guess[i].xk,initial_guess[i].uk,rb_[i],i,&constr_info_k,NULL);
        }
        else if(constr_ineqp && jac_constr_ineqp)
        {
            // constraints_.getConstraints(initial_guess[i].xk,initial_guess[i].uk,rbk,i,&constr_info_k,&jac_constr_k);
            constraints_.getConstraints(initial_guess[i].xk,initial_guess[i].uk,rb_[i],i,&constr_info_k,&jac_constr_k);
            // std:cout << "self collision constraint["<<i<<"]: " << constr_info_k.c_vec(si_index.con_selcol) << " < " << constr_info_k.c_uvec(si_index.con_selcol) <<std::endl;
        }
        if(jac_constr_ineqp)
        {
            jac_constr_ineqp->block(NPC*i, NX*i, NPC, NX) = jac_constr_k.c_x*normalization_param_.T_x;
            if(i!=N) jac_constr_ineqp->block(NPC*i, NX*(N+1) + NU*i, NPC, NU) = jac_constr_k.c_u*normalization_param_.T_u;
        }
        if(constr_ineqp) constr_ineqp->segment(NPC*i, NPC) = constr_info_k.c_vec;
        if(l_ineqp) l_ineqp->segment(NPC*i, NPC) = constr_info_k.c_lvec;
        if(u_ineqp) u_ineqp->segment(NPC*i, NPC) = constr_info_k.c_uvec;
    }
}

void OsqpInterface::setConstraints(const std::vector<OptVariables> &initial_guess,
                                   Eigen::MatrixXd *jac_constr, Eigen::VectorXd *constr, Eigen::VectorXd *l, Eigen::VectorXd *u)
{
    Eigen::MatrixXd jac_constr_eq, jac_constr_ineqb, jac_constr_ineqp;
    Eigen::VectorXd constr_eq, constr_ineqb, constr_ineqp;
    Eigen::VectorXd l_eq, u_eq, l_ineqb, u_ineqb, l_ineqp, u_ineqp;

    if(constr && !jac_constr)
    {
        setDynamics(initial_guess, NULL, &constr_eq, &l_eq, &u_eq);
        setBounds(initial_guess, NULL, &constr_ineqb, &l_ineqb, &u_ineqb);
        setPolytopicConstraints(initial_guess, NULL, &constr_ineqp, &l_ineqp, &u_ineqp);
    }
    else if(constr && jac_constr)
    {
        setDynamics(initial_guess, &jac_constr_eq, &constr_eq, &l_eq, &u_eq);
        setBounds(initial_guess, &jac_constr_ineqb, &constr_ineqb, &l_ineqb, &u_ineqb);
        setPolytopicConstraints(initial_guess, &jac_constr_ineqp, &constr_ineqp, &l_ineqp, &u_ineqp);
    }
    if(jac_constr)
    {
        jac_constr->block(0, 0, N_eq, N_var) = jac_constr_eq;
        jac_constr->block(N_eq, 0, N_ineqb, N_var) = jac_constr_ineqb;
        jac_constr->block(N_eq+N_ineqb, 0, N_ineqp, N_var) = jac_constr_ineqp;
    }
    if(constr)
    {
        constr->segment(0, N_eq) = constr_eq;
        constr->segment(N_eq, N_ineqb) = constr_ineqb;
        constr->segment(N_eq+N_ineqb, N_ineqp) = constr_ineqp;
    }
    if(l)
    {
        l->segment(0, N_eq) = l_eq;
        l->segment(N_eq, N_ineqb) = l_ineqb;
        l->segment(N_eq+N_ineqb, N_ineqp) = l_ineqp;
    }
    if(u)
    {
        u->segment(0, N_eq) = u_eq;
        u->segment(N_eq, N_ineqb) = u_ineqb;
        u->segment(N_eq+N_ineqb, N_ineqp) = u_ineqp;
    }
}

void OsqpInterface::setQP(const std::vector<OptVariables> &initial_guess,
                          Eigen::MatrixXd *hess_obj, Eigen::VectorXd *grad_obj, double *obj, Eigen::MatrixXd *jac_constr, Eigen::VectorXd *constr, Eigen::VectorXd *l, Eigen::VectorXd *u)
{
    setCost(initial_guess, obj, grad_obj, hess_obj);
    setConstraints(initial_guess, jac_constr, constr, l, u);
}

bool OsqpInterface::solveOCP(std::vector<OptVariables> &opt_sol, Status *status, ComputeTime *mpc_time)
{
    auto start_total = std::chrono::high_resolution_clock::now();

    // Initialize
    lambda_.setZero(N_constr);
    step_.setZero(N_var);
    step_prev_.setZero(N_var);
    step_lambda_.setZero(N_constr);

    grad_L_.setZero(N_var);
    delta_grad_L_.setZero(N_var);

    Hess_.setZero(N_var, N_var);
    grad_obj_.setZero(N_var);
    jac_constr_.setZero(N_constr, N_var);
    constr_.setZero(N_constr);
    l_.setZero(N_constr);
    u_.setZero(N_constr);

    filter_data_list_.clear();

    mpc_time->setZero();

    std::vector<OptVariables> zero_guess;
    zero_guess.resize(N+1);
    for(size_t i=0; i<=N; i++)
    {
        zero_guess[i].xk = initial_guess_[0].xk;
        zero_guess[i].uk.setZero();
    }

    // SQP itertion
    for(sqp_iter_=0; sqp_iter_<sqp_param_.max_iter; sqp_iter_++)
    {
        // std::cout <<"sqp_iter_: " <<sqp_iter_<<std::endl;

        auto start_set_qp = std::chrono::high_resolution_clock::now();

        // QP formulation
        if(sqp_param_.use_BFGS)
        {
            if(sqp_iter_ == 0) setQP(initial_guess_,&Hess_, &grad_obj_, &obj_, &jac_constr_, &constr_, &l_, &u_);
            else setQP(initial_guess_,NULL, &grad_obj_, &obj_, &jac_constr_, &constr_, &l_, &u_);
        }
        else
        {
            setQP(initial_guess_,&Hess_, &grad_obj_, &obj_, &jac_constr_, &constr_, &l_, &u_);
        }

        delta_grad_L_ = -grad_L_;
        grad_L_ = grad_obj_ + jac_constr_.transpose() * lambda_;
        delta_grad_L_ += grad_L_;  // delta_grad_L_ = grad_L - grad_L_prev

        // get Hessian
        if(sqp_param_.use_BFGS && sqp_iter_ != 0)  Hess_ = BFGSUpdate(Hess_, step_prev_, delta_grad_L_);
        if (!isPosdef(Hess_)) 
        {
            std::cout << "Hessian not positive definite\n";
            // double tau = 1e-3;
            // Eigen::VectorXd v(N_var);
            // while (!isPosdef(Hess_)) 
            // {
            //     v.setConstant(tau);
            //     Hess_ += v.asDiagonal();
            //     tau *= 10;
            // }
            (*status) = NON_PD_HESSIAN;
            break;
        }
        if (isNan(Hess_)) 
        {
            std::cout << "Hessian is NaN\n";
            (*status) = NAN_HESSIAN;
            break;
        }

        auto end_set_qp = std::chrono::high_resolution_clock::now();
        auto start_solve_qp = std::chrono::high_resolution_clock::now();

        // solve QP to get step_ and step_lambda_
        if(!solveQP(Hess_, grad_obj_, jac_constr_, l_-constr_, u_-constr_, step_, step_lambda_, qp_status_))
        {
            switch (qp_status_)
            {
            case OsqpEigen::Status::DualInfeasibleInaccurate:
                (*status) = QP_DualInfeasibleInaccurate;
                break;
            case OsqpEigen::Status::PrimalInfeasibleInaccurate:
                (*status) = QP_PrimalInfeasibleInaccurate;
                break;
            case OsqpEigen::Status::SolvedInaccurate:
                (*status) = QP_SolvedInaccurate;
                break;
            case OsqpEigen::Status::MaxIterReached:
                (*status) = QP_MaxIterReached;
                break;
            case OsqpEigen::Status::PrimalInfeasible:
                (*status) = QP_PrimalInfeasible;
                break;
            case OsqpEigen::Status::DualInfeasible:
                (*status) = QP_DualInfeasible;
                break;
            case OsqpEigen::Status::Sigint:
                (*status) = Sigint;
                break;
            }
        }
        if(sqp_param_.do_SOC)
        {
            if(!SecondOrderCorrection(initial_guess_,Hess_,grad_obj_,jac_constr_,step_,step_lambda_, qp_status_))
            {
                switch (qp_status_)
                {
                case OsqpEigen::Status::DualInfeasibleInaccurate:
                    (*status) = QP_DualInfeasibleInaccurate;
                    break;
                case OsqpEigen::Status::PrimalInfeasibleInaccurate:
                    (*status) = QP_PrimalInfeasibleInaccurate;
                    break;
                case OsqpEigen::Status::SolvedInaccurate:
                    (*status) = QP_SolvedInaccurate;
                    break;
                case OsqpEigen::Status::MaxIterReached:
                    (*status) = QP_MaxIterReached;
                    break;
                case OsqpEigen::Status::PrimalInfeasible:
                    (*status) = QP_PrimalInfeasible;
                    break;
                case OsqpEigen::Status::DualInfeasible:
                    (*status) = QP_DualInfeasible;
                    break;
                case OsqpEigen::Status::Sigint:
                    (*status) = Sigint;
                    break;
                }
            }
        }

        auto end_solve_qp = std::chrono::high_resolution_clock::now();
        auto start_get_alpha = std::chrono::high_resolution_clock::now();

        step_lambda_ -= lambda_;

        // double alpha = meritLineSearch(step_, Hess_, grad_obj_, obj_, constr_, l_, u_);
        double alpha = filterLineSearch(initial_guess_,step_,filter_data_list_);
        // std:cout << "\talpha: "<<alpha << std::endl;

        auto end_get_alpha = std::chrono::high_resolution_clock::now();

        // take step
        initial_guess_vec_ += alpha*deNormalizeStep(step_);
        lambda_ += alpha*step_lambda_;
        initial_guess_ = vectorToOptvar(initial_guess_vec_);
        // printOptVar( initial_guess_);

        // update step info
        step_prev_ = alpha * step_;
        primal_step_norm_ = alpha * step_.template lpNorm<Eigen::Infinity>();
        // primal_step_norm_ = (alpha * step_).norm();
        dual_step_norm_ = alpha * step_lambda_.template lpNorm<Eigen::Infinity>();
        // std::cout << "\tprimal_step_norm_: " << primal_step_norm_ << std::endl;
        // std::cout << "\tdual_step_norm_: " << dual_step_norm_ << std::endl;

        mpc_time->set_qp += std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp - start_set_qp).count();
        mpc_time->solve_qp += std::chrono::duration_cast<std::chrono::duration<double>>(end_solve_qp - start_solve_qp).count();
        mpc_time->get_alpha += std::chrono::duration_cast<std::chrono::duration<double>>(end_get_alpha - start_get_alpha).count();

        // termination condition
        // TODO: critertion about constraint
        // if(primal_step_norm_ < sqp_param_.eps_prim && dual_step_norm_ < sqp_param_.eps_dual)
        if(primal_step_norm_ < sqp_param_.eps_prim)
        {
            (*status) = SOLVED;
            break;
        }
    }
    if(sqp_iter_ == sqp_param_.max_iter) (*status) = MAX_ITER_EXCEEDED;

    auto end_total = std::chrono::high_resolution_clock::now();
    mpc_time->total = std::chrono::duration_cast<std::chrono::duration<double>>(end_total - start_total).count();

    if((*status) == SOLVED)
    {
        opt_sol = initial_guess_;
        return true;
    }
    else
    {
        opt_sol = zero_guess;
        return false;
    }
}

bool OsqpInterface::solveQP(const Eigen::MatrixXd &P, const Eigen::VectorXd &q, const Eigen::MatrixXd &A, const Eigen::VectorXd &l,const Eigen::VectorXd &u,
                            Eigen::VectorXd &step, Eigen::VectorXd &step_lambda, OsqpEigen::Status &qp_status)
{
    /*
    min   1/2 x' P x + q' x
     x

    subject to
    l <= A x <= u

    with :
    P sparse (n x n) positive definite
    q dense  (n x 1)
    A sparse (nc x n)
    l dense  (nc x 1)
    u dense  (nc x 1)
    */
   Eigen::SparseMatrix<double> P_sp(N_var, N_var);
   Eigen::SparseMatrix<double> A_sp(N_constr, N_var);
   Eigen::Matrix<double, N_var,1> q_ds;
   Eigen::Matrix<double, N_constr,1> l_ds;
   Eigen::Matrix<double, N_constr,1> u_ds;
   P_sp = P.sparseView();
   A_sp = A.sparseView();
   q_ds = q;
   l_ds = l;
   u_ds = u;

    OsqpEigen::Solver solver_;
    // settings
    solver_.settings()->setWarmStart(false);
    solver_.settings()->getSettings()->eps_abs = 1e-4;
    solver_.settings()->getSettings()->eps_rel = 1e-5;
    solver_.settings()->getSettings()->verbose = false;

    // set the initial data of the QP solver
    solver_.data()->setNumberOfVariables(N_var);
    solver_.data()->setNumberOfConstraints(N_constr);
    if (!solver_.data()->setHessianMatrix(P_sp))           return false;
    if (!solver_.data()->setGradient(q_ds))                return false;
    if (!solver_.data()->setLinearConstraintsMatrix(A_sp)) return false;
    if (!solver_.data()->setLowerBound(l_ds))              return false;
    if (!solver_.data()->setUpperBound(u_ds))              return false;

    // instantiate the solver
    if (!solver_.initSolver()) return false;

    // solve the QP problem
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return false;
    qp_status = solver_.getStatus();
    if (solver_.getStatus() != OsqpEigen::Status::Solved) return false;
    // if (solver_.getStatus() != OsqpEigen::Status::Solved && solver_.getStatus() != OsqpEigen::Status::SolvedInaccurate) return false;

    // get the controller input
    step = solver_.getSolution();
    step_lambda = solver_.getDualSolution();

    solver_.clearSolverVariables();
    solver_.clearSolver();

    return true; 
}

bool OsqpInterface::SecondOrderCorrection(const std::vector<OptVariables> &initial_guess, const Eigen::MatrixXd &hess_obj, const Eigen::VectorXd &grad_obj, const Eigen::MatrixXd & jac_constr,
                                          Eigen::VectorXd &step, Eigen::VectorXd &step_lambda, OsqpEigen::Status& qp_status)
{
    const std::vector<OptVariables> updates_initial_guess = vectorToOptvar(OptvarToVector(initial_guess) + step);

    Eigen::VectorXd constr_step;
    Eigen::VectorXd l_step, u_step;
    constr_step.setZero(N_constr);
    l_step.setZero(N_constr);
    u_step.setZero(N_constr);
    setConstraints(updates_initial_guess, NULL, &constr_step, &l_step, &u_step);

    // Constraints
    // from   l <= A.x + b <= u
    // to   l-b <= A.x     <= u-b
    Eigen::MatrixXd P = hess_obj;
    Eigen::VectorXd q = grad_obj;
    Eigen::MatrixXd A = jac_constr;
    Eigen::VectorXd d = constr_step - A*step;
    Eigen::VectorXd l = l_step - d;
    Eigen::VectorXd u = u_step - d;

    return solveQP(P, q, A, l, u, step, step_lambda, qp_status);
}

Eigen::MatrixXd OsqpInterface::BFGSUpdate(const Eigen::MatrixXd &Hess, const Eigen::VectorXd &step_prev, const Eigen::VectorXd &delta_grad_L)
{
    // Damped BFGS update
    // Implements "Procedure 18.2 Damped BFGS updating for SQP" form Numerical Optimization by Nocedal.
    double sy, sr, sBs;
    Eigen::VectorXd Bs, r;

    Bs = Hess * step_prev;
    sBs = step_prev.dot(Bs);
    sy = step_prev.dot(delta_grad_L);

    if (sy < 0.2 * sBs) 
    {
        // damped update to enforce positive definite B
        double theta;
        theta = 0.8 * sBs / (sBs - sy);
        r = theta * delta_grad_L + (1 - theta) * Bs;
        sr = theta * sy + (1 - theta) * sBs;
    } 
    else 
    {
        // unmodified BFGS
        r = delta_grad_L;
        sr = sy;
    }

    if (sr < std::numeric_limits<double>::epsilon()) 
    {
        return Hess;
    }

    return Hess - Bs * Bs.transpose() / sBs + r * r.transpose() / sr;
}

double OsqpInterface::meritLineSearch(const Eigen::VectorXd &step, const Eigen::MatrixXd &Hess, const Eigen::VectorXd &grad_obj, const double &obj, const Eigen::VectorXd &constr, const Eigen::VectorXd &l, const Eigen::VectorXd &u)
{
    double mu, phi_l1, Dp_phi_l1;
    const double tau = sqp_param_.line_search_tau; // line search step decrease, 0 < tau < settings.tau

    double constr_l1 = constraint_norm(constr, l, u);

    // get mu from merit function model using hessian of Lagrangian instead
    mu = (grad_obj.dot(step) + 0.5 * step.dot(Hess * step)) / ((1 - sqp_param_.line_search_rho) * constr_l1);

    phi_l1 = obj + mu * constr_l1;
    Dp_phi_l1 = grad_obj.dot(step) - mu * constr_l1;

    double updated_obj;
    Eigen::VectorXd updated_constr;
    Eigen::VectorXd updated_l, updated_u;

    double alpha = 1.0;
    for(size_t i=0; i<sqp_param_.line_search_max_iter; i++)
    {
        updated_constr.setZero(N_constr);
        updated_l.setZero(N_constr);
        updated_u.setZero(N_constr);

        Eigen::VectorXd updated_initial_guess_vec = initial_guess_vec_ + alpha * deNormalizeStep(step);
        std::vector<OptVariables> updated_initial_guess = vectorToOptvar(updated_initial_guess_vec);
        setQP(updated_initial_guess, NULL, NULL, &updated_obj, NULL, &updated_constr, &updated_l, &updated_u);

        double updated_phi_l1 = updated_obj + mu * constraint_norm(updated_constr, updated_l, updated_u);
        if (updated_phi_l1 <= phi_l1 + alpha * sqp_param_.line_search_eta * Dp_phi_l1) 
        {
            // accept step
            break;
        } 
        else 
        {
            alpha = tau * alpha;
        }
    }
    return alpha;
}

double OsqpInterface::filterLineSearch(const std::vector<OptVariables> &initial_guess, const Eigen::VectorXd &step, std::vector<FilterData> &filter_data_list)
{
    FilterData updated_filter_data;
    Eigen::VectorXd updated_constr, updated_l, updated_u;
    updated_constr.setZero(N_constr);
    updated_l.setZero(N_constr);
    updated_u.setZero(N_constr);

    bool is_alpha_accepted = true;

    double alpha = 1.0;
    for(size_t i=0; i<sqp_param_.line_search_max_iter; i++)
    {
        // get updated cost and constraint wrt x+delta_x
        const std::vector<OptVariables> updated_initial_guess = vectorToOptvar(OptvarToVector(initial_guess) + alpha*deNormalizeStep(step));
        setQP(updated_initial_guess, NULL, NULL, &updated_filter_data.obj, NULL, &updated_constr, &updated_l, &updated_u);
        updated_filter_data.gap_vio = constraint_norm(updated_constr, updated_l, updated_u);
        // filtering updated data
        for(size_t j=0; j<filter_data_list.size(); j++)
        {
            if(updated_filter_data.obj >= filter_data_list[j].obj && updated_filter_data.gap_vio >= filter_data_list[j].gap_vio)
            {
                is_alpha_accepted = false;
                break;
            }
        }
        if(is_alpha_accepted)
        {
            // update filter data list
            std::vector<FilterData> updated_filter_data_list;
            updated_filter_data_list.clear();
            for(size_t j=0; j<filter_data_list.size(); j++)
            {
                if(updated_filter_data.obj > filter_data_list[j].obj || updated_filter_data.gap_vio > filter_data_list[j].gap_vio)
                {
                    updated_filter_data_list.push_back(filter_data_list[j]);
                }
            }
            updated_filter_data_list.push_back(updated_filter_data);
            filter_data_list = updated_filter_data_list;
            break;
        }
        else
        {
            // decrease alpha
            alpha *= sqp_param_.line_search_tau;
        }
    }
    return alpha;
}

bool OsqpInterface::isPosdef(const Eigen::MatrixXd& H)
{
    Eigen::LLT<MatrixXd> llt(H);
    if (llt.info() == Eigen::NumericalIssue) {
        return false;
    }
    return true;
}

bool OsqpInterface::isNan(const Eigen::MatrixXd& x)
{
    return x.array().isNaN().any();
}

double OsqpInterface::constraint_norm(const Eigen::VectorXd &constr, const Eigen::VectorXd &l, const Eigen::VectorXd &u)
{
    double c_l1 = 0;

    // l <= c(x) <= u
    c_l1 += (l - constr).cwiseMax(0.0).sum();
    c_l1 += (constr - u).cwiseMax(0.0).sum();

    return c_l1;
}

std::vector<OptVariables> OsqpInterface::vectorToOptvar(const Eigen::VectorXd& opt_var_vec)
{
    std::vector<OptVariables> opt_var;
    opt_var.resize(N+1);
    for(size_t i=0;i<=N;i++)
    {
        opt_var[i].xk = vectorToState(opt_var_vec.segment(NX*i, NX));
        if(i!=N) opt_var[i].uk = vectorToInput(opt_var_vec.segment(NX*(N+1)+NU*i, NU));
    }
    return opt_var;
}

Eigen::VectorXd OsqpInterface::OptvarToVector(const std::vector<OptVariables>& opt_var)
{
    Eigen::VectorXd opt_var_vec;
    opt_var_vec.setZero(N_var);
    for(size_t i=0;i<=N;i++)
    {
        opt_var_vec.segment(NX*i,NX) = stateToVector(opt_var[i].xk);
        if(i!=N) opt_var_vec.segment(NX*(N+1)+NU*i, NU) = inputToVector(opt_var[i].uk);
    }
    return opt_var_vec;
}

Eigen::VectorXd OsqpInterface::deNormalizeStep(const Eigen::VectorXd& step)
{
    Eigen::VectorXd denormalized_step;
    denormalized_step.setZero(N_var);
    for(size_t i=0;i<=N;i++) 
    {
        denormalized_step.segment(NX*i, NX) = normalization_param_.T_x*step.segment(NX*i, NX);
        if(i!=N) denormalized_step.segment(NX*(N+1) + NU*i, NU) = normalization_param_.T_u*step.segment(NX*(N+1) + NU*i, NU);
    }
    return denormalized_step;
}

void OsqpInterface::printOptVar(std::vector<OptVariables> opt_var)
{
    for(size_t i=0;i<=N;i++)
    {
        std::cout << "State[" << i << "]: " << std::endl;; 
        std::cout << "\tq    : " << stateToJointVector(opt_var[i].xk).transpose() << std::endl;
        std::cout << "\ts    : " << opt_var[i].xk.s << std::endl;
        std::cout << "\ts dot: " << opt_var[i].xk.vs << std::endl;
    }
    std::cout <<" \n";
    for(size_t i=0;i<N;i++)
    {
        std::cout << "Input[" << i << "]: " << std::endl;; 
        std::cout << "\tqdot: " << inputTodJointVector(opt_var[i].uk).transpose() << std::endl;; 
        std::cout << "\tdVs  : " << opt_var[i].uk.dVs << std::endl;; 
    }
}
}