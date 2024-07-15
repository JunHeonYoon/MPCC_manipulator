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

#include "MPC/mpc.h"

namespace mpcc{
MPC::MPC()
:Ts_(1.0)
{
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

MPC::MPC(double Ts,const PathToJson &path)
:Ts_(Ts),
valid_initial_guess_(false),
solver_interface_(new OsqpInterface(Ts, path)),
track_(ArcLengthSpline(path)),
param_(Param(path.param_path)),
integrator_(Integrator(Ts,path)),
robot_(new RobotModel())
{
    initial_guess_.resize(N+1);
}

void MPC::updateInitialGuess(const State &x0)
{
    for(int i=1;i<N;i++) initial_guess_[i-1] = initial_guess_[i];

    initial_guess_[0].xk = x0;
    // initial_guess_[0].uk.setZero();

    initial_guess_[N-1].xk = initial_guess_[N-2].xk;
    initial_guess_[N-1].uk = initial_guess_[N-2].uk; //.setZero();

    initial_guess_[N].xk = integrator_.RK4(initial_guess_[N-1].xk,initial_guess_[N-1].uk,Ts_);
    initial_guess_[N].uk.setZero();

    unwrapInitialGuess();
}

void MPC::unwrapInitialGuess()
{
    double L = track_.getLength();
    for(int i=1;i<=N;i++)
    {
        initial_guess_[i].xk.s = std::min(initial_guess_[i].xk.s,L);
    }
}

void MPC::generateNewInitialGuess(const State &x0)
{
    std::cout<< "generate new initial guess!!"<<std::endl;
    for(int i = 0;i<=N;i++)
    {
        initial_guess_[i].xk = x0;
        initial_guess_[i].uk.setZero();
    }
    unwrapInitialGuess();
    valid_initial_guess_ = true;
}

bool MPC::runMPC(MPCReturn &mpc_return, State &x0)
{
    double last_s = x0.s;
    x0.s = track_.projectOnSpline(last_s, robot_->getEEPosition(stateToJointVector(x0)));
    if(fabs(last_s - x0.s) > param_.max_dist_proj) valid_initial_guess_ = false;

    if(valid_initial_guess_) updateInitialGuess(x0);
    else generateNewInitialGuess(x0);

    solver_interface_->setInitialGuess(initial_guess_);

    Status sqp_status;
    ComputeTime time_nmpc;

    solver_interface_->solveOCP(initial_guess_, &sqp_status, &time_nmpc);
    if(sqp_status == MAX_ITER_EXCEEDED) valid_initial_guess_ = false;
    else if(sqp_status != SOLVED)
    {
        std::cout << "===================================================" << std::endl;
        std::cout << "================ QP did not solved ================" << std::endl;
        switch (sqp_status)
        {
        case QP_DualInfeasible:
            std::cout << "================= Dual Infeasible ================="<< std::endl;
            break;
        case QP_DualInfeasibleInaccurate:
            std::cout << "============ Dual Infeasible Inaccurate ============"<< std::endl;
            break;
        case QP_MaxIterReached:
            std::cout << "================ Max Iter reached =================="<< std::endl;
            break;
        case QP_PrimalInfeasible:
            std::cout << "================= Primal Infeasible ================"<< std::endl;
            break;
        case QP_PrimalInfeasibleInaccurate:
            std::cout << "=========== Primal Infeasible Inaccurate ============"<< std::endl;
            break;
        case QP_SolvedInaccurate:
            std::cout << "================= Solved Inaccurate ================="<< std::endl;
            break;
        case NAN_HESSIAN:
            std::cout << "==================== Nan Hessian ===================="<< std::endl;
            break;
        }
        std::cout << "===================================================" << std::endl;
        valid_initial_guess_ = false;
    }

    mpc_return = {initial_guess_[0].uk,initial_guess_,time_nmpc};
    if(sqp_status == SOLVED) return true;
    else return false;
}

void MPC::setTrack(const Eigen::VectorXd &X, const Eigen::VectorXd &Y,const Eigen::VectorXd &Z,const std::vector<Eigen::Matrix3d> &R)
{
    track_.gen6DSpline(X,Y,Z,R);
    solver_interface_->setTrack(track_);
    valid_initial_guess_ = false;
}

double MPC::getTrackLength()
{
    return track_.getLength();
}

}