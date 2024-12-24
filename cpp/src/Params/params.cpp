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

#include "Params/params.h"
namespace mpcc{
    
Param::Param(){
    std::cout << "Default initialization of model params" << std::endl;
}

Param::Param(std::string file){
    /////////////////////////////////////////////////////
    // Loading Model and Constraint Parameters //////////
    /////////////////////////////////////////////////////
    // std::cout << "model" << std::endl;

    std::ifstream iModel(file);
    json jsonModel;
    iModel >> jsonModel;

    max_dist_proj = jsonModel["max_dist_proj"];
    desired_ee_velocity = jsonModel["desired_ee_velocity"];
    deacc_ratio = jsonModel["deaccelerate_ratio"];

    // initial warm start and trust region (model dependent)
    s_trust_region = jsonModel["s_trust_region"];

    tol_sing = jsonModel["tol_sing"];
    tol_selcol = jsonModel["tol_selcol"];
    tol_envcol = jsonModel["tol_envcol"];

    // std::cout << "max_dist_proj = " << max_dist_proj << std::endl;
    // std::cout << "desired_ee_velocity = " << desired_ee_velocity << std::endl;
    // std::cout << "deaccelerate_ratio = " << deacc_ratio << std::endl;
    // std::cout << "s_trust_region = " << s_trust_region << std::endl;
    // std::cout << "tol_sing = " << tol_sing << std::endl;
    // std::cout << "tol_selcol = " << tol_selcol << std::endl;
}

Param::Param(std::string file, std::map<std::string, double> param){
    /////////////////////////////////////////////////////
    // Loading Model and Constraint Parameters //////////
    /////////////////////////////////////////////////////
    // std::cout << "model" << std::endl;
    
    std::ifstream iModel(file);
    json jsonModel;
    iModel >> jsonModel;

    auto get_param_value = [&](const std::string& key, double& param_var) {
        if(param.find(key) == param.end()) param_var = jsonModel[key];
        else param_var = param[key];
    };

    get_param_value("max_dist_proj", max_dist_proj);
    get_param_value("desired_ee_velocity", desired_ee_velocity);
    get_param_value("deaccelerate_ratio", deacc_ratio);
    get_param_value("s_trust_region", s_trust_region);
    get_param_value("tol_sing", tol_sing);
    get_param_value("tol_selcol", tol_selcol);
    get_param_value("tol_envcol", tol_envcol);

    // std::cout << "max_dist_proj = " << max_dist_proj << std::endl;
    // std::cout << "desired_ee_velocity = " << desired_ee_velocity << std::endl;
    // std::cout << "deaccelerate_ratio = " << deacc_ratio << std::endl;
    // std::cout << "s_trust_region = " << s_trust_region << std::endl;
    // std::cout << "tol_sing = " << tol_sing << std::endl;
    // std::cout << "tol_selcol = " << tol_selcol << std::endl;
}

CostParam::CostParam(){
    std::cout << "Default initialization of cost" << std::endl;
}

CostParam::CostParam(std::string file){
    /////////////////////////////////////////////////////
    // Loading Cost Parameters //////////////////////////
    /////////////////////////////////////////////////////
    // std::cout << "cost" << std::endl;

    std::ifstream iCost(file);
    json jsonCost;
    iCost >> jsonCost;

    q_c = jsonCost["qC"];
    q_c_N_mult = jsonCost["qCNmult"];
    q_l = jsonCost["qL"];
    q_vs = jsonCost["qVs"];

    q_ori = jsonCost["qOri"];

    q_sing = jsonCost["qSing"];

    r_dq = jsonCost["rdq"];
    r_ddq = jsonCost["rddq"];
    r_Vee = jsonCost["rVee"];
    r_dVs = jsonCost["rdVs"];

    q_c_red_ratio = jsonCost["qC_reduction_ratio"];
    q_l_inc_ratio = jsonCost["qL_increase_ratio"];
    q_ori_red_ratio = jsonCost["qOri_reduction_ratio"];

    // std::cout << "q_c = " << q_c << std::endl;
    // std::cout << "q_c_N_mult = " << q_c_N_mult << std::endl;
    // std::cout << "q_l = " << q_l << std::endl;
    // std::cout << "q_vs = " << q_vs << std::endl;
    // std::cout << "q_ori = " << q_ori << std::endl;
    // std::cout << "r_dq = " << r_dq << std::endl;
    // std::cout << "r_Vee = " << r_Vee << std::endl;
    // std::cout << "r_dVs = " << r_dVs << std::endl;
    // std::cout << "q_c_red_ratio = " << q_c_red_ratio << std::endl;
    // std::cout << "q_l_inc_ratio = " << q_l_inc_ratio << std::endl;
    // std::cout << "q_ori_red_ratio = " << q_ori_red_ratio << std::endl;
}

CostParam::CostParam(std::string file, std::map<std::string, double> cost_param){
    /////////////////////////////////////////////////////
    // Loading Cost Parameters //////////////////////////
    /////////////////////////////////////////////////////
    // std::cout << "cost" << std::endl;

    std::ifstream iCost(file);
    json jsonCost;
    iCost >> jsonCost;

    auto get_cost_value = [&](const std::string& key, double& cost_var) {
        if(cost_param.find(key) == cost_param.end()) cost_var = jsonCost[key];
        else cost_var = cost_param[key];
    };

    get_cost_value("qC", q_c);
    get_cost_value("qCNmult", q_c_N_mult);
    get_cost_value("qL", q_l);
    get_cost_value("qVs", q_vs);

    get_cost_value("qOri", q_ori);

    get_cost_value("qSing", q_sing);

    get_cost_value("rdq", r_dq);
    get_cost_value("rddq", r_ddq);
    get_cost_value("rVee", r_Vee);
    get_cost_value("rdVs", r_dVs);

    get_cost_value("qC_reduction_ratio", q_c_red_ratio);
    get_cost_value("qL_increase_ratio", q_l_inc_ratio);
    get_cost_value("qOri_reduction_ratio", q_ori_red_ratio);
    
    // std::cout << "q_c = " << q_c << std::endl;
    // std::cout << "q_c_N_mult = " << q_c_N_mult << std::endl;
    // std::cout << "q_l = " << q_l << std::endl;
    // std::cout << "q_vs = " << q_vs << std::endl;
    // std::cout << "q_ori = " << q_ori << std::endl;
    // std::cout << "r_dq = " << r_dq << std::endl;
    // std::cout << "r_Vee = " << r_Vee << std::endl;
    // std::cout << "r_dVs = " << r_dVs << std::endl;
    // std::cout << "q_c_red_ratio = " << q_c_red_ratio << std::endl;
    // std::cout << "q_l_inc_ratio = " << q_l_inc_ratio << std::endl;
    // std::cout << "q_ori_red_ratio = " << q_ori_red_ratio << std::endl;
}

BoundsParam::BoundsParam() {
    std::cout << "Default initialization of bounds" << std::endl;
}

BoundsParam::BoundsParam(std::string file) {

    /////////////////////////////////////////////////////
    // Loading Cost Parameters //////////////////////////
    /////////////////////////////////////////////////////
    // std::cout << "bounds" << std::endl;

    std::ifstream iBounds(file);
    json jsonBounds;
    iBounds >> jsonBounds;

    lower_state_bounds.q1_l = jsonBounds["q1l"];
    lower_state_bounds.q2_l = jsonBounds["q2l"];
    lower_state_bounds.q3_l = jsonBounds["q3l"];
    lower_state_bounds.q4_l = jsonBounds["q4l"];
    lower_state_bounds.q5_l = jsonBounds["q5l"];
    lower_state_bounds.q6_l = jsonBounds["q6l"];
    lower_state_bounds.q7_l = jsonBounds["q7l"];
    lower_state_bounds.s_l  = jsonBounds["sl"];
    lower_state_bounds.vs_l = jsonBounds["vsl"];

    upper_state_bounds.q1_u = jsonBounds["q1u"];
    upper_state_bounds.q2_u = jsonBounds["q2u"];
    upper_state_bounds.q3_u = jsonBounds["q3u"];
    upper_state_bounds.q4_u = jsonBounds["q4u"];
    upper_state_bounds.q5_u = jsonBounds["q5u"];
    upper_state_bounds.q6_u = jsonBounds["q6u"];
    upper_state_bounds.q7_u = jsonBounds["q7u"];
    upper_state_bounds.s_u  = jsonBounds["su"];
    upper_state_bounds.vs_u = jsonBounds["vsu"];

    lower_input_bounds.dq1_l = jsonBounds["dq1l"];
    lower_input_bounds.dq2_l = jsonBounds["dq2l"];
    lower_input_bounds.dq3_l = jsonBounds["dq3l"];
    lower_input_bounds.dq4_l = jsonBounds["dq4l"];
    lower_input_bounds.dq5_l = jsonBounds["dq5l"];
    lower_input_bounds.dq6_l = jsonBounds["dq6l"];
    lower_input_bounds.dq7_l = jsonBounds["dq7l"];
    lower_input_bounds.dVs_l = jsonBounds["dVsl"];

    upper_input_bounds.dq1_u = jsonBounds["dq1u"];
    upper_input_bounds.dq2_u = jsonBounds["dq2u"];
    upper_input_bounds.dq3_u = jsonBounds["dq3u"];
    upper_input_bounds.dq4_u = jsonBounds["dq4u"];
    upper_input_bounds.dq5_u = jsonBounds["dq5u"];
    upper_input_bounds.dq6_u = jsonBounds["dq6u"];
    upper_input_bounds.dq7_u = jsonBounds["dq7u"];
    upper_input_bounds.dVs_u = jsonBounds["dVsu"];

    lower_ddjoint_bounds.ddq1_l = jsonBounds["ddq1l"];
    lower_ddjoint_bounds.ddq2_l = jsonBounds["ddq2l"];
    lower_ddjoint_bounds.ddq3_l = jsonBounds["ddq3l"];
    lower_ddjoint_bounds.ddq4_l = jsonBounds["ddq4l"];
    lower_ddjoint_bounds.ddq5_l = jsonBounds["ddq5l"];
    lower_ddjoint_bounds.ddq6_l = jsonBounds["ddq6l"];
    lower_ddjoint_bounds.ddq7_l = jsonBounds["ddq7l"];

    upper_ddjoint_bounds.ddq1_u = jsonBounds["ddq1u"];
    upper_ddjoint_bounds.ddq2_u = jsonBounds["ddq2u"];
    upper_ddjoint_bounds.ddq3_u = jsonBounds["ddq3u"];
    upper_ddjoint_bounds.ddq4_u = jsonBounds["ddq4u"];
    upper_ddjoint_bounds.ddq5_u = jsonBounds["ddq5u"];
    upper_ddjoint_bounds.ddq6_u = jsonBounds["ddq6u"];
    upper_ddjoint_bounds.ddq7_u = jsonBounds["ddq7u"];
}

BoundsParam::BoundsParam(std::string file, std::map<std::string, double> bounds_param) {
    std::ifstream iBounds(file);
    json jsonBounds;
    iBounds >> jsonBounds;

    auto get_bound_value = [&](const std::string& key, double& bound_var) {
        if(bounds_param.find(key) == bounds_param.end()) bound_var = jsonBounds[key];
        else bound_var = bounds_param[key];
    };

    get_bound_value("q1l", lower_state_bounds.q1_l);
    get_bound_value("q2l", lower_state_bounds.q2_l);
    get_bound_value("q3l", lower_state_bounds.q3_l);
    get_bound_value("q4l", lower_state_bounds.q4_l);
    get_bound_value("q5l", lower_state_bounds.q5_l);
    get_bound_value("q6l", lower_state_bounds.q6_l);
    get_bound_value("q7l", lower_state_bounds.q7_l);
    get_bound_value("sl", lower_state_bounds.s_l);
    get_bound_value("vsl", lower_state_bounds.vs_l);

    get_bound_value("q1u", upper_state_bounds.q1_u);
    get_bound_value("q2u", upper_state_bounds.q2_u);
    get_bound_value("q3u", upper_state_bounds.q3_u);
    get_bound_value("q4u", upper_state_bounds.q4_u);
    get_bound_value("q5u", upper_state_bounds.q5_u);
    get_bound_value("q6u", upper_state_bounds.q6_u);
    get_bound_value("q7u", upper_state_bounds.q7_u);
    get_bound_value("su", upper_state_bounds.s_u);
    get_bound_value("vsu", upper_state_bounds.vs_u);

    get_bound_value("dq1l", lower_input_bounds.dq1_l);
    get_bound_value("dq2l", lower_input_bounds.dq2_l);
    get_bound_value("dq3l", lower_input_bounds.dq3_l);
    get_bound_value("dq4l", lower_input_bounds.dq4_l);
    get_bound_value("dq5l", lower_input_bounds.dq5_l);
    get_bound_value("dq6l", lower_input_bounds.dq6_l);
    get_bound_value("dq7l", lower_input_bounds.dq7_l);
    get_bound_value("dVsl", lower_input_bounds.dVs_l);

    get_bound_value("dq1u", upper_input_bounds.dq1_u);
    get_bound_value("dq2u", upper_input_bounds.dq2_u);
    get_bound_value("dq3u", upper_input_bounds.dq3_u);
    get_bound_value("dq4u", upper_input_bounds.dq4_u);
    get_bound_value("dq5u", upper_input_bounds.dq5_u);
    get_bound_value("dq6u", upper_input_bounds.dq6_u);
    get_bound_value("dq7u", upper_input_bounds.dq7_u);
    get_bound_value("dVsu", upper_input_bounds.dVs_u);

    get_bound_value("ddq1l", lower_ddjoint_bounds.ddq1_l);
    get_bound_value("ddq2l", lower_ddjoint_bounds.ddq2_l);
    get_bound_value("ddq3l", lower_ddjoint_bounds.ddq3_l);
    get_bound_value("ddq4l", lower_ddjoint_bounds.ddq4_l);
    get_bound_value("ddq5l", lower_ddjoint_bounds.ddq5_l);
    get_bound_value("ddq6l", lower_ddjoint_bounds.ddq6_l);
    get_bound_value("ddq7l", lower_ddjoint_bounds.ddq7_l);

    get_bound_value("ddq1u", upper_ddjoint_bounds.ddq1_u);
    get_bound_value("ddq2u", upper_ddjoint_bounds.ddq2_u);
    get_bound_value("ddq3u", upper_ddjoint_bounds.ddq3_u);
    get_bound_value("ddq4u", upper_ddjoint_bounds.ddq4_u);
    get_bound_value("ddq5u", upper_ddjoint_bounds.ddq5_u);
    get_bound_value("ddq6u", upper_ddjoint_bounds.ddq6_u);
    get_bound_value("ddq7u", upper_ddjoint_bounds.ddq7_u);
}

NormalizationParam::NormalizationParam(){
    std::cout << "Default initialization of normalization" << std::endl;
}

NormalizationParam::NormalizationParam(std::string file)
{
    /////////////////////////////////////////////////////
    // Loading Normalization Parameters /////////////////
    /////////////////////////////////////////////////////
    // std::cout << "norm" << std::endl;

    std::ifstream iNorm(file);
    json jsonNorm;
    iNorm >> jsonNorm;

    T_x.setIdentity();
    T_x(si_index.q1,si_index.q1) = jsonNorm["q1"];
    T_x(si_index.q2,si_index.q2) = jsonNorm["q2"];
    T_x(si_index.q3,si_index.q3) = jsonNorm["q3"];
    T_x(si_index.q4,si_index.q4) = jsonNorm["q4"];
    T_x(si_index.q5,si_index.q5) = jsonNorm["q5"];
    T_x(si_index.q6,si_index.q6) = jsonNorm["q6"];
    T_x(si_index.q7,si_index.q7) = jsonNorm["q7"];
    T_x(si_index.s,si_index.s) = jsonNorm["s"];
    T_x(si_index.vs,si_index.vs) = jsonNorm["vs"];


    T_x_inv.setIdentity();
    for(int i = 0;i<NX;i++)
    {
        T_x_inv(i,i) = 1.0/T_x(i,i);
    }

    T_u.setIdentity();
    T_u(si_index.dq1,si_index.dq1) = jsonNorm["dq1"];
    T_u(si_index.dq2,si_index.dq2) = jsonNorm["dq2"];
    T_u(si_index.dq3,si_index.dq3) = jsonNorm["dq3"];
    T_u(si_index.dq4,si_index.dq4) = jsonNorm["dq4"];
    T_u(si_index.dq5,si_index.dq5) = jsonNorm["dq5"];
    T_u(si_index.dq6,si_index.dq6) = jsonNorm["dq6"];
    T_u(si_index.dq7,si_index.dq7) = jsonNorm["dq7"];
    T_u(si_index.dVs,si_index.dVs) = jsonNorm["dVs"];

    T_u_inv.setIdentity();
    for(int i = 0;i<NU;i++)
    {
        T_u_inv(i,i) = 1.0/T_u(i,i);
    }
}

NormalizationParam::NormalizationParam(std::string file, std::map<std::string, double> normal_praram) {
    std::ifstream iNorm(file);
    json jsonNorm;
    iNorm >> jsonNorm;

    auto get_norm_value = [&](const std::string& key, double& norm_var) {
        if(normal_praram.find(key) == normal_praram.end()) norm_var = jsonNorm[key];
        else norm_var = normal_praram[key];
    };

    T_x.setIdentity();
    get_norm_value("q1", T_x(si_index.q1, si_index.q1));
    get_norm_value("q2", T_x(si_index.q2, si_index.q2));
    get_norm_value("q3", T_x(si_index.q3, si_index.q3));
    get_norm_value("q4", T_x(si_index.q4, si_index.q4));
    get_norm_value("q5", T_x(si_index.q5, si_index.q5));
    get_norm_value("q6", T_x(si_index.q6, si_index.q6));
    get_norm_value("q7", T_x(si_index.q7, si_index.q7));
    get_norm_value("s", T_x(si_index.s, si_index.s));
    get_norm_value("vs", T_x(si_index.vs, si_index.vs));

    T_u_inv.setIdentity();
    for(int i = 0; i < NX; i++) {
        T_x_inv(i, i) = 1.0 / T_x(i, i);
    }

    T_u.setIdentity();
    get_norm_value("dq1", T_u(si_index.dq1, si_index.dq1));
    get_norm_value("dq2", T_u(si_index.dq2, si_index.dq2));
    get_norm_value("dq3", T_u(si_index.dq3, si_index.dq3));
    get_norm_value("dq4", T_u(si_index.dq4, si_index.dq4));
    get_norm_value("dq5", T_u(si_index.dq5, si_index.dq5));
    get_norm_value("dq6", T_u(si_index.dq6, si_index.dq6));
    get_norm_value("dq7", T_u(si_index.dq7, si_index.dq7));
    get_norm_value("dVs", T_u(si_index.dVs, si_index.dVs));

    T_u_inv.setIdentity();
    for(int i = 0; i < NU; i++) {
        T_u_inv(i, i) = 1.0 / T_u(i, i);
    }
}

SQPParam::SQPParam(){
    std::cout << "Default initialization of SQP" << std::endl;
}

SQPParam::SQPParam(std::string file){
    std::ifstream iSQP(file);
    json jsonSQP;
    iSQP >> jsonSQP;

    eps_prim = jsonSQP["eps_prim"];
    eps_dual = jsonSQP["eps_dual"];
    max_iter = jsonSQP["max_iter"];
    line_search_max_iter = jsonSQP["line_search_max_iter"];
    do_SOC = jsonSQP["do_SOC"];
    use_BFGS = jsonSQP["use_BFGS"];

    line_search_tau = jsonSQP["line_search_tau"];
    line_search_eta = jsonSQP["line_search_eta"];
    line_search_rho = jsonSQP["line_search_rho"];
}

SQPParam::SQPParam(std::string file, std::map<std::string, double> sqp_param) {
    std::ifstream iSQP(file);
    json jsonSQP;
    iSQP >> jsonSQP;

    auto get_sqp_value = [&](const std::string& key, double& sqp_var) {
        if(sqp_param.find(key) == sqp_param.end()) sqp_var = jsonSQP[key];
        else sqp_var = sqp_param[key];
    };

    get_sqp_value("eps_prim", eps_prim);
    get_sqp_value("eps_dual", eps_dual);
    get_sqp_value("line_search_tau", line_search_tau);
    get_sqp_value("line_search_eta", line_search_eta);
    get_sqp_value("line_search_rho", line_search_rho);

    if(sqp_param.find("max_iter") == sqp_param.end()) max_iter = jsonSQP["max_iter"];
    else max_iter = static_cast<int>(sqp_param["max_iter"]);

    if(sqp_param.find("line_search_max_iter") == sqp_param.end()) line_search_max_iter = jsonSQP["line_search_max_iter"];
    else line_search_max_iter = static_cast<int>(sqp_param["line_search_max_iter"]);

    if(sqp_param.find("do_SOC") == sqp_param.end()) do_SOC = jsonSQP["do_SOC"];
    else do_SOC = static_cast<bool>(sqp_param["do_SOC"]);

    if(sqp_param.find("use_BFGS") == sqp_param.end()) use_BFGS = jsonSQP["use_BFGS"];
    else use_BFGS = static_cast<bool>(sqp_param["use_BFGS"]);
}
}
