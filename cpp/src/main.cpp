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

#include <iostream>
#include <fstream>
#include "MPC/mpc.h"
#include "Model/integrator.h"
#include "Params/track.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#ifdef USE_CVPLOT
#include <CvPlot/cvplot.h>
#include <opencv2/opencv.hpp>
#endif

int main() {

    using namespace mpcc;
    std::ifstream iConfig(pkg_path + "Params/config.json");
    json jsonConfig;
    iConfig >> jsonConfig;

    PathToJson json_paths {pkg_path + std::string(jsonConfig["model_path"]),
                           pkg_path + std::string(jsonConfig["cost_path"]),
                           pkg_path + std::string(jsonConfig["bounds_path"]),
                           pkg_path + std::string(jsonConfig["track_path"]),
                           pkg_path + std::string(jsonConfig["normalization_path"]),
                           pkg_path + std::string(jsonConfig["sqp_path"])};

    // std::map<std::string, double> param, cost_param;
    // param["desired_ee_velocity"] = 0.2;
    // cost_param["qOri_reduction_ratio"] = 0.1;
    // ParamValue param_value = {param, cost_param};


    Integrator integrator = Integrator(jsonConfig["Ts"],json_paths);
    RobotModel robot = RobotModel();
    SelCollNNmodel selcolNN = SelCollNNmodel();
    selcolNN.setNeuralNetwork(PANDA_DOF, 1,(Eigen::VectorXd(2) << 256 , 64).finished(), true);

    std::vector<MPCReturn> log;
    MPC mpc(jsonConfig["Ts"],json_paths);
    // MPC mpc(jsonConfig["Ts"],json_paths,param_value);

    State x0 = {0., 0., 0., -M_PI/2, 0, M_PI/2, M_PI/4,
                0., 0., 0., 0., 0., 0., 0.,
                0., 0.};
    Eigen::Vector3d ee_pos = robot.getEEPosition(stateToJointVector(x0));
    Eigen::Matrix3d ee_ori = robot.getEEOrientation(stateToJointVector(x0));

    Track track = Track(json_paths.track_path);
    TrackPos track_xyzr = track.getTrack(ee_pos);
    mpc.setTrack(track_xyzr.X,track_xyzr.Y,track_xyzr.Z,track_xyzr.R);

    Eigen::Vector3d end_point;
    Eigen::Matrix3d end_ori;
    double end_s;
    end_point(0) = track_xyzr.X(track_xyzr.X.size() - 1); 
    end_point(1) = track_xyzr.Y(track_xyzr.Y.size() - 1);
    end_point(2) = track_xyzr.Z(track_xyzr.Z.size() - 1);
    end_ori = track_xyzr.R[track_xyzr.R.size() - 1];
    end_s = mpc.getTrackLength();
    std::cout<<"end posi: "<<end_point.transpose()<<std::endl;
    
    mpcc::ArcLengthSpline spline_track;
    spline_track.gen6DSpline(track_xyzr.X,track_xyzr.Y,track_xyzr.Z,track_xyzr.R);
    mpcc::PathData spline_path = spline_track.getPathData();

    ofstream debug_file, splined_path_file;
    debug_file.open("debug.txt");
    splined_path_file.open("splined_path.txt");

    for(size_t i=0; i<spline_path.n_points; i++)
    {
        Eigen::Quaterniond quaternion(spline_path.R[i]);
        splined_path_file << spline_path.X(i) << " "
                          << spline_path.Y(i) << " "
                          << spline_path.Z(i) << " "
                          << quaternion.x() << " "
                          << quaternion.y() << " "
                          << quaternion.z() << " "
                          << quaternion.w() << std::endl;
    }

    for(int i = 0; i < jsonConfig["n_sim"]; i++)
    {
        MPCReturn mpc_sol;
        // if(i == 200)
        // {
        //     mpc.setParam(param_value);
        // }
        bool mpc_status = mpc.runMPC(mpc_sol, x0);
        if(mpc_status == false)
        {
            std::cout<<"MPC did not solved properly!!"<<std::endl;
            break;
        }
        x0 = integrator.simTimeStep(x0,mpc_sol.u0,jsonConfig["Ts"]);
        ee_pos = robot.getEEPosition(stateToJointVector(x0));
        ee_ori = robot.getEEOrientation(stateToJointVector(x0));

        std::cout << "==============================================================="<<std::endl;
        std::cout << "time step: " << i << std::endl;;
        std::cout << "q now           :\t";
        std::cout << std::fixed << std::setprecision(6) << stateToJointVector(x0).transpose() << std::endl;
        std::cout << "q_dot now       :\t";
        std::cout << std::fixed << std::setprecision(6) << stateTodJointVector(x0).transpose()  << std::endl;
        std::cout << "q_ddot now       :\t";
        std::cout << std::fixed << std::setprecision(6) << inputToddJointVector(mpc_sol.u0).transpose()  << std::endl;
        std::cout << "x               :\t";
        std::cout << std::fixed << std::setprecision(6) << ee_pos.transpose() << std::endl;
        std::cout << "x_dot           :\t";
        std::cout << std::fixed << std::setprecision(6) << (robot.getJacobianv(stateToJointVector(x0))*stateTodJointVector(x0)).transpose() << std::endl;
        std::cout << std::fixed << std::setprecision(6) << (robot.getJacobianv(stateToJointVector(x0))*stateTodJointVector(x0)).norm() << std::endl;
        std::cout << "R               :" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << ee_ori << std::endl;
        std::cout << "manipulability  :\t";
        std::cout << std::fixed << std::setprecision(6) << robot.getManipulability(stateToJointVector(x0)) << std::endl;
        std::cout << "min distance[cm]:\t";
        std::cout << std::fixed << std::setprecision(6) << (selcolNN.calculateMlpOutput(stateToJointVector(x0),false)).first << std::endl;
        std::cout << "s               :";
        std::cout << std::fixed << std::setprecision(6) << x0.s << std::endl;
        std::cout << "vs              :";
        std::cout << std::fixed << std::setprecision(6) << x0.vs << std::endl;
        std::cout << "dVs              :";
        std::cout << std::fixed << std::setprecision(6) << mpc_sol.u0.dVs << std::endl;
        std::cout << "x_error         :";
        std::cout << std::fixed << std::setprecision(6) << (end_point - ee_pos).transpose() << std::endl;
        std::cout << "R error         :";
        std::cout << std::fixed << std::setprecision(6) << getInverseSkewVector(LogMatrix(end_ori.transpose()*ee_ori)).transpose() << std::endl;
        std::cout << "MPC time        :";
        std::cout << std::fixed << std::setprecision(6) << mpc_sol.compute_time.total << std::endl;
        std::cout << "==============================================================="<<std::endl;
        
        debug_file << stateToJointVector(x0).transpose() << " " 
                   << stateTodJointVector(x0).transpose() << " " 
                   << inputToddJointVector(mpc_sol.u0).transpose() << " " 
                   << (selcolNN.calculateMlpOutput(stateToJointVector(x0),false)).first << " "
                   << robot.getManipulability(stateToJointVector(x0)) << " ";
                //    << std::endl;
        for(size_t j=0;j<N; j++)
        {
            auto pred_ee_pos = robot.getEEPosition(stateToJointVector(mpc_sol.mpc_horizon[j].xk));
            auto pred_ee_rot = robot.getEEOrientation(stateToJointVector(mpc_sol.mpc_horizon[j].xk));
            Eigen::Quaterniond pred_ee_quat(pred_ee_rot);
            debug_file << pred_ee_pos.transpose() << " ";
            debug_file << pred_ee_quat.coeffs().transpose() << " ";
        }
        for(size_t j=0;j<N; j++)
        {
            auto ref_ee_pos = spline_track.getPosition(mpc_sol.mpc_horizon[j].xk.s);
            auto ref_ee_rot = spline_track.getOrientation(mpc_sol.mpc_horizon[j].xk.s);
            Eigen::Quaterniond ref_ee_quat(ref_ee_rot);
            debug_file << ref_ee_pos.transpose() << " ";
            debug_file << ref_ee_quat.coeffs().transpose() << " ";
        }
        debug_file << "\n";

        log.push_back(mpc_sol);

        if((end_point - ee_pos).norm() < 1e-2 && (getInverseSkewVector(LogMatrix(end_ori.transpose()*ee_ori))).norm() < 1e-3 && fabs(x0.s - end_s) < 1e-2)
        {
            std::cout << "End point reached!!!"<< std::endl;
            break;
        }
    }

    double mean_time = 0.0;
    double max_time = 0.0;
    // for(MPCReturn log_i : log)
    for(size_t i=0;i<log.size();i++)
    {
        MPCReturn log_i = log[i];
        mean_time += log_i.compute_time.total;
        if(log_i.compute_time.total > max_time)
            max_time = log_i.compute_time.total;
    }
    std::cout << "mean nmpc time [sec]: " << mean_time/log.size() << std::endl;
    std::cout << "max nmpc time [sec]: " << max_time << std::endl;

    // ========== Plot MPC computation time ==========
    #ifdef USE_CVPLOT

    struct LegendLabel :public CvPlot::Drawable{
        std::string _text;
        cv::Point _position;
        const int _fontFace = cv::FONT_HERSHEY_SIMPLEX;
        const double _fontScale = .4;
        const int _fontThickness = 1;
        cv::Scalar _color = cv::Scalar(0, 0, 0);
        void render(CvPlot::RenderTarget& renderTarget) override {
            int baseline;
            cv::Size size = cv::getTextSize(_text, _fontFace, _fontScale, _fontThickness, &baseline);
            auto pos = renderTarget.innerToOuter(renderTarget.project(_position)) + cv::Point2d(size.height * 2, size.height / 2);
            cv::putText(renderTarget.outerMat(), _text, pos, _fontFace, _fontScale, _color, _fontThickness, cv::LINE_AA);
        }
    };

    struct Legend :public CvPlot::Drawable {
        CvPlot::Axes* _parentAxes;
        int _width = 200;
        int _height = 70;
        int _margin = 20;
        void render(CvPlot::RenderTarget& renderTarget) override {
            std::vector<CvPlot::Series*> seriesVec;
            for (const auto& drawable : _parentAxes->drawables()) {
                auto series = dynamic_cast<CvPlot::Series*>(drawable.get());
                if (series) {
                    seriesVec.push_back(series);
                }
            }
            CvPlot::Axes axes;
            axes.setMargins(5, _width - 2 * _margin - 60, 5, 5)
                .setXLim({ -.2,1.2 })
                .setYLim({ -.2,seriesVec.size() - 1 + .2 })
                .setYReverse();
            for (size_t i = 0; i < seriesVec.size(); i++) {
                auto& series = *seriesVec[i];
                axes.create<CvPlot::Series>(std::vector<cv::Point>{ {0,(int)i},{1,(int)i} })
                    .setLineType(series.getLineType())
                    .setLineWidth(series.getLineWidth())
                    .setColor(series.getColor())
                    .setMarkerType(series.getMarkerType())
                    .setMarkerSize(series.getMarkerSize());
                auto& label = axes.create<LegendLabel>();
                label._position = { 1,(int)i };
                label._text = series.getName();
            }
            cv::Rect rect(renderTarget.innerMat().cols - _width - _margin, _margin, _width, _height);
            if (rect.x >= 0 && rect.x + rect.width < renderTarget.innerMat().cols && rect.y >= 0 && rect.y + rect.height < renderTarget.innerMat().rows) {
                axes.render(renderTarget.innerMat()(rect));
                cv::rectangle(renderTarget.innerMat(), rect, cv::Scalar::all(0));
            }
        }
    };

    std::vector<double> step(log.size()), limit_time(step.size()),
                        time_total(step.size()), time_set_qp(step.size()), time_solve_qp(step.size()), time_get_alpha(step.size());
    for (size_t i = 0; i < step.size(); i++) 
    {
        step[i] = i;
        limit_time[i] = jsonConfig["Ts"];
        time_total[i] = log[i].compute_time.total;
        time_set_qp[i] = log[i].compute_time.set_qp;
        time_solve_qp[i] = log[i].compute_time.solve_qp;
        time_get_alpha[i] = log[i].compute_time.get_alpha;
    }
    auto axes = CvPlot::makePlotAxes();
    axes.create<CvPlot::Series>(step, limit_time, "-k").setName("Ts");
    axes.create<CvPlot::Series>(step, time_total, "-r").setName("Total");
    axes.create<CvPlot::Series>(step, time_set_qp, "-b").setName("Set QP");
    axes.create<CvPlot::Series>(step, time_solve_qp, "-g").setName("Solve QP");
    axes.create<CvPlot::Series>(step, time_get_alpha, "-c").setName("Linesearch");

    axes.create<Legend>()._parentAxes = &axes;
    axes.setXLim({0.,double(step.size())});
    axes.setYLim({0.,2.*double(jsonConfig["Ts"])});
    CvPlot::show("mywindow", axes);
    #endif

    return 0;
}


