import sys
sys.path.append('../cpp/build')
import numpy as np
import json
import os

import MPC_WRAPPER as MPC_CPP
from .robot_model import RobotModel

class MPC():
    def __init__(self) -> None:
        config_path = os.path.join(MPC_CPP.pkg_path, "Params/config.json")
        with open(config_path, 'r') as iConfig:
            self.jsonConfig = json.load(iConfig)
        self.json_paths = MPC_CPP.PathToJson()
        self.json_paths.param_path = os.path.join(MPC_CPP.pkg_path, self.jsonConfig["model_path"])
        self.json_paths.cost_path = os.path.join(MPC_CPP.pkg_path, self.jsonConfig["cost_path"])
        self.json_paths.bounds_path = os.path.join(MPC_CPP.pkg_path, self.jsonConfig["bounds_path"])
        self.json_paths.track_path = os.path.join(MPC_CPP.pkg_path, self.jsonConfig["track_path"])
        self.json_paths.normalization_path = os.path.join(MPC_CPP.pkg_path, self.jsonConfig["normalization_path"])
        self.json_paths.sqp_path = os.path.join(MPC_CPP.pkg_path, self.jsonConfig["sqp_path"])

        self.Ts = self.jsonConfig["Ts"]
        self.pred_horizon = MPC_CPP.N

        self.robot_model = RobotModel()

        self.mpc = MPC_CPP.MPC(self.Ts, self.json_paths)
        self.track_set = False
    
    def setParam(self, param_value:dict) -> None:
        param_list = ["param", "cost", "bounds", "normalization", "sqp"]
        assert set(param_value.keys()).issubset(param_list) == True, f"List of Parameters must be a subset of {param_list}, but got {list(param_value.keys())}"

        param_dict = {
            "param": ["max_dist_proj","desired_ee_velocity","tol_sing", "tol_selcol", "tol_envcol"],
            "cost": ["qE","qENmult","qOri","rdq"],
            "bounds": ["q1l","q2l","q3l","q4l","q5l","q6l","q7l","q1u","q2u","q3u","q4u","q5u","q6u","q7u","dq1l","dq2l","dq3l","dq4l","dq5l","dq6l","dq7l","dq1u","dq2u","dq3u","dq4u","dq5u","dq6u","dq7u"],
            "normalization": ["q1","q2","q3","q4","q5","q6","q7","dq1","dq2","dq3","dq4","dq5","dq6","dq7"],
            "sqp": ["eps_prim","eps_dual","line_search_tau","line_search_eta","line_search_rho","max_iter","line_search_max_iter","do_SOC","use_BFGS"]
        }

        param_value_cpp = MPC_CPP.ParamValue()
        # print(dir(param_value_cpp))

        for key, value in param_value.items():
            valid_keys = param_dict.get(key, [])
            assert set(value.keys()).issubset(valid_keys), f"Keys for {key} must be a subset of {valid_keys}, but got {list(value.keys())}"

            for sub_key, sub_value in value.items():
                getattr(param_value_cpp, key)[sub_key] = sub_value

        self.mpc.setParam(param_value_cpp)
        self.spline_track = self.mpc.getTrack()
        self.ref_traj = self.spline_track.getTrajectory()
        self.track_set = True


    def setTrack(self, state:np.array) -> None:
        assert state.size == MPC_CPP.NX, f"State size {state.size} does not match expected size {MPC_CPP.NX}"
        self.init_state = state

        ee_pos = self.robot_model.getEEPosition(self.init_state[:self.robot_model.num_q])
        track = MPC_CPP.Track(self.json_paths.track_path)
        track_xyzr = track.getTrack(ee_pos)

        self.mpc.setTrack(track_xyzr.X, 
                          track_xyzr.Y,
                          track_xyzr.Z,
                          track_xyzr.R)
        
        self.spline_track = self.mpc.getTrack()
        self.ref_traj = self.spline_track.getTrajectory()

        self.track_set = True
    
    def getTotalTrajectory(self) -> {np.array, np.array}:
        assert self.track_set == True, "Set Track first!"
        return np.array(self.ref_traj.P), np.array(self.ref_traj.R)
    
    def getNTrajectory(self, time_idx:int) -> {np.array, np.array}:
        assert self.track_set == True, "Set Track first!"
        ref_n_traj = self.spline_track.getNTrajectory(time_idx)
        return np.array(ref_n_traj.P),np.array(ref_n_traj.R)
    
    def getContourError(self, s_guess:float, ee_posi:np.array)-> {float, float}:
        s_opt = self.spline_track.projectOnSpline(s_guess, ee_posi)
        ref_posi = self.spline_track.getPosition(s_opt)
        return s_opt, np.linalg.norm(ref_posi - ee_posi)


    def runMPC(self, state:np.array, input:np.array, time_idx:int, voxel:np.array = np.zeros(int(36*36*36))) -> {bool, np.array, np.array, list, dict}:
        assert self.track_set == True, "Set Track first!"
        assert state.size == MPC_CPP.NX, f"State size {state.size} does not match expected size {MPC_CPP.NX}"
        x0 = MPC_CPP.vectorToState(state)
        u0 = MPC_CPP.vectorToInput(input)
        mpc_sol = MPC_CPP.MPCReturn()
        mpc_status = self.mpc.runMPC_(mpc_sol, x0, u0, voxel, time_idx)
        updated_state = MPC_CPP.stateToVector(x0)

        mpc_horizon=[]
        for mpc_horizon_raw in mpc_sol.mpc_horizon:
            state_k = MPC_CPP.stateToVector(mpc_horizon_raw.xk)
            input_k = MPC_CPP.inputToVector(mpc_horizon_raw.uk)
            mpc_horizon.append({"state": state_k, "input": input_k})
        
        compute_time = {"total": mpc_sol.compute_time.total,
                        "set_qp": mpc_sol.compute_time.set_qp,
                        "solve_qp": mpc_sol.compute_time.solve_qp,
                        "get_alpha": mpc_sol.compute_time.get_alpha,
                        "set_env": mpc_sol.compute_time.set_env}
        
        return mpc_status, updated_state, MPC_CPP.inputToVector(mpc_sol.u0), mpc_horizon, compute_time
