import sys
sys.path.append('../cpp/build')
import numpy as np
import json
import os

import MPCC_WRAPPER as MPCC_CPP
from .robot_model import RobotModel

class MPCC():
    def __init__(self) -> None:
        config_path = os.path.join(MPCC_CPP.pkg_path, "Params/config.json")
        with open(config_path, 'r') as iConfig:
            self.jsonConfig = json.load(iConfig)
        self.json_paths = MPCC_CPP.PathToJson()
        self.json_paths.param_path = os.path.join(MPCC_CPP.pkg_path, self.jsonConfig["model_path"])
        self.json_paths.cost_path = os.path.join(MPCC_CPP.pkg_path, self.jsonConfig["cost_path"])
        self.json_paths.bounds_path = os.path.join(MPCC_CPP.pkg_path, self.jsonConfig["bounds_path"])
        self.json_paths.track_path = os.path.join(MPCC_CPP.pkg_path, self.jsonConfig["track_path"])
        self.json_paths.normalization_path = os.path.join(MPCC_CPP.pkg_path, self.jsonConfig["normalization_path"])
        self.json_paths.sqp_path = os.path.join(MPCC_CPP.pkg_path, self.jsonConfig["sqp_path"])

        self.Ts = self.jsonConfig["Ts"]
        self.pred_horizon = MPCC_CPP.N

        self.robot_model = RobotModel()

        self.mpc = MPCC_CPP.MPC(self.Ts, self.json_paths)
        self.track_set = False
    
    def setTrack(self, state:np.array) -> None:
        assert state.size == MPCC_CPP.NX, f"State size {state.size} does not match expected size {MPCC_CPP.NX}"
        self.init_state = state

        ee_pos = self.robot_model.getEEPosition(self.init_state[:self.robot_model.num_q])
        track = MPCC_CPP.Track(self.json_paths.track_path)
        track_xyzr = track.getTrack(ee_pos)

        self.mpc.setTrack(track_xyzr.X, 
                          track_xyzr.Y,
                          track_xyzr.Z,
                          track_xyzr.R)
        
        self.spline_track = MPCC_CPP.ArcLengthSpline()
        self.spline_track.gen6DSpline(track_xyzr.X, 
                                      track_xyzr.Y,
                                      track_xyzr.Z,
                                      track_xyzr.R)
        self.spline_path = self.spline_track.getPathData()

        self.track_set = True
    
    def getSplinePath(self) -> {np.array, np.array, np.array}:
        assert self.track_set == True, "Set Track first!"
        position = np.stack([self.spline_path.X, self.spline_path.Y, self.spline_path.Z], axis=1)
        rotation = np.array(self.spline_path.R)
        arc_length = self.spline_path.s

        return position, rotation, arc_length
    
    def getRefPose(self, path_parameter:float) -> {np.array, np.array}:
        assert path_parameter >= np.min(self.spline_path.s) and path_parameter <= np.max(self.spline_path.s), f"Path parameter must be in [{np.min(self.spline_path.s), np.max(self.spline_path.s)}] and your input is {path_parameter}"
        return self.spline_track.getPosition(path_parameter), self.spline_track.getOrientation(path_parameter)


    def runMPC(self, state:np.array) -> {bool, np.array, np.array, list, dict}:
        assert self.track_set == True, "Set Track first!"
        assert state.size == MPCC_CPP.NX, f"State size {state.size} does not match expected size {MPCC_CPP.NX}"
        x0 = MPCC_CPP.vectorToState(state)
        mpc_sol = MPCC_CPP.zeroReturn()
        mpc_status = self.mpc.runMPC(mpc_sol, x0)
        updated_state = MPCC_CPP.stateToVector(x0)

        mpc_horizon=[]
        for mpc_horizon_raw in mpc_sol.mpc_horizon:
            state_k = MPCC_CPP.stateToVector(mpc_horizon_raw.xk)
            input_k = MPCC_CPP.inputToVector(mpc_horizon_raw.uk)
            mpc_horizon.append({"state": state_k, "input": input_k})
        
        compute_time = {"total": mpc_sol.compute_time.total,
                        "set_qp": mpc_sol.compute_time.set_qp,
                        "solve_qp": mpc_sol.compute_time.solve_qp,
                        "get_alpha": mpc_sol.compute_time.get_alpha}
        
        return mpc_status, updated_state, MPCC_CPP.inputToVector(mpc_sol.u0), mpc_horizon, compute_time
