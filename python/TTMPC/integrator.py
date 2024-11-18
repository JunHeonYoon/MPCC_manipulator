import sys
sys.path.append('../cpp/build')
import json
import os
import numpy as np

from MPC_WRAPPER import Integrator as Integrator_CPP
from MPC_WRAPPER import NX, NU, vectorToState, vectorToInput, stateToVector
from MPC_WRAPPER import pkg_path, PathToJson

class Integrator():
    def __init__(self) -> None:
        config_path = os.path.join(pkg_path, "Params/config.json")
        with open(config_path, 'r') as iConfig:
            self.jsonConfig = json.load(iConfig)
        json_paths = PathToJson()
        json_paths.param_path = os.path.join(pkg_path, self.jsonConfig["model_path"])
        json_paths.cost_path = os.path.join(pkg_path, self.jsonConfig["cost_path"])
        json_paths.bounds_path = os.path.join(pkg_path, self.jsonConfig["bounds_path"])
        json_paths.track_path = os.path.join(pkg_path, self.jsonConfig["track_path"])
        json_paths.normalization_path = os.path.join(pkg_path, self.jsonConfig["normalization_path"])
        json_paths.sqp_path = os.path.join(pkg_path, self.jsonConfig["sqp_path"])

        self.integrator = Integrator_CPP(self.jsonConfig["Ts"], json_paths)

    def simTimeStep(self, state:np.array, input:np.array, time_step:float = None)->np.array:
        assert state.size == NX, f"State size {state.size} does not match expected size {NX}"
        assert input.size == NU, f"State size {input.size} does not match expected size {NU}"
        
        x0 = vectorToState(state)
        u0 = vectorToInput(input)

        if(not time_step):
            time_step = self.jsonConfig["Ts"]

        x1 = self.integrator.simTimeStep(x0, u0, time_step)

        return stateToVector(x1)