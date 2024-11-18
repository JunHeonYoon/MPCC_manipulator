import sys
sys.path.append('../cpp/build')
import numpy as np

from MPCC_WRAPPER import EnvCollNNmodel as EnvCollNNmodel_CPP

class EnvCollisionNN():
    def __init__(self, model_path:str = None):
        if(model_path):
            self.NNmodel = EnvCollNNmodel_CPP(model_path)
        else:
            self.NNmodel = EnvCollNNmodel_CPP()
        pass

    def setNeuralNetwork(self, input_size:int, output_size:int, hidden_layer_size:np.array, is_nerf:bool):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.NNmodel.setNeuralNetwork(input_size, output_size, hidden_layer_size, is_nerf)

    def calculateMlpOutput(self, input:np.array, time_verbose:bool = False) -> {np.array, np.array}:
        assert input.size == self.input_size, f"Input size {input.size} does not match expected size {self.input_size}"
        mlp_output = self.NNmodel.calculateMlpOutput(input, time_verbose)
        return mlp_output[0], mlp_output[1][:9, :7] # output, jacobian of output
