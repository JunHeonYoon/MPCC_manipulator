import sys
sys.path.append('../cpp/build')
import numpy as np
from scipy.spatial.transform import Rotation

from MPC_WRAPPER import getSkewMatrix as getSkewMatrix_CPP
from MPC_WRAPPER import getInverseSkewVector as getInverseSkewVector_CPP
from MPC_WRAPPER import LogMatrix as LogMatrix_CPP
from MPC_WRAPPER import ExpMatrix as ExpMatrix_CPP

def getSkewMatrix(input_vec: np.array) -> np.array:
    assert input_vec.size == 3, f"Input vector size {input_vec.size} does not match expected size {3}"
    return getSkewMatrix_CPP(input_vec)

def getInverseSkewVector(input_mat: np.array) -> np.array:
    assert input_mat.shape == (3,3), f"Input matrix shape {input_mat.shape} does not match expected size {(3,3)}"
    return getInverseSkewVector_CPP(input_mat)

def LogMatrix(input_Rot: np.array) -> np.array:
    assert input_Rot.shape == (3,3), f"Input rotation matrix shape {input_Rot.shape} does not match expected size {(3,3)}"
    return LogMatrix_CPP(input_Rot)

def ExpMatrix(input_skew: np.array) -> np.array:
    assert input_skew.shape == (3,3), f"Input skew matrix shape {input_skew.shape} does not match expected size {(3,3)}"
    return ExpMatrix_CPP(ExpMatrix)

def Log(input_Rot: np.array) -> np.array:
    assert input_Rot.shape == (3,3), f"Input rotation matrix shape {input_Rot.shape} does not match expected size {(3,3)}"
    return getInverseSkewVector(LogMatrix(input_Rot))

def Exp(input_vec: np.array) -> np.array:
    assert input_vec.size == 3, f"Input vector size {input_vec.size} does not match expected size {3}"
    return ExpMatrix(getSkewMatrix(input_vec))

def RotToQuat(rotation_matrix:np.array) -> np.array:
    assert rotation_matrix.shape == (3,3), f"Input rotation matrix shape {rotation_matrix.shape} does not match expected size {(3,3)}"
    rot = Rotation.from_matrix(rotation_matrix)
    return rot.as_quat().reshape(-1)

def QuatToRot(quaternion: np.array) -> np.array:
    assert quaternion.shape == (4,), f"Input quaternion shape {quaternion.shape} does not match expected size {(4,)}"
    rot = Rotation.from_quat(quaternion)
    return rot.as_matrix()