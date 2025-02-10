import sys
sys.path.append('../cpp/build')
import numpy as np

from MPCC_WRAPPER import RobotModel as RobotModel_CPP

class RobotModel():
    def __init__(self):
        self.robot = RobotModel_CPP()
        self.num_q = self.robot.getNumq()

    def getEEJacobian(self, joint_angle: np.array) -> np.array:
        assert joint_angle.size == self.num_q, f"Joint angle size {joint_angle.size} does not match expected size {self.num_q}"
        return self.robot.getJacobian(joint_angle)
    
    def getEEJacobianv(self, joint_angle: np.array) -> np.array:
        assert joint_angle.size == self.num_q, f"Joint angle size {joint_angle.size} does not match expected size {self.num_q}"
        return self.robot.getJacobianv(joint_angle)
    
    def getEEJacobianw(self, joint_angle: np.array) -> np.array:
        assert joint_angle.size == self.num_q, f"Joint angle size {joint_angle.size} does not match expected size {self.num_q}"
        return self.robot.getJacobianw(joint_angle)
    
    def getEEPosition(self, joint_angle: np.array) -> np.array:
        assert joint_angle.size == self.num_q, f"Joint angle size {joint_angle.size} does not match expected size {self.num_q}"
        return self.robot.getEEPosition(joint_angle)
    
    def getEEOrientation(self, joint_angle: np.array) -> np.array:
        assert joint_angle.size == self.num_q, f"Joint angle size {joint_angle.size} does not match expected size {self.num_q}"
        return self.robot.getEEOrientation(joint_angle)
    
    def getEETransformation(self, joint_angle: np.array) -> np.array:
        assert joint_angle.size == self.num_q, f"Joint angle size {joint_angle.size} does not match expected size {self.num_q}"
        return self.robot.getEETransformation(joint_angle)
    
    def getEEManipulability(self, joint_angle: np.array) -> np.array:
        assert joint_angle.size == self.num_q, f"Joint angle size {joint_angle.size} does not match expected size {self.num_q}"
        return self.robot.getManipulability(joint_angle)
    