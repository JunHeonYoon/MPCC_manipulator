from math import pi
import numpy as np
from scipy.spatial.transform import Rotation as R

# For self collision experiment
x = [0, -0.20, -0.40, 0]
y = [0, 0, 0, 0]
z = [0, -0.51, -0.4,0]

rot1 = R.from_matrix([[1,  0,  0],
                     [0, -1, 0],
                     [0,  0, -1]])

rot2 = R.from_matrix([[0,  0,  -1],
                     [0, -1, 0],
                     [-1,  0, 0]])
quat1 = rot1.as_quat().reshape(1,-1)
quat2 = rot2.as_quat().reshape(1,-1)
quat_list = np.concatenate([quat1, quat2, quat2, quat1], axis=0)


# For singularity experiment
# r = 0.1
# t = np.linspace(0, 2*pi, 100)

# x = 2.4*r * np.sin(t)
# y = 1.5*r * np.sin(2 * t)
# # z = 2.15*r * np.cos(t)
# z = 0*r * np.cos(t)

# rot = R.from_matrix([[1,  0,  0],
#                      [0, -1, 0],
#                      [0,  0, -1]])
# quat = rot.as_quat()
# quat_list = np.tile(quat, (x.size, 1))

total_desc = f""" 
{{
    "X": {list(map(np.double, x))},
    "Y": {list(map(np.double, y))},
    "Z": {list(map(np.double, z))},
    "quat_X": {list(map(np.double, quat_list[:,0]))},
    "quat_Y": {list(map(np.double, quat_list[:,1]))},
    "quat_Z": {list(map(np.double, quat_list[:,2]))},
    "quat_W": {list(map(np.double, quat_list[:,3]))}
}} 
"""

open('track.json','w').write(total_desc)
