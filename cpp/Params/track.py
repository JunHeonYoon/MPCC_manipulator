from math import pi
import numpy as np
from scipy.spatial.transform import Rotation as R

r = 0.1
t = np.linspace(pi/2, 2*pi+pi/2,10)

# For Self  & Singularity
# x = 2.5*r * np.sin(t)
# y = 2.0*r * np.sin(2 * t)
# z = 2.3*r * np.cos(t)

# rot = R.from_matrix([[1, 0, 0],
#                      [0, 1, 0],
#                      [0, 0, 1]])
# quat = rot.as_quat()
# quat_list = np.tile(quat, (x.size, 1))

# For Self collision & Env collision
# x = 2.2*r * np.sin(t)
# y = 2.6*r * np.sin(2 * t)
# z = 0*r * np.cos(t)

# rot = R.from_matrix([[1, 0, 0],
#                      [0, 1, 0],
#                      [0, 0, 1]])
# quat = rot.as_quat()
# quat_list = np.tile(quat, (x.size, 1))

# For Non-constant Orientation path 
# x = 1.0*r * np.sin(t)
# y = 1.0*r * np.sin(2 * t)
# z = 0*r * np.cos(t)

# rot1 = R.from_matrix([[1, 0, 0],
#                      [0, 1, 0],
#                      [0, 0, 1]])
# rot2 = R.from_matrix([[0,  -1, 0],
#                      [1, 0, 0],
#                      [0,  0, 1]])
# quat1 = rot1.as_quat()
# quat2 = rot2.as_quat()
# quat1_list = np.tile(quat1, (int(x.size/2), 1))
# quat2_list = np.tile(quat2, (x.size-quat1_list.shape[0], 1))
# quat_list = np.concatenate([quat1_list, quat2_list], axis=0)

# x = np.array([0, 0])
# y = np.array([0, 0])
# z = np.array([0, 0])
# rot1 = R.from_matrix([[1, 0, 0],
#                      [0, 1, 0],
#                      [0, 0, 1]])
# rot2 = R.from_matrix([[0,  -1, 0],
#                      [1, 0, 0],
#                      [0,  0, 1]])
# quat1 = rot1.as_quat()
# quat2 = rot2.as_quat()
# quat1_list = np.tile(quat1, (int(x.size/2), 1))
# quat2_list = np.tile(quat2, (x.size-quat1_list.shape[0], 1))
# quat_list = np.concatenate([quat1_list, quat2_list], axis=0)

x = [0, -0.2, -0.4, 0]
y = [0, 0, 0, 0]
z = [0, -0.35, -0.35,0]

rot1 = R.from_matrix([[1,  0,  0],
                     [0, -1, 0],
                     [0,  0, -1]])

rot2 = R.from_matrix([[0,  0,  -1],
                     [0, -1, 0],
                     [-1,  0, 0]])
quat1 = rot1.as_quat().reshape(1,-1)
quat2 = rot2.as_quat().reshape(1,-1)
quat_list = np.concatenate([quat1, quat2, quat2, quat1], axis=0)


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
