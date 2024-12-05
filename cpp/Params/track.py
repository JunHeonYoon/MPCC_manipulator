from math import pi
import numpy as np
from scipy.spatial.transform import Rotation as R

r = 0.1
t = np.linspace(pi/2, 2*pi+pi/2, 100)

# For Self  & Singularity
# x = 2.5*r * np.sin(t)
# y = 2.0*r * np.sin(2 * t)
# z = 2.3*r * np.cos(t)

# For Self collision & Env collision
x = 2.2*r * np.sin(t)
y = 2.6*r * np.sin(2 * t)
z = 0*r * np.cos(t)

rot = R.from_matrix([[1,  0,  0],
                     [0, -1, 0],
                     [0,  0, -1]])
quat = rot.as_quat()
quat_list = np.tile(quat, (x.size, 1))

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
