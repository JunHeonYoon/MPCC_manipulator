import TTMPC
import numpy as np
from math import pi
np.set_printoptions(suppress=True, precision=3)

integrator = TTMPC.Integrator()
robot = TTMPC.RobotModel()
robot_dof = robot.num_q
selcolNN = TTMPC.SelfCollisionNN()
selcolNN.setNeuralNetwork(input_size=robot_dof, output_size=1, hidden_layer_size=np.array([256, 64]), is_nerf=True)

TTMPC = TTMPC.MPCC()

state = np.array([0., 0., 0., -pi/2, 0., pi/2, pi/4, 0., 0.])
input = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
TTMPC.setTrack(state)
spline_pos, spline_ori, spline_arc_length = TTMPC.getSplinePath()

debug_data = {}
debug_data["q"] = []
debug_data["qdot"] = []
debug_data["min_dist"] = []
debug_data["mani"] = []
debug_data["pred_ee_pose"] = [] # x, y, z, q_x, q_y, q_z, q_w
debug_data["ref_ee_pose"] = []  # x, y, z, q_x, q_y, q_z, q_w

time_data = {}
time_data["total"] = []
time_data["set_qp"] = []
time_data["solve_qp"] = []
time_data["get_alpha"] = []

param_value = {"cost": {"qOri": 0.5}}

for time_idx in range(10000):
    # if(time_idx == 100):
    #     mpc.setParam(param_value)

    status, state, input, mpc_horizon, compute_time = TTMPC.runMPC(state, input)
    if status == False:
        print("MPC did not solve properly!!")
        break
    state = integrator.simTimeStep(state, input)

    ee_pos = robot.getEEPosition(state[:robot_dof])
    ee_ori = robot.getEEOrientation(state[:robot_dof])
    ee_vel = robot.getEEJacobianv(state[:robot_dof]) @ input[:robot_dof]
    min_dist, _ = selcolNN.calculateMlpOutput(state[:robot_dof])
    mani = robot.getEEManipulability(state[:robot_dof])

    print("===============================================================")
    print("time step   : ",time_idx)
    print("state       : ", state)
    print("q           : ", state[:robot_dof])
    print("qdot        : ", input[:robot_dof])
    print("x           : ", ee_pos)
    print("xdot        : {:0.5f}".format(np.linalg.norm(ee_vel)))
    print("xdot        : ", ee_vel)
    print("R           :\n", ee_ori)
    print("mani        : {:0.5f}".format(mani))
    print("min dist[cm]: {:0.5f}".format(min_dist[0]))
    print("s           : {:0.6f}".format(state[-2]))
    print("vs          : {:0.6f}".format(state[-1]))
    print("dVs         : {:0.5f}".format(input[-1]))
    print("MPC time    : {:0.5f}".format(compute_time["total"]))
    print("===============================================================")

    debug_data["q"].append(state[:robot_dof]) 
    debug_data["qdot"].append(input[:robot_dof])
    debug_data["min_dist"].append(min_dist[0])
    debug_data["mani"].append(mani)

    pred_ee_pose = np.zeros([TTMPC.pred_horizon + 1, 7])
    ref_ee_pose  = np.zeros([TTMPC.pred_horizon + 1, 7])
    for i in range(TTMPC.pred_horizon + 1):
        pred_ee_pose[i,:3] = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
        pred_ee_pose[i,3:] = TTMPC.RotToQuat(robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof]))
        ref_pos, ref_ori = TTMPC.getRefPose(mpc_horizon[i]["state"][-2])
        ref_ee_pose[i,:3] = ref_pos
        ref_ee_pose[i,3:] = TTMPC.RotToQuat(ref_ori)

    debug_data["pred_ee_pose"].append(pred_ee_pose)
    debug_data["ref_ee_pose"].append(ref_ee_pose)

    time_data["total"].append(compute_time["total"])
    time_data["set_qp"].append(compute_time["set_qp"])
    time_data["solve_qp"].append(compute_time["solve_qp"])
    time_data["get_alpha"].append(compute_time["get_alpha"])

    if np.linalg.norm((spline_pos[-1] - ee_pos), 2) < 1E-2 and np.linalg.norm(TTMPC.Log(spline_ori[-1].T @ ee_ori), 2) and abs(state[-2] - spline_arc_length[-1]) < 1E-2:
        print("End point reached!!!")
        break

with open('splined_path.txt', 'w') as splined_path_file:
    for pos, ori in zip(spline_pos, spline_ori):
        quaternion = TTMPC.RotToQuat(ori)
        data_to_write = np.concatenate([pos, quaternion], axis=0)
        splined_path_file.write(" ".join(map(str, data_to_write)) + "\n")
print("Data written to splined_path.txt")


with open('debug.txt', 'w') as debug_file:
    for q, qdot, min_dist, mani, pred_ee_pose, ref_ee_pose in zip(debug_data["q"], debug_data["qdot"], debug_data['min_dist'], debug_data['mani'], debug_data['pred_ee_pose'], debug_data['ref_ee_pose']):
        data_to_write = np.concatenate([q, qdot, np.array([min_dist, mani]), pred_ee_pose.flatten(), ref_ee_pose.flatten()], axis=0)
        debug_file.write(" ".join(map(str, data_to_write)) + "\n")
print("Data written to debug.txt")

time_data["total"] = np.array(time_data["total"])
time_data["set_qp"] = np.array(time_data["set_qp"])
time_data["solve_qp"] = np.array(time_data["solve_qp"])
time_data["get_alpha"] = np.array(time_data["get_alpha"])

print("mean nmpc time[sec]: {:0.6f} ".format(np.mean(time_data["total"])))
print("max nmpc time[sec]: {:0.6f} ".format(np.max(time_data["total"])))

import matplotlib.pyplot as plt

# Plotting the computation times
plt.figure(figsize=(14, 8))

plt.plot(time_data["total"], label="Total Time", color='b')
plt.plot(time_data["set_qp"], label="Set QP Time", color='g')
plt.plot(time_data["solve_qp"], label="Solve QP Time", color='r')
plt.plot(time_data["get_alpha"], label="Get Alpha Time", color='c')
plt.axhline(y=TTMPC.Ts, color='black', linestyle='--', label="Ts")

plt.xlabel("Time Step")
plt.ylabel("Time (s)")
plt.title("Computation Times per Time Step")
plt.ylim(-0.01, 0.05)  
plt.xlim(0, len(time_data["total"]))
plt.legend()
plt.grid(True)
plt.show()
