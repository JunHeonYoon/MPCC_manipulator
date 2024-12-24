import MPCC
import numpy as np
from numpy.linalg import inv
from math import pi

import matplotlib.pyplot as plt
from srmt.planning_scene import PlanningScene
import argparse
import scipy.io

## ROS library
import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Float32

from main_utils import create_path_message2, create_pred_path_message, plt_func

np.set_printoptions(suppress=True, precision=3)

## MPCC parameters
param_value = {'cost': {
                    "qC" : 500.0,
                    "qCNmult": 5,
                    "qL" : 100.0,
                    "qVs" : 20.0,

                    "qOri": 50,

                    "rdq"  : 0.002,
                    "rddq"  : 10,
                    "rdVs" : 0.1,
                    "rVee" : 0,
                    },
            'param': {
                    "desired_ee_velocity": 0.1,
                    "tol_sing": 0.018,
                    "tol_selcol": 1.0,
                    "tol_envcol": 1.0,
                    }
            }

## Obstacle information
obs_position = np.array([0.48,  0.218, 0.521]) # unit: [m]
obs_limit = np.array([[0.48,  0.218, 0.421],   # lower limit
                      [0.48,  0.218, 0.621]])  # upper limit
obs_radius = 5         # unit: [cm]
obs_speed = 0.05       # unit: [m/s]

def main(args):
    ## Create Planning Scene
    pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="world")

    ## ros publisher
    splined_path_pub = rospy.Publisher('/mpcc/splined_path', Path, queue_size=10)
    local_path_pub = rospy.Publisher('/mpcc/local_path', Path, queue_size=10)
    ref_local_path_pub = rospy.Publisher('/mpcc/ref_local_path', Path, queue_size=10)
    ee_speed_pub = rospy.Publisher('/mpcc/ee_speed', Float32, queue_size=1)
    mani_pub = rospy.Publisher('/mpcc/mani', Float32, queue_size=1)
    sel_min_dist_pub = rospy.Publisher('/mpcc/sel_min_dist', Float32, queue_size=1)
    env_min_dist_pub = rospy.Publisher('/mpcc/env_min_dist', Float32, queue_size=1)
    contour_error_pub = rospy.Publisher('/mpcc/contour_error', Float32, queue_size=1)

    ## Create mpc controller
    mpc = MPCC.MPCC()
    N = mpc.pred_horizon
    panda_num_links = mpc.num_links
    robot_dof = mpc.robot_dof

    ## Create robot data processor
    integrator = MPCC.Integrator()
    robot = MPCC.RobotModel()
    selcolNN = MPCC.SelfCollisionNN()
    selcolNN.setNeuralNetwork(input_size=robot_dof, output_size=1, hidden_layer_size=np.array([256, 64]), is_nerf=True)
    envcolNN = MPCC.EnvCollisionNN()
    envcolNN.setNeuralNetwork(input_size=robot_dof+3, output_size=panda_num_links, hidden_layer_size=np.array([256, 256, 256, 256]), is_nerf=True)

    ## Set initial state and control input
    state = np.array([0., 0., 0., -pi/2, 0., pi/2, pi/4, 0., 0.])
    input = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    mpc.setTrack(state)
    mpc.setParam(param_value)
    spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()
    spline_T = np.zeros([spline_pos.shape[0], 4, 4])
    for i, (posi, rot) in enumerate(zip(spline_pos, spline_ori)):
        spline_T[i, :3, :3] = rot
        spline_T[i, :3, 3] = posi


    splined_path_msg = create_path_message2(spline_pos, spline_ori)

    ## Debugging data  
    debug_data = {}
    debug_data["q"] = []                    # joint angle
    debug_data["qdot"] = []                 # joint velocity
    debug_data["qddot"] = []                # joint acceleration
    debug_data["s"] = []                    # path parameter
    debug_data["vs"] = []                   # velocity of path parameter
    debug_data["dVs"] = []                  # acceleration of path parameter
    debug_data["ee_vel"] = []               # End-Effector velocity
    debug_data["ee_speed"] = []             # End-Effector speed
    debug_data["sel_min_dist"] = []         # Minium distance for self collision
    debug_data["env_min_dist"] = []         # Minium distance for environment collision
    debug_data["mani"] = []                 # Manipulability
    debug_data["contour_error"] = []        # Contouring error
    debug_data["spline_ee_pose"] = spline_T # splined track path pose (track length, 4, 4)
    debug_data["pred_ee_pose"] = []         # predicted End-Effector path pose (N+1, 4, 4)
    debug_data["ref_ee_pose"] = []          # predicted reference path pose (N+1, 4, 4)

    time_data = {}
    time_data["total"] = []
    time_data["set_env"] = []
    time_data["set_qp"] = []
    time_data["solve_qp"] = []
    time_data["get_alpha"] = []

    SLECOL_BUFFER = param_value["param"]["tol_selcol"]
    MANI_BUFFER = param_value["param"]["tol_sing"]

    if(args.plot):
        ## Animated plotter
        plt.ion()
        fig, (selcol_ax, mani_ax) = plt.subplots(2, 1, figsize=(12, 8))

        min_dist_true_line, = selcol_ax.plot([],[], label='ans', color="blue", linewidth=4.0, linestyle='--')
        min_dist_pred_line, = selcol_ax.plot([],[], label='pred', color = "red", linewidth=2.0)
        min_dist_pred_line, = selcol_ax.plot([],[], label='min dist', color = "red", linewidth=2.0)
        selcol_ax.legend()
        selcol_ax.set_ylim([SLECOL_BUFFER - 5, 25])
        selcol_ax.grid()

        ## Second subplot
        mani_line, = mani_ax.plot([], [], label='mani', color="red", linewidth=2.0)
        mani_ax.legend()
        mani_ax.set_ylim([MANI_BUFFER - 0.05, 0.2])
        mani_ax.grid()

        sim_time_data = np.zeros((1))
        min_dist_real_data = np.zeros((1))
        min_dist_pred_data = np.zeros((1))
        mani_data = np.zeros((1))

    obs_step = obs_speed * mpc.Ts
    
    time_idx=0
    rate = rospy.Rate(1./mpc.Ts)
    qdot_pre = np.zeros(robot_dof)

    while not rospy.is_shutdown():
        ##  Obstacle moovement
        if args.is_obs:
            if (obs_step > 0 and obs_position[2] >= obs_limit[1,2]) or (obs_step < 0 and obs_position[2] <= obs_limit[0,2]):
                obs_step = obs_step*-1
            obs_position[2] = obs_position[2] + obs_step
            pc.add_sphere("obs", 0.01*obs_radius, obs_position + np.array([0.3, 0, 0.256]), np.array([1,0,0,0]))

        ## run MPCC
        status, state, input, mpc_horizon, compute_time = mpc.runMPC(state, input, obs_position, obs_radius) if args.is_obs else mpc.runMPC(state, input)
        if status == False:
            print("MPC did not solve properly!!")
            break
        ## Virtual contact scenario
        # if time_idx > 500 and time_idx < 600:
        #     j = robot.getEEJacobian(state[:robot_dof])
        #     xdot = j @ input[:robot_dof]
        #     j_pinv = j.T.dot(inv(j.dot(j.T)))
        #     input[:robot_dof] = 0.5 * j_pinv @ xdot + (np.identity(robot_dof) - j_pinv.dot(j)) @ input[:robot_dof]
        state = integrator.simTimeStep(state, input)

        ## get robot information
        q = state[:robot_dof]
        qdot = input[:robot_dof]
        qddot = (qdot - qdot_pre)/mpc.Ts
        x = robot.getEEPosition(q)
        xdot = robot.getEEJacobianv(q) @ qdot
        x_speed = np.linalg.norm(xdot)
        rotation = robot.getEEOrientation(q)
        s = state[-2]
        vs = state[-1]
        dVs = input[-1]
        sel_min_dist, _ = selcolNN.calculateMlpOutput(q)
        real_sel_min_dist = pc.min_distance(q)*100
        env_min_dist, _ = envcolNN.calculateMlpOutput(np.concatenate((q, obs_position), axis=0))
        mani = robot.getEEManipulability(q)
        contour_error = mpc.getContourError(s, x)
        pred_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        ref_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        for i in range(mpc.pred_horizon + 1):
            pred_ee_T[i, :3, :3] = robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof])
            pred_ee_T[i, :3, 3]  = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
            ref_ee_T[i, :3, 3], ref_ee_T[i, :3, :3] = mpc.getRefPose(mpc_horizon[i]["state"][-2])

        ## visualize
        pc.display(q)

        ## Print robot information
        print("===============================================================")
        print("time step       : ",time_idx)
        print("q               : ", q)
        print("qdot            : ", qdot)
        print("qddot           : ", qddot)
        print("x               : ", x)
        print("xdot            : {:0.5f}".format(x_speed))
        print("xdot            : ", xdot)
        print("R               :\n", rotation)
        print("s               : {:0.6f}".format(s))
        print("vs              : {:0.6f}".format(vs))
        print("dVs             : {:0.5f}".format(input[-1]))
        print("mani            : {:0.5f}".format(mani))
        print("sel min dist[cm]: {:0.5f}".format(sel_min_dist))
        print("env min dist[cm]: ",env_min_dist)
        print("MPC time        : {:0.5f}".format(compute_time["total"]))
        print("===============================================================")


        if(args.plot):
            sim_time_data = np.append(sim_time_data, np.array([time_idx*mpc.Ts]), axis=0)
            min_dist_real_data = np.append(min_dist_real_data, np.array([real_sel_min_dist]), axis=0)
            min_dist_pred_data = np.append(min_dist_pred_data, np.array([sel_min_dist]), axis=0)
            mani_data = np.append(mani_data, np.array([mani]), axis=0)

            if sim_time_data.shape[0] > 10:
                sim_time_data = sim_time_data[-10:]
                min_dist_real_data = min_dist_real_data[-10:]
                min_dist_pred_data = min_dist_pred_data[-10:]
                mani_data = mani_data[-10:]

            plt_func(fig, selcol_ax, mani_ax, min_dist_true_line, min_dist_pred_line, mani_line, sim_time_data, min_dist_real_data, min_dist_pred_data, mani_data, SLECOL_BUFFER, MANI_BUFFER)

        ## Save data 
        debug_data["q"].append(q) 
        debug_data["qdot"].append(qdot)
        debug_data["qddot"].append(qdot)
        debug_data["s"].append(s)
        debug_data["vs"].append(vs)
        debug_data["dVs"].append(dVs)
        debug_data["ee_vel"].append(xdot)
        debug_data["ee_speed"].append(x_speed)
        debug_data["sel_min_dist"].append(sel_min_dist)
        debug_data["env_min_dist"].append(env_min_dist)
        debug_data["mani"].append(mani)
        debug_data["contour_error"].append(contour_error)
        debug_data["pred_ee_pose"].append(pred_ee_T)
        debug_data["ref_ee_pose"].append(ref_ee_T)

        time_data["total"].append(compute_time["total"])
        time_data["set_env"].append(compute_time["set_env"])
        time_data["set_qp"].append(compute_time["set_qp"])
        time_data["solve_qp"].append(compute_time["solve_qp"])
        time_data["get_alpha"].append(compute_time["get_alpha"])

        ## Publish data
        local_path_msg = create_pred_path_message(pred_ee_T[:, :3, 3], pred_ee_T[:, :3, :3])
        ref_local_path_msg = create_pred_path_message(ref_ee_T[:, :3, 3], ref_ee_T[:, :3, :3])
        ee_speed_msg = Float32()
        mani_msg = Float32()
        sel_min_dist_msg = Float32()
        env_min_dist_msg = Float32()
        contour_error_msg = Float32()
        ee_speed_msg.data = x_speed
        mani_msg.data = mani
        sel_min_dist_msg.data = sel_min_dist
        env_min_dist_msg.data = np.min(env_min_dist)
        contour_error_msg.data = contour_error*100 # [m] -> [cm]
        
        splined_path_pub.publish(splined_path_msg)
        local_path_pub.publish(local_path_msg)
        ref_local_path_pub.publish(ref_local_path_msg)
        ee_speed_pub.publish(ee_speed_msg)
        mani_pub.publish(mani_msg)
        sel_min_dist_pub.publish(sel_min_dist_msg)
        env_min_dist_pub.publish(env_min_dist_msg)
        contour_error_pub.publish(contour_error_msg)
        
        

        ## End condition 
        if np.linalg.norm((spline_pos[-1] - x), 2) < 1E-2 and np.linalg.norm(MPCC.Log(spline_ori[-1].T @ rotation), 2) < 1E-2 and abs(state[-2] - spline_arc_length[-1]) < 1E-2:
            print("End point reached!!!")
            break
        
        rate.sleep()
        qdot_pre = qdot
        time_idx += 1

    ## Convert lists to NumPy arrays
    for key in debug_data:
        debug_data[key] = np.array(debug_data[key])
    for key in time_data:
        time_data[key] = np.array(time_data[key])

    ## Save the data to a .mat file
    scipy.io.savemat("debug_data.mat", debug_data)
    print("Data written to debug.mat")

    scipy.io.savemat("time_data.mat", time_data)
    print("Data written to debug.mat")


    print("mean nmpc time[sec]: {:0.6f} ".format(np.mean(time_data["total"])))
    print("max nmpc time[sec]: {:0.6f} ".format(np.max(time_data["total"])))

    ## Plotting the computation times
    plt.figure(figsize=(14, 8))

    plt.plot(time_data["total"], label="Total Time", color='b')
    plt.plot(time_data["set_env"], label="Set Env Time", color='m')
    plt.plot(time_data["set_qp"], label="Set QP Time", color='g')
    plt.plot(time_data["solve_qp"], label="Solve QP Time", color='r')
    plt.plot(time_data["get_alpha"], label="Get Alpha Time", color='c')
    plt.axhline(y=mpc.Ts, color='black', linestyle='--', label="Ts")

    plt.xlabel("Time Step")
    plt.ylabel("Time (s)")
    plt.title("Computation Times per Time Step")
    plt.ylim(-0.01, 0.05)  
    plt.xlim(0, len(time_data["total"]))
    plt.legend()
    plt.grid(True)

    ## Plotting the s/vs, ee_speed
    fig=plt.figure(figsize=(14, 8))
    fig.subplots_adjust(hspace=1)
    plt.subplot(411)
    # plt.plot("s", "vs", data=debug_data, label="vs", color='b')
    plt.plot("s", "ee_speed", data=debug_data, label="ee_speed", color='r')
    plt.axhline(y=param_value["param"]["desired_ee_velocity"], color='black', linestyle='--', label="desired")
    plt.xlabel("s (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("EE Speed per Arc length")
    plt.ylim(-0.01, max(max(debug_data["ee_speed"]), max(debug_data["vs"]), param_value["param"]["desired_ee_velocity"])*1.2)  
    plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    ## Plotting the s/min_dist
    plt.subplot(412)
    plt.plot("s", "sel_min_dist", data=debug_data, label="minimum distance", color='b')
    plt.axhline(y=SLECOL_BUFFER, color='black', linestyle='--', label="buffer")
    plt.xlabel("s (m)")
    plt.ylabel("distance (cm)")
    plt.title("Minimum distance per Arc length")
    plt.ylim(-0.01, max(debug_data["sel_min_dist"])*1.2)  
    plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    ## Plotting the s/manipulability
    plt.subplot(413)
    plt.plot("s", "mani", data=debug_data, label="manip", color='b')
    plt.axhline(y=MANI_BUFFER, color='black', linestyle='--', label="buffer")
    plt.xlabel("s (m)")
    plt.ylabel("Manipulability")
    plt.title("Manipulability per Arc length")
    plt.ylim(-0.01, max(debug_data["mani"])*1.2)  
    plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    ## Plotting the s/contouring error
    plt.subplot(414)
    plt.plot("s", "contour_error", data=debug_data, label="Contour Error", color='b')
    plt.xlabel("s (m)")
    plt.ylabel("Error (m)")
    plt.title("Contouring Error per Arc length")
    plt.ylim(-max(debug_data["contour_error"])*0.3, max(debug_data["contour_error"])*1.2)  
    plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_obs", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()
    main(args)