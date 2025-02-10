import MPCC
import numpy as np
from numpy.linalg import inv
from math import pi
import time
from srmt2.planning_scene import PlanningScene

import matplotlib.pyplot as plt
import argparse
import scipy.io

## ROS library
import rclpy
from nav_msgs.msg import Path
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data

from main_utils import create_path_message, create_pred_path_message

np.set_printoptions(suppress=True, precision=3)

## MPCC parameters
param_value = {'cost': {
                    "qC" : 1000.0,
                    "qCNmult": 5,
                    "qL" : 100.0,
                    "qVs" : 20.0,

                    "qOri": 200,

                    "rdq"  : 0.002,
                    "rddq"  : 10,
                    "rdVs" : 0.1,
                    },
            'param': {
                    "desired_ee_velocity": 0.1,
                    "tol_sing": 0.018,
                    "tol_selcol": 1.0,
                    "tol_envcol": 1.0,
                    }
            }


q_mjc = np.zeros(7)
qdot_mjc = np.zeros(7)

def stateCallback(msg: JointState):
    global q_mjc, qdot_mjc
    q_mjc = np.array(msg.position)
    qdot_mjc = np.array(msg.velocity)

def main(args):
    ## Create Planning Scene
    pc = PlanningScene(arm_names=["fr3"], arm_dofs=[7], base_link="world")

    # Create publishers
    node = rclpy.create_node('mpcc_node')
    splined_path_pub = node.create_publisher(Path, '/mpcc/splined_path', 10)
    local_path_pub = node.create_publisher(Path, '/mpcc/local_path', 10)
    # ee_speed_pub = node.create_publisher(Float32, '/mpcc/ee_speed', 1)
    # mani_pub = node.create_publisher(Float32, '/mpcc/mani', 1)
    # sel_min_dist_pub = node.create_publisher(Float32, '/mpcc/sel_min_dist', 1)
    # contour_error_pub = node.create_publisher(Float32, '/mpcc/contour_error', 1)
    
    # Create joint publisher and subscriber
    ctrl_input_pub = node.create_publisher(JointState, "/joint_commands", 1)
    state_sub = node.create_subscription(JointState, '/joint_states', stateCallback, qos_profile_sensor_data)

    ## Create mpc controller
    mpc = MPCC.MPCC()
    N = mpc.pred_horizon
    robot_dof = mpc.robot_dof

    ## Create robot data processor
    robot = MPCC.RobotModel()
    selcolNN = MPCC.SelfCollisionNN()
    selcolNN.setNeuralNetwork(input_size=robot_dof, output_size=1, hidden_layer_size=np.array([256, 64]), is_nerf=True)
    rclpy.spin_once(node, timeout_sec=0.01)

    ## Set initial state and control input
    state = np.concatenate([q_mjc, np.zeros(2)])
    input = np.concatenate([qdot_mjc, np.zeros(1)])
    
    mpc.setTrack(state)
    mpc.setParam(param_value)
    spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()
    spline_T = np.zeros([spline_pos.shape[0], 4, 4])
    for i, (posi, rot) in enumerate(zip(spline_pos, spline_ori)):
        spline_T[i, :3, :3] = rot
        spline_T[i, :3, 3] = posi
    splined_path_msg = create_path_message(node, spline_pos, spline_ori)

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
    
    time_idx=0
    qdot_pre = input[:robot_dof]

    while rclpy.ok():
        start = time.time()
        
        if time_idx == 1000:
            path = np.array([[[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]],
                             [[0, -1, 0, 0.2],
                              [1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]])
            mpc.setTrack(state, path)
            spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()
            spline_T = np.zeros([spline_pos.shape[0], 4, 4])
            for i, (posi, rot) in enumerate(zip(spline_pos, spline_ori)):
                spline_T[i, :3, :3] = rot
                spline_T[i, :3, 3] = posi
            splined_path_msg = create_path_message(node, spline_pos, spline_ori)
        

        ## run MPCC
        status, state, input, mpc_horizon, compute_time = mpc.runMPC(state, input)
        if status == False:
            print("MPC did not solve properly!!")
            break
        
        
        ## publish q_desired, qdot_desired
        joint_state_msg = JointState()
        joint_state_msg.position = (state[:robot_dof] + input[:robot_dof]*mpc.Ts).tolist()
        joint_state_msg.velocity = input[:robot_dof].tolist()
    
        ctrl_input_pub.publish(joint_state_msg)
        
        rclpy.spin_once(node, timeout_sec=0.001)
        
        ## update state from simulation
        state[:robot_dof] = q_mjc
        input[:robot_dof] = qdot_mjc
        state[robot_dof] = state[robot_dof] + state[robot_dof+1]*mpc.Ts
        state[robot_dof+1] = state[robot_dof+1] + input[robot_dof]*mpc.Ts
        
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
        mani = robot.getEEManipulability(q)
        contour_error = mpc.getContourError(s, x)
        pred_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        ref_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        for i in range(mpc.pred_horizon + 1):
            pred_ee_T[i, :3, :3] = robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof])
            pred_ee_T[i, :3, 3]  = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
            ref_ee_T[i, :3, 3], ref_ee_T[i, :3, :3] = mpc.getRefPose(mpc_horizon[i]["state"][-2])
            
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
        print("MPC time        : {:0.5f}".format(compute_time["total"]))
        print("===============================================================")

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
        local_path_msg = create_pred_path_message(node, pred_ee_T[:, :3, 3], pred_ee_T[:, :3, :3])
        # ref_local_path_msg = create_pred_path_message(node, ref_ee_T[:, :3, 3], ref_ee_T[:, :3, :3])
        # ee_speed_msg = Float32()
        # mani_msg = Float32()
        # sel_min_dist_msg = Float32()
        # contour_error_msg = Float32()
        # ee_speed_msg.data = x_speed
        # mani_msg.data = mani
        # sel_min_dist_msg.data = sel_min_dist
        # contour_error_msg.data = contour_error*100 # [m] -> [cm]
        
        splined_path_pub.publish(splined_path_msg)
        local_path_pub.publish(local_path_msg)
        # ref_local_path_pub.publish(ref_local_path_msg)
        # ee_speed_pub.publish(ee_speed_msg)
        # mani_pub.publish(mani_msg)
        # sel_min_dist_pub.publish(sel_min_dist_msg)
        # contour_error_pub.publish(contour_error_msg)
        
        

        ## End condition 
        if np.linalg.norm((spline_pos[-1] - x), 2) < 1E-2 and np.linalg.norm(MPCC.Log(spline_ori[-1].T @ rotation), 2) < 1E-2 and abs(state[-2] - spline_arc_length[-1]) < 1E-2:
            print("End point reached!!!")
            break
        end = time.time()
        elapsed = end - start
        print("elapsed: ", elapsed*1000)
        # if elapsed < mpc.Ts:
        #     time.sleep(mpc.Ts - elapsed)# rate.sleep()
        qdot_pre = qdot
        time_idx += 1

    ## Convert lists to NumPy arrays
    # for key in debug_data:
    #     debug_data[key] = np.array(debug_data[key])
    for key in time_data:
        time_data[key] = np.array(time_data[key])

    # ## Save the data to a .mat file
    # scipy.io.savemat("debug_data.mat", debug_data)
    # print("Data written to debug.mat")

    # scipy.io.savemat("time_data.mat", time_data)
    # print("Data written to debug.mat")


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
    
    args = parser.parse_args()
    main(args)