import MPC
import numpy as np
from math import pi

import numpy as np
import matplotlib.pyplot as plt
from srmt.planning_scene import PlanningScene
import json
from time import time, sleep
import argparse
import matplotlib.pyplot as plt


# ROS library
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


np.set_printoptions(suppress=True, precision=3)

# Function to create a Path message from the track data
def create_path_message1(track_data, init_position):
    path = Path()
    path.header = Header()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'panda_link0'

    initial_x = track_data['X'][0] - 0.
    initial_y = track_data['Y'][0] - 0.
    initial_z = track_data['Z'][0] - 0.

    for x, y, z, quat_x, quat_y, quat_z, quat_w in zip(track_data['X'], track_data['Y'], track_data['Z'], 
                                                       track_data['quat_X'], track_data['quat_Y'], 
                                                       track_data['quat_Z'], track_data['quat_W']):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = x - initial_x + init_position[0] 
        pose.pose.position.y = y - initial_y + init_position[1]
        pose.pose.position.z = z - initial_z + init_position[2]
        pose.pose.orientation.x = quat_x
        pose.pose.orientation.y = quat_y
        pose.pose.orientation.z = quat_z
        pose.pose.orientation.w = quat_w
        path.poses.append(pose)
    
    return path

def create_path_message2(track_pos, track_ori):
    assert track_pos.shape[0] == track_ori.shape[0]
    path = Path()
    path.header = Header()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'panda_link0'


    initial_x = track_pos[0,0] - 0.
    initial_y = track_pos[0,1] - 0.
    initial_z = track_pos[0,2] - 0.

    for pos, ori in zip(track_pos, track_ori):
        quat = MPC.RotToQuat(ori)
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = pos[0] - initial_x + 0.55450 
        pose.pose.position.y = pos[1] - initial_y + 0.
        pose.pose.position.z = pos[2] - initial_z + 0.52110
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        path.poses.append(pose)
    
    return path

# Function to create a Path message from the dataset
def create_pred_path_message(track_pos, track_ori):
    assert track_pos.shape[0] == track_ori.shape[0]
    path = Path()
    path.header = Header()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'panda_link0'

    for pos, ori in zip(track_pos, track_ori):
        if(ori.shape == (3,3)):
            quat = mpc.RotToQuat(ori)
        else:
            quat = track_ori.flatten()
            
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        path.poses.append(pose)
    
    return path

def plt_func(fig, selcol_ax, mani_ax, 
             min_dist_true_line, min_dist_pred_line, mani_line, 
             time_data, min_dist_real_data, min_dist_pred_data, mani_data,
             SLECOL_BUFFER, MANI_BUFFER):

    min_dist_true_line.set_data(time_data, min_dist_real_data)
    min_dist_pred_line.set_data(time_data, min_dist_pred_data)
    mani_line.set_data(time_data, mani_data)

    # Adjust xlims with a small margin if the range is too small
    if time_data[0] == time_data[-1]:
        x_min, x_max = time_data[0] - 0.01, time_data[0] + 0.01
    else:
        x_min, x_max = time_data[0], time_data[-1]

    selcol_ax.set_xlim(x_min, x_max)
    mani_ax.set_xlim(x_min, x_max)
    
    # Add horizontal lines manually to the legend
    buffer_line_selcol = selcol_ax.hlines(y=SLECOL_BUFFER, xmin=x_min, xmax=x_max, label='buffer (selcol)', color="black", linewidth=2.0)
    buffer_line_mani = mani_ax.hlines(y=MANI_BUFFER, xmin=x_min, xmax=x_max, label='buffer (mani)', color="black", linewidth=2.0)

    # Ensure all lines are included in the legend
    lines_selcol = [min_dist_true_line, min_dist_pred_line, buffer_line_selcol]
    lines_selcol = [min_dist_pred_line, buffer_line_selcol]
    labels_selcol = [line.get_label() for line in lines_selcol]
    selcol_ax.legend(lines_selcol, labels_selcol)

    lines_mani = [mani_line, buffer_line_mani]
    labels_mani = [line.get_label() for line in lines_mani]
    mani_ax.legend(lines_mani, labels_mani)

    selcol_ax.set_title("Minimum distance")
    selcol_ax.set_xlabel("Time (sec)")
    selcol_ax.set_ylabel("Distance (cm)")

    mani_ax.set_title("Manipulability Index")
    mani_ax.set_xlabel("Time (sec)")
    mani_ax.set_ylabel("Manipulability Index")

    fig.canvas.draw()
    fig.canvas.flush_events()


def main(args):
    # Create Planning Scene
    pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="world")

    global_path_pub = rospy.Publisher('/mpcc/global_path', Path, queue_size=10)
    splined_path_pub = rospy.Publisher('/mpcc/splined_path', Path, queue_size=10)
    local_path_pub = rospy.Publisher('/mpcc/local_path', Path, queue_size=10)
    ref_local_path_pub = rospy.Publisher('/mpcc/ref_local_path', Path, queue_size=10)



    with open('../cpp/Params/track.json', 'r') as f:
        track_data = json.load(f)

    integrator = MPC.Integrator()
    robot = MPC.RobotModel()
    robot_dof = robot.num_q
    selcolNN = MPC.SelfCollisionNN()
    selcolNN.setNeuralNetwork(input_size=robot_dof, output_size=1, hidden_layer_size=np.array([256, 64]), is_nerf=True)

    mpc = MPC.MPC()

    N = mpc.pred_horizon

    state = np.array([0., 0., 0., -pi/2, 0., pi/2, pi/4])
    input = np.array([0., 0., 0., 0., 0., 0., 0.])
    mpc.setTrack(state)
    total_ref_traj_posi, total_ref_traj_ori = mpc.getTotalTrajectory()

    # global_path_msg = create_path_message1(track_data, pred_ee_posi_set[0,0:3])
    total_traj_msg = create_path_message2(total_ref_traj_posi, total_ref_traj_ori)


    debug_data = {}
    debug_data["q"] = []
    debug_data["qdot"] = []
    debug_data["s"] = []
    debug_data["vs"] = []
    debug_data["ee_speed"] = []
    debug_data["min_dist"] = []
    debug_data["mani"] = []
    debug_data["Ec"] = []
    debug_data["pred_ee_pose"] = [] # (x, y, z, q_x, q_y, q_z, q_w) * (N+1)
    debug_data["ref_ee_pose"] = []  # (x, y, z, q_x, q_y, q_z, q_w) * (N+1)

    time_data = {}
    time_data["total"] = []
    time_data["set_qp"] = []
    time_data["solve_qp"] = []
    time_data["get_alpha"] = []

    param_value = {'cost': {
                            "qE" : 300.0,
                            "qENmult": 10,

                            "qOri": 10,

                            "rdq"  : 0.01
                            },
                    'param': {
                            "desired_ee_velocity" : 0.15,

                            "tol_sing": 0.01,
                            "tol_selcol": 1.0,
                    }}

    mpc.setParam(param_value)
    SLECOL_BUFFER = param_value["param"]["tol_selcol"]
    MANI_BUFFER = param_value["param"]["tol_sing"]

    if(args.plot):
        # Animated plotter
        plt.ion()
        fig, (selcol_ax, mani_ax) = plt.subplots(2, 1, figsize=(12, 8))

        min_dist_true_line, = selcol_ax.plot([],[], label='ans', color="blue", linewidth=4.0, linestyle='--')
        min_dist_pred_line, = selcol_ax.plot([],[], label='pred', color = "red", linewidth=2.0)
        min_dist_pred_line, = selcol_ax.plot([],[], label='min dist', color = "red", linewidth=2.0)
        selcol_ax.legend()
        selcol_ax.set_ylim([SLECOL_BUFFER - 5, 15])
        selcol_ax.grid()

        # Second subplot
        mani_line, = mani_ax.plot([], [], label='mani', color="red", linewidth=2.0)
        mani_ax.legend()
        mani_ax.set_ylim([MANI_BUFFER - 0.05, 0.2])
        mani_ax.grid()

    sim_time_data = np.zeros((1))
    min_dist_real_data = np.zeros((1))
    min_dist_pred_data = np.zeros((1))
    mani_data = np.zeros((1))
    
    time_idx=0
    rate = rospy.Rate(1./mpc.Ts)
    while not rospy.is_shutdown():
        status, state, input, mpc_horizon, compute_time = mpc.runMPC(state, input, time_idx)
        if status == False:
            print("MPC did not solve properly!!")
            break
        state = integrator.simTimeStep(state, input)

        pc.display(state[:robot_dof])
        real_min_dist = pc.min_distance(state[:robot_dof])*100

        ee_posi = robot.getEEPosition(state[:robot_dof])
        ee_ori = robot.getEEOrientation(state[:robot_dof])
        ee_vel = robot.getEEJacobianv(state[:robot_dof]) @ input[:robot_dof]
        min_dist, _ = selcolNN.calculateMlpOutput(state[:robot_dof])
        mani = robot.getEEManipulability(state[:robot_dof])
        s_opt, contour_error = mpc.getContourError(time_idx * mpc.Ts * param_value["param"]["desired_ee_velocity"], ee_posi)

        print("===============================================================")
        print("time step   : ",time_idx)
        # print("state       : ", state)
        print("q           : ", state[:robot_dof])
        print("qdot        : ", input[:robot_dof])
        print("x           : ", ee_posi)
        print("xdot        : {:0.5f}".format(np.linalg.norm(ee_vel)))
        print("xdot        : ", ee_vel)
        print("R           :\n", ee_ori)
        print("mani        : {:0.5f}".format(mani))
        print("min dist[cm]: {:0.5f}".format(min_dist[0]))
        print("MPC time    : {:0.5f}".format(compute_time["total"]))
        print("===============================================================")

        sim_time_data = np.append(sim_time_data, np.array([time_idx*mpc.Ts]), axis=0)
        min_dist_real_data = np.append(min_dist_real_data, np.array([real_min_dist]), axis=0)
        min_dist_pred_data = np.append(min_dist_pred_data, np.array([min_dist[0]]), axis=0)
        mani_data = np.append(mani_data, np.array([mani]), axis=0)

        if sim_time_data.shape[0] > 10:
            sim_time_data = sim_time_data[-10:]
            min_dist_real_data = min_dist_real_data[-10:]
            min_dist_pred_data = min_dist_pred_data[-10:]
            mani_data = mani_data[-10:]

        if(args.plot):
            plt_func(fig, selcol_ax, mani_ax, min_dist_true_line, min_dist_pred_line, mani_line, sim_time_data, min_dist_real_data, min_dist_pred_data, mani_data, SLECOL_BUFFER, MANI_BUFFER)

        debug_data["q"].append(state[:robot_dof]) 
        debug_data["qdot"].append(input[:robot_dof])
        debug_data["s"].append(s_opt)
        debug_data["vs"].append(param_value["param"]["desired_ee_velocity"])
        debug_data["ee_speed"].append(np.linalg.norm(ee_vel))
        debug_data["min_dist"].append(min_dist[0])
        debug_data["mani"].append(mani)
        debug_data["Ec"].append(contour_error)

        pred_ee_pose = np.zeros([mpc.pred_horizon + 1, 7])
        ref_ee_pose  = np.zeros([mpc.pred_horizon + 1, 7])
        ref_n_traj_posi, ref_n_traj_ori = mpc.getNTrajectory(time_idx)
        for i in range(mpc.pred_horizon + 1):
            pred_ee_pose[i,:3] = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
            pred_ee_pose[i,3:] = MPC.RotToQuat(robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof]))
            ref_ee_pose[i,:3] = ref_n_traj_posi[i]
            ref_ee_pose[i,3:] = MPC.RotToQuat(ref_n_traj_ori[i])

        local_path_msg = create_pred_path_message(pred_ee_pose[:,:3], pred_ee_pose[:,3:])
        ref_local_traj_msg = create_pred_path_message(ref_ee_pose[:,:3], ref_ee_pose[:,3:])
        # global_path_pub.publish(global_path_msg)
        splined_path_pub.publish(total_traj_msg)
        local_path_pub.publish(local_path_msg)
        ref_local_path_pub.publish(ref_local_traj_msg)


        debug_data["pred_ee_pose"].append(pred_ee_pose)
        debug_data["ref_ee_pose"].append(ref_ee_pose)

        time_data["total"].append(compute_time["total"])
        time_data["set_qp"].append(compute_time["set_qp"])
        time_data["solve_qp"].append(compute_time["solve_qp"])
        time_data["get_alpha"].append(compute_time["get_alpha"])

        if np.linalg.norm((total_ref_traj_posi[-1] - ee_posi), 2) < 1E-2 and np.linalg.norm(MPC.Log(total_ref_traj_ori[-1].T @ ee_ori), 2) < 1E-2 and (time_idx > total_ref_traj_posi.shape[0]):
            print("End point reached!!!")
            break
        
        rate.sleep()
        
        time_idx += 1

    with open('splined_path.txt', 'w') as splined_path_file:
        for pos, ori in zip(total_ref_traj_posi, total_ref_traj_ori):
            quaternion = MPC.RotToQuat(ori)
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

    # Plotting the computation times
    plt.figure(figsize=(14, 8))

    plt.plot(time_data["total"], label="Total Time", color='b')
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
    
    # Plotting the s/vs, ee_speed
    fig=plt.figure(figsize=(14, 8))
    fig.subplots_adjust(hspace=1)
    plt.subplot(411)
    plt.plot("s", "ee_speed", data=debug_data, label="ee_speed", color='r')
    plt.axhline(y=param_value["param"]["desired_ee_velocity"], color='black', linestyle='--', label="desired")

    plt.xlabel("s (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("EE Speed per Arc length")
    plt.ylim(-0.01, max(max(debug_data["ee_speed"]),param_value["param"]["desired_ee_velocity"])*1.2)  
    plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    # Plotting the s/min_dist
    # plt.figure(figsize=(14, 8))
    plt.subplot(412)

    plt.plot("s", "min_dist", data=debug_data, label="minimum distance", color='b')
    plt.axhline(y=SLECOL_BUFFER, color='black', linestyle='--', label="buffer")

    plt.xlabel("s (m)")
    plt.ylabel("distance (cm)")
    plt.title("Minimum distance per Arc length")
    plt.ylim(-0.01, max(debug_data["min_dist"])*1.2)  
    plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    # Plotting the s/manipulability
    # plt.figure(figsize=(14, 8))
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

    plt.subplot(414)

    plt.plot("s", "Ec", data=debug_data, label="Contour Error", color='b')

    plt.xlabel("s (m)")
    plt.ylabel("Error (m)")
    plt.title("Contouring Error per Arc length")
    plt.ylim(-max(debug_data["Ec"])*0.3, max(debug_data["Ec"])*1.2)  
    plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()
    main(args)