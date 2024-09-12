import mpcc
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

N = 10
SLECOL_BUFFER = 1.0
MANI_BUFFER = 0.01


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
        quat = mpcc.RotToQuat(ori)
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
            quat = mpcc.RotToQuat(ori)
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
             time_data, min_dist_real_data, min_dist_pred_data, mani_data):

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


    with open('../cpp/Params/track.json', 'r') as f:
        track_data = json.load(f)

    integrator = mpcc.Integrator()
    robot = mpcc.RobotModel()
    robot_dof = robot.num_q
    selcolNN = mpcc.SelfCollisionNN()
    selcolNN.setNeuralNetwork(input_size=robot_dof, output_size=1, hidden_layer_size=np.array([256, 64]), is_nerf=True)

    mpc = mpcc.MPCC()

    state = np.array([0., 0., 0., -pi/2, 0., pi/2, pi/4, 0., 0.])
    input = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    mpc.setTrack(state)
    spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()


    # global_path_msg = create_path_message1(track_data, pred_ee_posi_set[0,0:3])
    splined_path_msg = create_path_message2(spline_pos, spline_ori)


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

    param_value = {'cost': {'qC': 500.34952437742376, 
                            'qCNmult': 9.608598867805625, 
                            'qL': 100.215373616038, 
                            'qVs': 8.857749970111477, 
                            # 'qVs': 2.0, 
                            'qOri': 8.530613025311919, 
                            # 'qC_reduction_ratio': 0.14511793559277225, 
                            'qC_reduction_ratio': 1, 
                            # 'qL_increase_ratio': 3.609714386084117, 
                            'qL_increase_ratio': 1, 
                            # 'qOri_reduction_ratio': 0.7380133441510759,
                            'qOri_reduction_ratio': 1,
                            }}

    # mpc.setParam(param_value)

    
    time_idx=0
    while not rospy.is_shutdown():
        start = time()
        status, state, input, mpc_horizon, compute_time = mpc.runMPC(state, input)
        if status == False:
            print("MPC did not solve properly!!")
            break
        state = integrator.simTimeStep(state, input)

        pc.display(state[:robot_dof])
        real_min_dist = pc.min_distance(state[:robot_dof])*100

        ee_pos = robot.getEEPosition(state[:robot_dof])
        ee_ori = robot.getEEOrientation(state[:robot_dof])
        ee_vel = robot.getEEJacobianv(state[:robot_dof]) @ input[:robot_dof]
        min_dist, _ = selcolNN.calculateMlpOutput(state[:robot_dof])
        mani = robot.getEEManipulability(state[:robot_dof])


        print("===============================================================")
        print("time step   : ",time_idx)
        # print("state       : ", state)
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
            plt_func(fig, selcol_ax, mani_ax, min_dist_true_line, min_dist_pred_line, mani_line, sim_time_data, min_dist_real_data, min_dist_pred_data, mani_data)

        debug_data["q"].append(state[:robot_dof]) 
        debug_data["qdot"].append(input[:robot_dof])
        debug_data["min_dist"].append(min_dist[0])
        debug_data["mani"].append(mani)

        pred_ee_pose = np.zeros([mpc.pred_horizon + 1, 7])
        ref_ee_pose  = np.zeros([mpc.pred_horizon + 1, 7])
        for i in range(mpc.pred_horizon + 1):
            pred_ee_pose[i,:3] = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
            pred_ee_pose[i,3:] = mpcc.RotToQuat(robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof]))
            ref_pos, ref_ori = mpc.getRefPose(mpc_horizon[i]["state"][-2])
            ref_ee_pose[i,:3] = ref_pos
            ref_ee_pose[i,3:] = mpcc.RotToQuat(ref_ori)

        local_path_msg = create_pred_path_message(pred_ee_pose[:,:3], pred_ee_pose[:,3:])
        ref_local_path_msg = create_pred_path_message(ref_ee_pose[:,:3], ref_ee_pose[:,3:])
        # global_path_pub.publish(global_path_msg)
        splined_path_pub.publish(splined_path_msg)
        local_path_pub.publish(local_path_msg)
        ref_local_path_pub.publish(ref_local_path_msg)


        debug_data["pred_ee_pose"].append(pred_ee_pose)
        debug_data["ref_ee_pose"].append(ref_ee_pose)

        time_data["total"].append(compute_time["total"])
        time_data["set_qp"].append(compute_time["set_qp"])
        time_data["solve_qp"].append(compute_time["solve_qp"])
        time_data["get_alpha"].append(compute_time["get_alpha"])

        if np.linalg.norm((spline_pos[-1] - ee_pos), 2) < 1E-2 and np.linalg.norm(mpcc.Log(spline_ori[-1].T @ ee_ori), 2) and abs(state[-2] - spline_arc_length[-1]) < 1E-2:
            print("End point reached!!!")
            break
        
        end = time()
        elapsed = end - start
        if elapsed < mpc.Ts:
            sleep(mpc.Ts - elapsed)
        
        time_idx += 1

    with open('splined_path.txt', 'w') as splined_path_file:
        for pos, ori in zip(spline_pos, spline_ori):
            quaternion = mpcc.RotToQuat(ori)
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
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()
    main(args)