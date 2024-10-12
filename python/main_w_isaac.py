import MPC
import numpy as np
from math import pi

import numpy as np
import json
from time import time, sleep
import argparse
import matplotlib.pyplot as plt
# import threading


# ROS library
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
# from std_msgs.msg import Int8MultiArray

N = 10
SLECOL_BUFFER = 1.0
MANI_BUFFER = 0.01

joint_angle = None
joint_vel = None
# voxel = np.zeros(int(36*36*36), dtype=np.float32)

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
            quat = MPC.RotToQuat(ori)
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

def joint_callback(data: JointState):
    conf_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    global joint_angle, joint_vel
    joint_angle = np.array([
        data.position[joint_idx] 
        for joint_idx, joint_name in enumerate(data.name) 
        if joint_name in conf_joint_names
    ], dtype=np.float32)
    joint_vel = np.array([
        data.velocity[joint_idx] 
        for joint_idx, joint_name in enumerate(data.name) 
        if joint_name in conf_joint_names
    ], dtype=np.float32)

# def voxel_callback(data: Int8MultiArray):
#         # 메시지로부터 배열의 크기 복원
#         # dim_x = data.layout.dim[0].size
#         # dim_y = data.layout.dim[1].size
#         # dim_z = data.layout.dim[2].size

#         global voxel
#         voxel = np.array(data.data, dtype=np.float32)

def main(args):
    rospy.init_node('MPCC_ISAAC', anonymous=True)

    # rospy.Subscriber('/masked_voxel', Int8MultiArray, voxel_callback)
    rospy.Subscriber('/joint_states', JointState, joint_callback)
    joint_command_pub = rospy.Publisher('/joint_command', JointState, queue_size=10)
    global_path_pub = rospy.Publisher('/mpcc/global_path', Path, queue_size=10)
    splined_path_pub = rospy.Publisher('/mpcc/splined_path', Path, queue_size=10)
    local_path_pub = rospy.Publisher('/mpcc/local_path', Path, queue_size=10)
    ref_local_path_pub = rospy.Publisher('/mpcc/ref_local_path', Path, queue_size=10)

    with open('../cpp/Params/track.json', 'r') as f:
        track_data = json.load(f)

    integrator = mpc.Integrator()
    robot = mpc.RobotModel()
    robot_dof = robot.num_q
    selcolNN = mpc.SelfCollisionNN()
    selcolNN.setNeuralNetwork(input_size=robot_dof, output_size=1, hidden_layer_size=np.array([256, 64]), is_nerf=True)

    mpc = mpc.MPCC()

    state = np.zeros(9)
    input = np.zeros(8)
    while(joint_angle is None):
        pass
    
    state[0:robot_dof] = joint_angle
    input[0:robot_dof] = joint_vel
    print(state)

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
    time_data["set_env"] = []
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
                            'rVee': 0.5
                            }}

    # mpc.setParam(param_value)

    joint_command_msg = JointState()
    joint_command_msg.header = Header()
    joint_command_msg.name = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel'] + ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    # joint_command_msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    joint_command_msg.effort = []

    # for time_idx in range(10000):

    rate = rospy.Rate(1./mpc.Ts)
    time_idx = 0
    while not rospy.is_shutdown():
        start = time()
        # status, state, input, mpc_horizon, compute_time = mpc.runMPC(state, voxel)
        status, state, input, mpc_horizon, compute_time = mpc.runMPC(state, input)
        if status == False:
            print("MPC did not solve properly!!")
            break
        state = integrator.simTimeStep(state, input)

        joint_command_msg.header.stamp = rospy.Time.now()
        joint_command_msg.position = [0., 0., 0., 0.] + state[:robot_dof].tolist()
        # joint_command_msg.position = state[:robot_dof].tolist()
        joint_command_msg.velocity = [0., 0., 0., 0.] + input[:robot_dof].tolist()
        # joint_command_msg.velocity = input[:robot_dof].tolist()
        joint_command_pub.publish(joint_command_msg)
        
        end = time()
        elapsed = end - start
        if elapsed < mpc.Ts:
            sleep(mpc.Ts - elapsed)
        # rate.sleep()

        # state[:robot_dof] = joint_angle
        # input[:robot_dof] = joint_vel

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

        pred_ee_pose = np.zeros([mpc.pred_horizon + 1, 7])
        ref_ee_pose  = np.zeros([mpc.pred_horizon + 1, 7])
        for i in range(mpc.pred_horizon + 1):
            pred_ee_pose[i,:3] = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
            pred_ee_pose[i,3:] = mpc.RotToQuat(robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof]))
            ref_pos, ref_ori = mpc.getRefPose(mpc_horizon[i]["state"][-2])
            ref_ee_pose[i,:3] = ref_pos
            ref_ee_pose[i,3:] = mpc.RotToQuat(ref_ori)

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
        time_data["set_env"].append(compute_time["set_env"])

        if np.linalg.norm((spline_pos[-1] - ee_pos), 2) < 1E-2 and np.linalg.norm(mpc.Log(spline_ori[-1].T @ ee_ori), 2) and abs(state[-2] - spline_arc_length[-1]) < 1E-2:
            print("End point reached!!!")
            break
        
        time_idx += 1

    with open('splined_path.txt', 'w') as splined_path_file:
        for pos, ori in zip(spline_pos, spline_ori):
            quaternion = mpc.RotToQuat(ori)
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
    time_data["set_env"] = np.array(time_data["set_env"])

    print("mean nmpc time[sec]: {:0.6f} ".format(np.mean(time_data["total"])))
    print("max nmpc time[sec]: {:0.6f} ".format(np.max(time_data["total"])))

    # Plotting the computation times
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
    plt.ylim(-0.01, mpc.Ts*2.5)  
    plt.xlim(0, len(time_data["total"]))
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)