import numpy as np
import matplotlib.pyplot as plt
from srmt.planning_scene import PlanningScene
import json
from time import time, sleep
import argparse


# ROS library
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

N = 10
SLECOL_BUFFER = 1.0
MANI_BUFFER = 0.01

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

def create_path_message2(track_data):
    path = Path()
    path.header = Header()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'panda_link0'

    initial_x = track_data[0,0] - 0.
    initial_y = track_data[0,1] - 0.
    initial_z = track_data[0,2] - 0.

    for x, y, z, quat_x, quat_y, quat_z, quat_w in zip(track_data[:,0], track_data[:,1], track_data[:,2],
                                                       track_data[:,3], track_data[:,4], track_data[:,5], track_data[:,6]):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = x - initial_x + 0.55450 
        pose.pose.position.y = y - initial_y + 0.
        pose.pose.position.z = z - initial_z + 0.52110
        pose.pose.orientation.x = quat_x
        pose.pose.orientation.y = quat_y
        pose.pose.orientation.z = quat_z
        pose.pose.orientation.w = quat_w
        path.poses.append(pose)
    
    return path

# Function to create a Path message from the dataset
def create_pred_path_message(pred_data):
    path = Path()
    path.header = Header()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'panda_link0'

    pred_data = pred_data.reshape(-1,3)

    for ee_pos in pred_data:
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = ee_pos[0]
        pose.pose.position.y = ee_pos[1]
        pose.pose.position.z = ee_pos[2]
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = 0
        pose.pose.orientation.w = 1
        path.poses.append(pose)
    
    return path

import numpy as np
import matplotlib.pyplot as plt

# This is the plt_func function with necessary corrections
def plt_func(fig, selcol_ax, mani_ax, 
             min_dist_true_line, min_dist_pred_line, mani_line, 
             time_data, min_dist_real_data, min_dist_pred_data, mani_data):

    # min_dist_true_line.set_data(time_data, min_dist_real_data)
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
    # lines_selcol = [min_dist_true_line, min_dist_pred_line, buffer_line_selcol]
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

    # Read simulated dataset
    dataset = np.loadtxt("build/debug.txt")
    print(dataset.shape)
    q_set = dataset[:,0:7]
    qdot_set = dataset[:,7:14]
    pred_min_dist_set = dataset[:,14]
    mani_set = dataset[:,15]
    pred_ee_posi_set = dataset[:, 16:16+(3*N)]
    ref_ee_posi_set = dataset[:, 16+(3*N):16+2*(3*N)]

    if(args.plot):
        # Animated plotter
        plt.ion()
        fig, (selcol_ax, mani_ax) = plt.subplots(2, 1, figsize=(12, 8))

        min_dist_true_line, = selcol_ax.plot([],[], label='ans', color="blue", linewidth=4.0, linestyle='--')
        min_dist_pred_line, = selcol_ax.plot([],[], label='pred', color = "red", linewidth=2.0)
        min_dist_pred_line, = selcol_ax.plot([],[], label='min dist', color = "red", linewidth=2.0)
        selcol_ax.legend()
        selcol_ax.set_ylim([min(np.min(pred_min_dist_set), SLECOL_BUFFER) - 5, np.max(pred_min_dist_set) + 5])
        selcol_ax.grid()

        # Second subplot
        mani_line, = mani_ax.plot([], [], label='mani', color="red", linewidth=2.0)
        mani_ax.legend()
        mani_ax.set_ylim([min(np.min(mani_set), MANI_BUFFER) - 0.05, np.max(mani_set) + 0.05])
        mani_ax.grid()



    global_path_pub = rospy.Publisher('/mpcc/global_path', Path, queue_size=10)
    splined_path_pub = rospy.Publisher('/mpcc/splined_path', Path, queue_size=10)
    local_path_pub = rospy.Publisher('/mpcc/local_path', Path, queue_size=10)
    ref_local_path_pub = rospy.Publisher('/mpcc/ref_local_path', Path, queue_size=10)

    with open('Params/track.json', 'r') as f:
        track_data = json.load(f)
    
    global_path_msg = create_path_message1(track_data, pred_ee_posi_set[0,0:3])

    splined_path_set = np.loadtxt("build/splined_path.txt")
    splined_path_msg = create_path_message2(splined_path_set)
    
    time_data = np.zeros((1))
    min_dist_real_data = np.zeros((1))
    min_dist_pred_data = np.zeros((1))
    mani_data = np.zeros((1))
    if(args.plot):
        sleep(2)

    for iter in range(q_set.shape[0]):
        start = time()
        joint_state = q_set[iter,:]
        pc.display(joint_state)
        min_dist = pc.min_distance(joint_state)*100

        time_data = np.append(time_data, np.array([iter*0.01]), axis=0)
        min_dist_real_data = np.append(min_dist_real_data, np.array([min_dist]), axis=0)
        min_dist_pred_data = np.append(min_dist_pred_data, np.array([pred_min_dist_set[iter]]), axis=0)
        mani_data = np.append(mani_data, np.array([mani_set[iter]]), axis=0)

        if time_data.shape[0] > 10:
            time_data = time_data[-10:]
            min_dist_real_data = min_dist_real_data[-10:]
            min_dist_pred_data = min_dist_pred_data[-10:]
            mani_data = mani_data[-10:]

        if(args.plot):
            plt_func(fig, selcol_ax, mani_ax, min_dist_true_line, min_dist_pred_line, mani_line, time_data, min_dist_real_data, min_dist_pred_data, mani_data)
        else:
            plt.pause(0.01)
        
        local_path_msg = create_pred_path_message(pred_ee_posi_set[iter,:])
        ref_local_path_msg = create_pred_path_message(ref_ee_posi_set[iter,:])
        global_path_pub.publish(global_path_msg)
        splined_path_pub.publish(splined_path_msg)
        local_path_pub.publish(local_path_msg)
        ref_local_path_pub.publish(ref_local_path_msg)
        end = time()
        elapsed = end - start
        if(args.plot):
            if elapsed < 0.2:
                sleep(0.2 - elapsed)
        
        print('time elapsed:', time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()
    main(args)