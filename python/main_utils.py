import MPCC
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

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
        quat = MPCC.RotToQuat(ori)
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
            quat = MPCC.RotToQuat(ori)
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
