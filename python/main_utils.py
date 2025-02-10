import MPCC
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

# Function to create a Path message from the track data
def create_path_message(node, track_pos, track_ori):
    assert track_pos.shape[0] == track_ori.shape[0]
    path = Path()
    path.header = Header()
    path.header.stamp = node.get_clock().now().to_msg()
    path.header.frame_id = 'fr3_link0'

    for pos, ori in zip(track_pos, track_ori):
        quat = MPCC.RotToQuat(ori)
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

# Function to create a Path message from the dataset
def create_pred_path_message(node, track_pos, track_ori):
    assert track_pos.shape[0] == track_ori.shape[0]
    path = Path()
    path.header = Header()
    path.header.stamp = node.get_clock().now().to_msg()
    path.header.frame_id = 'fr3_link0'

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