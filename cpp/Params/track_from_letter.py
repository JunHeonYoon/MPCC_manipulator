import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Function to detect the moving point between frames
def get_moving_point(prev_frame, curr_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Apply a binary threshold to get a binary image
    _, thresh = cv2.threshold(diff, 1.0, 255, cv2.THRESH_BINARY)  # Lowered threshold for more sensitivity
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour assuming it's the moving point
        max_contour = max(contours, key=cv2.contourArea)
        # Get the center of the contour
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None

# Path to the input video file
video_path = 'DYROS_letter.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Parameters
frame_interval = 1  # Extract points every frame
points = []

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error reading the first frame")
    exit()

# Process the video
frame_count = 0
while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        point = get_moving_point(prev_frame, curr_frame)
        if point:
            points.append(point)
    
    prev_frame = curr_frame.copy()
    frame_count += 1

# Release the video capture object
cap.release()

# Extract the video frame height for y-coordinate adjustment
frame_height = prev_frame.shape[0]

# Convert the extracted points into arrays for plotting, adjusting the y-coordinate
x_points, y_points = zip(*points)
x_points = np.array(x_points)
y_points = np.array([frame_height - y for y in y_points])  # Invert y-coordinates

# Scale data
x_points = x_points*(0.5/1100)
y_points = y_points*(0.5/1100)

# Plot the extracted points as a continuous line
plt.figure(figsize=(10, 5))
plt.plot(x_points, y_points, 'k',linewidth=2)
plt.scatter(x_points, y_points, s=10)
plt.title('Extracted Drawn Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()

rot = R.from_matrix([[1,  0,  0],
                     [0, -1, 0],
                     [0,  0, -1]])
quat = rot.as_quat()
quat_list = np.tile(quat, (x_points.size, 1))

total_desc = f""" 
{{
    "X": {list(map(np.double, np.zeros(x_points.shape)))},
    "Y": {list(map(np.double, x_points))},
    "Z": {list(map(np.double, y_points))},
    "quat_X": {list(map(np.double, quat_list[:,0]))},
    "quat_Y": {list(map(np.double, quat_list[:,1]))},
    "quat_Z": {list(map(np.double, quat_list[:,2]))},
    "quat_W": {list(map(np.double, quat_list[:,3]))}
}} 
"""

open('track.json','w').write(total_desc)