# MPCC for manipulator
This code run at ROS2 humble with Ubuntu 22.04.

## Installation guide
### C++ dependency
Install C++ dependencies and build
```sh
git clone -b humble https://github.com/JunHeonYoon/MPCC_manipulator.git
cd MPCC_manipulator/cpp
./install.sh

mkdir build && cd build
cmake ..
make -j30
```

If your cmake version lower than 3.18, than update cmake
```sh
# Download new version of cmake at https://cmake.org/download/
cd path/to/cmake/folder

make -j30 && sudo make install

echo 'export PATH=$HOME/cmake-install/bin:$PATH' >> ~/.bashrc
echo 'export CMAKE_PREFIX_PATH=$HOME/cmake-install:$CMAKE_PREFIX_PATH' >> ~/.bashrc

source ~/.bashrc
```

Check if install was done successfully
```sh
cd build
./MPCC_TEST
```

### Python dependency
1. [MoveIt](https://moveit.ai/)
```sh
sudo apt install ros-$ROS_DISTRO-moveit
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
source ~/.bashrc
```
2. [Franka description](https://github.com/frankaemika/franka_description)
```sh
cd ~/ros2_ws/src
git clone https://github.com/frankaemika/franka_description.git
cd ../
colcon build
```
3. [Husky description](https://github.com/clearpathrobotics/clearpath_common/tree/humble)
```sh
cd ~/ros2_ws/src
git clone -b humble https://github.com/clearpathrobotics/clearpath_common.git 
cd ../
colcon build
```
4. [suhan_robot_model_tools2](https://github.com/JunHeonYoon/suhan_robot_model_tools/tree/humble)
```sh
sudo apt install ros-$ROS_DISTRO-eigenpy
cd ~/ros2_ws/src
git clone -b humble https://github.com/JunHeonYoon/suhan_robot_model_tools.git
cd ../
colcon build
```
5. [husky_fr3](https://github.com/JunHeonYoon/husky_fr3_ros2)
```sh
cd ~/ros2_ws/src
git clone https://github.com/JunHeonYoon/husky_fr3_ros2.git
cd ../
colcon build

source install/setup.bash
```
## Running MPCC with Python
Robot visualize setup
```sh
ros2 launch husky_fr3_moveit_config vis.launch.py
```

Running MPCC
```sh
cd MPCC_manipulator/python
python3 main_w_sim.py
```

## User parameter
Users can modify parameters and path in [Params](https://github.com/JunHeonYoon/MPCC_manipulator/tree/humble/cpp/Params)

### [bounds](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/bounds.json)
Upper, lower bound for state and input
- joint angle(q)
- joint velocity(qdot)
- joint acceleration(qddot)
- path parameter(s)
- velocity of s(vs)
- acceleration of s(dVs) 

### [config](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/config.json)
- n_sim: Limit for simulation [tick]
- Ts: Sampling time for MPCC [sec]

### [cost](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/cost.json)
Weighing scalar paramters
- qC: Contouring error cost
- qCNmult: Multiplication factor for terminal consouring error cost
- qL: Lag error cost
- qVs: Velocity of path parameter error cost
- qOri: Orientation error cost
- qSing: Singularity maximazation cost
- rdq: Joint velocity regularization cost
- rddq: Change of joint velocity regularization cost
- rdVs: Acceleration of path parameter regularization cost

### [model](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/model.json)
MPCC model parameters
- max_dist_proj:
- desired_ee_velocity: desired velocity for path parameter
- s_trust_region:
- deaccelerate_ratio:
- tol_sing: tolerence for singularity avoidance constraint
- tol_selcol: tolerence for self-collision avoidance constraint [cm]
- tol_envcol: tolerence for environment-collision avoidance constraint [cm]
 
### [noramalization](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/normalization.json)
Normalization for state and control input setting QP
  
### [sqp](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/sqp.json)
  
### [track](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/track.json)
Reference path for MPCC.
This can be generated by using [track.py](https://github.com/JunHeonYoon/MPCC_manipulator/blob/master/cpp/Params/track.py)
