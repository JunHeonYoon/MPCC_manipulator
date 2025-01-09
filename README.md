# MPCC for manipulator

## Installation guide
### C++ dependency
Install C++ dependencies and build
```sh
git clone https://github.com/JunHeonYoon/MPCC_manipulator.git
cd MPCC_manipulator/cpp
./install.sh

mkdir build && cd build
cmake ..
make -j8
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
```
2. [Franka description](https://github.com/frankaemika/franka_ros/tree/develop/franka_description)
```sh
sudo apt install ros-$ROS_DISTRO-franka-ros
```
3. [Husky description](https://github.com/husky/husky/tree/noetic-devel/husky_description)
```sh
sudo apt install ros-$ROS_DISTRO-husky-desktop
```
4. [suhan_robot_model_tools](https://github.com/psh117/suhan_robot_model_tools)
```sh
sudo apt install ros-$ROS_DISTRO-combined-robot-hw ros-$ROS_DISTRO-trac-ik ros-$ROS_DISTRO-nlopt
sudo apt install libnlopt-dev libnlopt-cxx-dev libglfw3-dev

pip install tqdm matplotlib
```
```sh
cd ~/catkin_ws/src/

git clone https://github.com/psh117/gl_depth_sim
git clone https://github.com/psh117/suhan_robot_model_tools.git

cd .. && catkin build

source ../devel/setup.bash
```
5. [husky_panda](https://github.com/JunHeonYoon/husky_panda)
```sh
cd ~/catkin_ws/src/

git clone https://github.com/JunHeonYoon/husky_panda.git

cd .. && catkin build

source ../devel/setup.bash
```
## Running MPCC with Python
Robot visualize setup
```sh
roslaunch husky_panda_moveit_config vis.launch
```

Running MPCC
```sh
cd MPCC_manipulator/python
python3 main_w_sim.py
```
