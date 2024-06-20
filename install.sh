#!/bin/bash

ws_name=${1:-"segment_slam_ws"}
python_env=${2:-"segment_slam"}

# Create a new workspace
mkdir -p ~/$ws_name/src
mkdir -p ~/$ws_name/python

# Install ROS packages
pushd ~/$ws_name/src
# git clone https://gitlab.com/mit-acl/sparse_mapping/segment_track_ros.git
git clone git@gitlab.com:mit-acl/sparse_mapping/segment_track_ros.git
git clone git@gitlab.com:mit-acl/sparse_mapping/segment_slam_msgs.git
git clone https://github.com/eric-wieser/ros_numpy.git
git clone git@gitlab.com:mit-acl/sparse_mapping/segment_slam.git
cd segment_slam && git checkout feature/volume_align && cd ..
git clone git@github.com:borglab/gtsam.git
cd gtsam && git checkout 4.2a9 && cd ..
git clone git@github.com:ethz-asl/gflags_catkin.git
git clone git@github.com:ethz-asl/glog_catkin.git
git clone git@github.com:catkin/catkin_simple.git
cd ~/$ws_name
catkin config -DCMAKE_BUILD_TYPE=Release -DGTSAM_TANGENT_PREINTEGRATION=OFF \
              -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF 
catkin build

# Install python packages
sudo apt install -y python3-venv
if [ ! -d ~/.envs/$python_env ]; then
    mkdir -p ~/.envs
    python3 -m venv ~/.envs/$python_env
fi
source ~/.envs/$python_env/bin/activate
pip install --upgrade pip
pip install rospkg

cd ~/$ws_name/python
git clone https://github.com/mbpeterson70/robot_utils.git
pip install -e robot_utils
git clone git@gitlab.com:mit-acl/sparse_mapping/segment_track.git
cd segment_track && git checkout mason/dev && cd ..
pip install -e segment_track
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
pip install -r FastSAM/requirements.txt
pip install git+https://github.com/openai/CLIP.git
wget -P FastSAM/weights https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
popd