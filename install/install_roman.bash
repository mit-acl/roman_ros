#!/bin/bash
ROMAN_ROS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."
ROMAN_DIR="$ROMAN_ROS_DIR/../roman"
cd $ROMAN_DIR

# Install CLIPPER
git submodule update --init --recursive
mkdir dependencies/clipper/build
cd dependencies/clipper/build
cmake .. && make && make pip-install

# pip install
cd $ROMAN_DIR
pip install --no-build-isolation git+https://github.com/CASIA-IVA-Lab/FastSAM.git@4d153e9
pip install .
# TODO figure out how to get ros setup.py to include these
# (or just include them in roman setup.py)
pip install transforms3d gdown

# download weights
mkdir -p $ROMAN_ROS_DIR/weights
cd $ROMAN_ROS_DIR/weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
gdown 'https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv'

# maybe handle this somewhere else
sudo apt install ros-${ROS_DISTRO}-tf-transformations
sudo apt install ros-${ROS_DISTRO}-image-transport