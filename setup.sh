#!/bin/bash

ws_name=${1:-"segment_slam_ws"}
python_env=${2:-"segment_slam"}

export SETUP_SEG_TRACK="source ~/.envs/$python_env/bin/activate && export PYTHONPATH=$PYTHONPATH:~/$ws_name/python"
export SEG_TRACK_WS="~/$ws_name"
export FASTSAM_WEIGHTS_PATH="~/$ws_name/src/FastSAM/weights/FastSAM.pt"