# Segment Track ROS

This is ROS wrapper code for running `segment_track` in real-time.

# Install

`pip install -r requirements.txt`

and `catkin build` your workspace

# Running

Set environment variables:

```
WEST_POINT_2023_PATH # path to west point data root
SETUP_SEG_TRACK # command to source python environment 
SEG_TRACK_WS # path to ROS workspace containing segment_track_ros
```

`tmuxp load ./tmux/offline_jackal.yaml`

---

This research is supported by Ford Motor Company, DSTA, ONR, and
ARL DCIST under Cooperative Agreement Number W911NF-17-2-0181.