# A ROMAN submap
# header.stamp: submap reference time
# header.frame_id: the robot's odometry frame
std_msgs/Header header
int32 robot_id
int32 submap_id
# Submap segments, expressed in the submap's gravity-aligned frame (the submap pose with
# roll and pitch removed), not in the odometry frame
Segment[] segments
# Submap pose (FLU body frame at the reference time) in the odometry frame
geometry_msgs/Pose pose
# Submap descriptor (empty if none)
std_msgs/Float64MultiArray descriptors
