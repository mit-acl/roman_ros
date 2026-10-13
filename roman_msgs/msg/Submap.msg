# A ROMAN submap
# header.stamp: submap reference time
# header.frame_id: the robot's FLU reference frame
std_msgs/Header header
int32 robot_id
int32 submap_id
# Submap segments, expressed in header.frame_id at header.stamp (the true
# submap frame defined by submap_pose), not the gravity-aligned or odometry frame.
Segment[] segments
# Pose of header.frame_id at header.stamp, expressed in odom_frame_id.
# The parent frame is explicitly the robot's odometry frame (odom).
geometry_msgs/Pose submap_pose
# Pose of the gravity-aligned submap frame in odom_frame_id at header.stamp.
# Same position and yaw as submap_pose, with roll and pitch removed.
geometry_msgs/Pose submap_gravity_aligned_pose
string odom_frame_id

# Submap descriptor (empty if none)
std_msgs/Float64MultiArray descriptors
