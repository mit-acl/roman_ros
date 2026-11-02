std_msgs/Header header

int32 robot1_id
int32 robot2_id

int32 submap1_id
int32 submap2_id

builtin_interfaces/Time robot1_time
builtin_interfaces/Time robot2_time

int32 num_associations
int32[] robot1_associated_segment_ids
int32[] robot2_associated_segment_ids

geometry_msgs/Pose transform_robot1_robot2
float64[36] covariance # rotation first

# Submap poses in each robot's odometry frame: odom1 <- submap1 (robot1's FLU frame at robot1_time)
# and odom2 <- submap2 (robot2's FLU frame at robot2_time); transform_robot1_robot2 is submap1 <- submap2
geometry_msgs/Pose robot1_submap_pose
geometry_msgs/Pose robot2_submap_pose
