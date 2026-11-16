std_msgs/Header header

int32 robot1_id
int32 robot2_id

int32 submap1_id
int32 submap2_id

builtin_interfaces/Time robot1_time
builtin_interfaces/Time robot2_time

int32 num_associations
float64 submap_similarity # submap descriptor (VPR) cosine similarity, NaN if submaps have no descriptor
int32[] robot1_associated_segment_ids
int32[] robot2_associated_segment_ids

geometry_msgs/Pose transform_robot1_robot2
float64[36] covariance # rotation first