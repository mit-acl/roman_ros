std_msgs/Header header # header.stamp: same as last_seen
int32 robot_id
int32 segment_id
float64 first_seen # first time the segment was seen (s)
float64 last_seen # last time the segment was seen (s)
geometry_msgs/Point position  # Position in odom frame
float64 volume
float64[4] shape_attributes # volume, linearity, planarity, scattering
float64[] semantic_descriptor # semantic embedding