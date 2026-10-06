std_msgs/Header header
int32 robot_id
int32 segment_id
geometry_msgs/Point position  # Position in odom frame
float64 volume
float64[4] shape_attributes # volume, linearity, planarity, scattering
float64[] semantic_descriptor # semantic embedding
float32[] dense_points # optional flattened Nx3 points (x0,y0,z0,x1,...) in odom frame, empty if not published
