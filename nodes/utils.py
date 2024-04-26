import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rospy
import ros_numpy as rnp

from segment_track.observation import Observation
from segment_track.segment import Segment
import segment_slam_msgs.msg as segment_slam_msgs
import geometry_msgs.msg as geometry_msgs

def observation_from_msg(observation_msg: segment_slam_msgs.Observation):
    """
    Convert observation message to observation data class

    Args:
        observation_msg (segment_slam_msgs.Observation): observation message

    Returns:
        Observation: observation data class
    """
    observation = Observation(
        time=observation_msg.stamp.to_sec(),
        pose=rnp.numpify(observation_msg.pose),
        pixel=np.array(observation_msg.pixel),
        width=observation_msg.mask_width,
        height=observation_msg.mask_height,
        mask=np.array(observation_msg.mask).reshape(
            (observation_msg.img_height, observation_msg.img_width)
        ) if observation_msg.mask else None,
        mask_downsampled=np.array(observation_msg.mask).reshape(
            (observation_msg.img_height, observation_msg.img_width)
        ) if observation_msg.mask else None,
        point_cloud=(np.array(observation_msg.point_cloud).reshape((-1, 3)) 
                     if observation_msg.point_cloud else None),
    )
    return observation

def observation_to_msg(observation: Observation):
# def observation_to_msg(observation: Observation, img_width: int, img_height: int):
    """
    Convert observation data class to observation message

    Args:
        observation (Observation): observation data class

    Returns:
        segment_slam_msgs.Observation: observation message
    """
    observation_msg = segment_slam_msgs.Observation(
        stamp=rospy.Time.from_sec(observation.time),
        pose=geometry_msgs.Pose(
            position=rnp.msgify(geometry_msgs.Point, observation.pose[:3,3]),
            orientation=rnp.msgify(geometry_msgs.Quaternion, Rot.from_matrix(observation.pose[:3,:3]).as_quat())
        ),
        pixel=observation.pixel.tolist(),
        img_width=int(observation.mask_downsampled.shape[1]),
        img_height=int(observation.mask_downsampled.shape[0]),
        mask_width=int(observation.width),
        mask_height=int(observation.height),
        mask=observation.mask_downsampled.flatten().astype(np.int8).tolist() if observation.mask is not None else None,
        point_cloud=(observation.point_cloud.flatten().tolist() 
                     if observation.point_cloud is not None else None),
    )
    return observation_msg

"""
segment.msg

std_msgs/Header header
int32 robot_id
int32 segment_id
geometry_msgs/Point position  # Position in odom frame
float64 volume
"""

def segment_to_msg(robot_id: int, segment: Segment):
    """
    Convert segment data class to segment message

    Args:
        segment (Segment): segment data class

    Returns:
        segment_slam_msgs.Segment: segment message
    """
    segment_msg = segment_slam_msgs.Segment(
        header=rospy.Header(stamp=rospy.Time.from_sec(segment.last_seen)),
        robot_id=robot_id,
        segment_id=segment.id,
        position=rnp.msgify(geometry_msgs.Point, centroid_from_segment(segment)),
        volume=estimate_volume(segment.points) if segment.points is not None else 0.0,
    )
    return segment_msg

def centroid_from_segment(segment: Segment):
    """
    Method to get a single point representing a segment.

    Args:
        segment (Segment): segment object

    Returns:
        np.array, shape=(3,): representative point
    """
    if segment.points is not None:
        pt = np.median(segment.points, axis=0)
        pt[2] = np.min(segment.points[:,2])
        return pt
    else:
        return None
    
def estimate_volume(points, axis_discretization=10):
    """Estimate the volume by voxelizing the bounding box and checking whether sampled points 
    are inside each voxel"""
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    x_seg_size = (max_bounds[0] - min_bounds[0])/ axis_discretization
    y_seg_size = (max_bounds[1] - min_bounds[1])/ axis_discretization
    z_seg_size = (max_bounds[2] - min_bounds[2])/ axis_discretization
    volume = 0.0
    for i in range(axis_discretization):
        x = min_bounds[0] + x_seg_size * i 
        for j in range(axis_discretization):
            y = min_bounds[1] + y_seg_size * j 
            for k in range(axis_discretization):
                z = min_bounds[2] + z_seg_size * k 
                if np.any(np.bitwise_and(points < np.array([x + x_seg_size, y + y_seg_size, z + z_seg_size]), 
                                            points > np.array([x, y, z]))):
                    volume += x_seg_size * y_seg_size * z_seg_size
    return volume