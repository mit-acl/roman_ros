import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rclpy
from builtin_interfaces.msg import Time
import roman_msgs.msg as roman_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs

import ros2_numpy as rnp

from roman.map.observation import Observation
from roman.object.segment import Segment, SegmentMinimalData

# Function to convert a float timestamp to ROS 2 Time
def float_to_ros_time(float_time):
    ros_time = Time()
    ros_time.sec = int(float_time)
    ros_time.nanosec = int((float_time % 1.0) * 1e9)
    return ros_time

def time_stamp_to_float(stamp):
    return rclpy.time.Time.from_msg(stamp).nanoseconds * 1e-9

def observation_from_msg(observation_msg: roman_msgs.Observation):
    """
    Convert observation message to observation data class

    Args:
        observation_msg (roman_msgs.Observation): observation message

    Returns:
        Observation: observation data class
    """
    observation = Observation(
        time=time_stamp_to_float(observation_msg.stamp),
        pose=rnp.numpify(observation_msg.pose),
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
        roman_msgs.Observation: observation message
    """
    observation_msg = roman_msgs.Observation(
        stamp=float_to_ros_time(observation.time),
        pose=geometry_msgs.Pose(
            position=rnp.msgify(geometry_msgs.Point, observation.pose[:3,3]),
            orientation=rnp.msgify(geometry_msgs.Quaternion, Rot.from_matrix(observation.pose[:3,:3]).as_quat())
        ),
        img_width=int(observation.mask_downsampled.shape[1]),
        img_height=int(observation.mask_downsampled.shape[0]),
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
        roman_msgs.Segment: segment message
    """
    e = segment.normalized_eigenvalues()
    segment_msg = roman_msgs.Segment(
        header=std_msgs.Header(stamp=float_to_ros_time(segment.last_seen)),
        robot_id=robot_id,
        segment_id=segment.id,
        position=rnp.msgify(geometry_msgs.Point, centroid_from_segment(segment)),
        # volume=estimate_volume(segment.points) if segment.points is not None else 0.0,
        volume=segment.volume,
        shape_attributes=[segment.volume, segment.linearity(e), segment.planarity(e), segment.scattering(e)]
    )
    return segment_msg

def msg_to_segment(segment_msg: roman_msgs.Segment) -> SegmentMinimalData:
    """
    Convert segment message to segment data class

    Args:
        segment_msg (roman_msgs.Segment): segment message

    Returns:
        Segment: segment data class
    """
    segment = SegmentMinimalData(
        id=segment_msg.segment_id,
        center=np.array([segment_msg.position.x, segment_msg.position.y, segment_msg.position.z]),
        volume=segment_msg.shape_attributes[0],
        linearity=segment_msg.shape_attributes[1],
        planarity=segment_msg.shape_attributes[2],
        scattering=segment_msg.shape_attributes[3],
        semantic_descriptor=None,
        extent=None,
        first_seen=None,
        last_seen=time_stamp_to_float(segment_msg.header.stamp),
    )
    return segment

def centroid_from_segment(segment: Segment):
    """
    Method to get a single point representing a segment.

    Args:
        segment (Segment): segment object

    Returns:
        np.array, shape=(3,): representative point
    """
    if segment.points is not None:
        pt = np.mean(segment.points, axis=0)
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