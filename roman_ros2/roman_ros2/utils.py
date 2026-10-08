import numpy as np
from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass
from typing import Tuple, Union

import rclpy
from builtin_interfaces.msg import Time
import roman_msgs.msg as roman_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import visualization_msgs.msg as visualization_msgs
from pose_graph_tools_msgs.msg import PoseGraph, PoseGraphEdge

import ros2_numpy as rnp

from roman.map.observation import Observation
from roman.map.map import Submap
from roman.object.segment import Segment, SegmentMinimalData

class MapColors:

    ego_map: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    other_map: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    correspondences: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    @classmethod
    def dimmed(cls, color):
        return tuple([c * 0.5 for c in color])
    

# Function to convert a float timestamp to ROS 2 Time
def float_to_ros_time(float_time):
    ros_time = Time()
    ros_time.sec = int(float_time)
    ros_time.nanosec = int((float_time % 1.0) * 1e9)
    return ros_time

def time_stamp_to_float(stamp):
    return rclpy.time.Time.from_msg(stamp).nanoseconds * 1e-9

def numpy_to_float64_multiarray(array: np.ndarray) -> std_msgs.Float64MultiArray:
    """
    Convert a NumPy array to a ROS Float64MultiArray with proper layout (sizes and strides).
    """
    msg = std_msgs.Float64MultiArray()
    msg.data = array.astype(np.float64).flatten().tolist() # store flattened data
    msg.layout.data_offset = 0

    dims = []
    shape = array.shape
    for i, size in enumerate(shape):
        dim = std_msgs.MultiArrayDimension()
        dim.label = f"dim{i}"
        dim.size = size
        dim.stride = int(np.prod(shape[i:]))
        dims.append(dim)
    msg.layout.dim = dims
    return msg

def float64_multiarray_to_numpy(msg: std_msgs.Float64MultiArray) -> np.ndarray:
    """
    Convert a ROS Float64MultiArray to a NumPy array using the layout information.
    """
    shape = tuple(dim.size for dim in msg.layout.dim)
    array = np.array(msg.data, dtype=np.float64).reshape(shape)
    return array

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
        semantic_descriptor=(np.array(observation_msg.descriptor).reshape((-1,))
                        if observation_msg.descriptor else None),
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
        descriptor=observation.semantic_descriptor.flatten().tolist() if observation.semantic_descriptor is not None else [],
    )
    return observation_msg

def descriptor_to_array_msg(descriptor: Union[np.ndarray, None]) -> std_msgs.Float64MultiArray:
    return numpy_to_float64_multiarray(descriptor) if descriptor is not None else std_msgs.Float64MultiArray()

def descriptor_from_array_msg(descriptor_array_msg: std_msgs.Float64MultiArray) -> Union[np.ndarray, None]:
    return float64_multiarray_to_numpy(descriptor_array_msg) if descriptor_array_msg.data else None

def frame_descriptor_to_msg(descriptor_array_msg: std_msgs.Float64MultiArray, stamp, pose: np.ndarray) -> roman_msgs.FrameDescriptor:
    descriptor_msg = roman_msgs.FrameDescriptor()
    descriptor_msg.stamp = stamp
    descriptor_msg.pose = rnp.msgify(geometry_msgs.Pose, pose)
    descriptor_msg.descriptor = descriptor_array_msg
    return descriptor_msg

def frame_descriptor_from_msg(descriptor_msg: roman_msgs.FrameDescriptor) -> Tuple[float, np.ndarray, np.ndarray]:
    descriptor = descriptor_from_array_msg(descriptor_msg.descriptor)
    pose = rnp.numpify(descriptor_msg.pose)
    time_stamp = time_stamp_to_float(descriptor_msg.stamp)
    return time_stamp, pose, descriptor

"""
segment.msg

std_msgs/Header header # header.stamp: same as last_seen
int32 robot_id
int32 segment_id
float64 first_seen # first time the segment was seen (s)
float64 last_seen # last time the segment was seen (s)
geometry_msgs/Point position  # Position in odom frame
float64 volume
"""

def segment_to_msg(robot_id: int, segment: Union[Segment, SegmentMinimalData]):
    """
    Convert segment data class to segment message

    Args:
        robot_id (int): id of the robot the segment belongs to
        segment (Union[Segment, SegmentMinimalData]): full or minimal-data segment

    Returns:
        roman_msgs.Segment: segment message
    """
    segment_msg = roman_msgs.Segment(
        header=std_msgs.Header(stamp=float_to_ros_time(segment.last_seen)),
        robot_id=robot_id,
        segment_id=segment.id,
        first_seen=float(segment.first_seen),
        last_seen=float(segment.last_seen),
        # full segments compute their center from points (respecting center ref);
        # minimal-data segments return the stored center
        position=rnp.msgify(geometry_msgs.Point, np.asarray(segment.center, dtype=np.float64).reshape(3)),
        # volume=estimate_volume(segment.points) if segment.points is not None else 0.0,
        volume=segment.volume,
        shape_attributes=[segment.volume, segment.linearity, segment.planarity, segment.scattering],
        semantic_descriptor=segment.semantic_descriptor.flatten().tolist() if segment.semantic_descriptor is not None else [],
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
        semantic_descriptor=np.array(segment_msg.semantic_descriptor) if len(segment_msg.semantic_descriptor) != 0 else None,
        extent=None,
        first_seen=segment_msg.first_seen,
        last_seen=segment_msg.last_seen,
    )
    return segment

def submap_to_msg(robot_id: int, submap: Submap, odom_frame: str) -> roman_msgs.Submap:
    """
    Convert a submap to a submap message. Segments stay in the submap's gravity-aligned
    frame (submap.segment_frame must be 'submap_gravity_aligned').

    Args:
        robot_id (int): id of the robot the submap belongs to
        submap (Submap): submap whose segments are SegmentMinimalData or Segments
        odom_frame (str): odometry frame the submap pose is expressed in

    Returns:
        roman_msgs.Submap: submap message
    """
    assert submap.segment_frame == 'submap_gravity_aligned', \
        f"ERROR: expected segments in the submap gravity-aligned frame, got {submap.segment_frame}."
    submap_msg = roman_msgs.Submap(
        header=std_msgs.Header(stamp=float_to_ros_time(submap.time), frame_id=odom_frame),
        robot_id=robot_id,
        submap_id=submap.id,
        segments=[segment_to_msg(robot_id, seg) for seg in submap.segments],
        pose=rnp.msgify(geometry_msgs.Pose, submap.pose_flu),
        descriptors=descriptor_to_array_msg(submap.descriptor),
    )
    return submap_msg

def msg_to_submap(submap_msg: roman_msgs.Submap) -> Submap:
    """
    Convert a submap message to a submap (segments in the submap gravity-aligned frame).

    Args:
        submap_msg (roman_msgs.Submap): submap message

    Returns:
        Submap: submap with SegmentMinimalData segments
    """
    return Submap(
        id=submap_msg.submap_id,
        time=time_stamp_to_float(submap_msg.header.stamp),
        segments=[msg_to_segment(seg_msg) for seg_msg in submap_msg.segments],
        pose_flu=rnp.numpify(submap_msg.pose).astype(np.float64),
        segment_frame='submap_gravity_aligned',
        descriptor=descriptor_from_array_msg(submap_msg.descriptors),
    )

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

def default_marker(position: Tuple[float, float, float], color: Tuple[float, float, float], id=0) -> visualization_msgs.Marker:
    """
    Create a default marker for visualization

    Args:
        position (Tuple[float, float, float]): position of the marker

    Returns:
        visualization_msgs.Marker: marker message
    """
    marker = visualization_msgs.Marker()
    marker.header.frame_id = "map" # TODO: not sure what frame we want to do this visualization
    marker.header.stamp = rclpy.time.Time().to_msg()
    marker.ns = "default"
    marker.id = id
    marker.type = visualization_msgs.Marker.SPHERE
    marker.action = visualization_msgs.Marker.ADD
    marker.pose.position = rnp.msgify(geometry_msgs.Point, np.array(position).reshape(-1))
    marker.pose.orientation = rnp.msgify(geometry_msgs.Quaternion, Rot.from_euler('xyz', [0, 0, 0]).as_quat())
    marker.scale.x = 0.25
    marker.scale.y = 0.25
    marker.scale.z = 0.25
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
    return marker

def lc_to_pose_graph_msg(robot_id1: int, robot_id2: int, submap1: Submap, submap2: Submap, 
                         T_submap1_submap2: np.ndarray, covariance: np.ndarray, stamp):
    edge = PoseGraphEdge()
    edge.header.stamp = stamp
    
    edge.key_from = int(submap1.time*1e9)
    edge.key_to = int(submap2.time*1e9)
    edge.robot_from = robot_id1
    edge.robot_to = robot_id2
    edge.type = PoseGraphEdge.LOOPCLOSE
    
    # Pose Graph Tools assumes T_to_from or T_submap2_submap1
    edge.pose = rnp.msgify(geometry_msgs.Pose, np.linalg.inv(T_submap1_submap2))
    edge.covariance = covariance.reshape(-1).tolist()
    
    pg = PoseGraph()
    pg.header.stamp = stamp
    pg.edges.append(edge)
    
    return pg
    
def lc_to_msg(robot_id1: int, robot_id2: int, submap1: Submap, submap2: Submap, 
        associations: np.ndarray, T_submap1_submap2: np.ndarray, covariance: np.ndarray, stamp):
    lc_msg = roman_msgs.LoopClosure()
    lc_msg.header.stamp = stamp
    
    lc_msg.robot1_id = robot_id1
    lc_msg.robot2_id = robot_id2

    lc_msg.submap1_id = submap1.id
    lc_msg.submap2_id = submap2.id
    
    lc_msg.robot1_time = float_to_ros_time(submap1.time)
    lc_msg.robot2_time = float_to_ros_time(submap2.time)
    
    lc_msg.num_associations = len(associations)
    lc_msg.robot1_associated_segment_ids = [submap1.segments[i].id for i in associations[:,0]]
    lc_msg.robot2_associated_segment_ids = [submap2.segments[i].id for i in associations[:,1]]

    lc_msg.transform_robot1_robot2 = rnp.msgify(geometry_msgs.Pose, T_submap1_submap2)
    lc_msg.covariance = covariance.reshape(-1).tolist()
    
    return lc_msg


class TimingFifo:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = []
    
    def update(self, value: float):
        self.data.append(value)
        self.data = self.data[-self.max_size:]
    
    def mean(self) -> float:
        return np.mean(self.data) if self.data else 0.0
    
    def __len__(self) -> int:
        return len(self.data)