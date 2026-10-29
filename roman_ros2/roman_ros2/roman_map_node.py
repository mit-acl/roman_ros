#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
import cv2 as cv
import copy
import pickle
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

# ROS imports
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterUninitializedException
from rclpy.time import Time
import cv_bridge
import ros2_numpy as rnp
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import tf2_ros
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty
import time

# ROS msgs
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import roman_msgs.msg as roman_msgs
import visualization_msgs.msg as visualization_msgs
from ros_system_monitor_msgs.msg import NodeInfoMsg

# robot_utils
from robotdatapy.camera import CameraParams

# ROMAN
from roman.map.mapper import Mapper, MapperParams
from roman.map.map import SubmapParams, aggregate_submap_point_cloud
from roman.params.submap_align_params import SubmapAlignParams
from roman.utils import DepthProjectionMetadata
from roman.object.segment import Segment
from roman.viz import visualize_segment_on_img

# relative
from roman_ros2.utils import observation_from_msg, descriptor_from_array_msg, segment_to_msg, \
    msg_to_segment, submap_to_msg, time_stamp_to_float, TimingFifo
from roman_ros2.submap_builder import SubmapBuilder

POINT_TIME_FIELDS = {
    'timestamp': ('<f8', False),    # Livox: absolute time, ns
    't': ('<u4', True),             # Ouster: ns after the header stamp
}


def pointcloud2_to_xyz_time(msg: sensor_msgs.PointCloud2):
    """(N, 3) float64 points and (N,) absolute per-point times in s (None if the cloud has none)."""
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in 'xyz'):
        raise ValueError("point cloud has no x, y, z fields")
    names, formats, offsets = ['x', 'y', 'z'], ['<f4'] * 3, [fields[k].offset for k in 'xyz']
    time_field = next((k for k in POINT_TIME_FIELDS if k in fields), None)
    if time_field is not None:
        names.append('time')
        formats.append(POINT_TIME_FIELDS[time_field][0])
        offsets.append(fields[time_field].offset)
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                      'itemsize': msg.point_step})
    if msg.is_bigendian:
        dtype = dtype.newbyteorder('>')
    data = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    points = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float64)
    if time_field is None:
        return points, None
    times = data['time'].astype(np.float64) * 1e-9
    if POINT_TIME_FIELDS[time_field][1]:
        times = times + time_stamp_to_float(msg.header.stamp)
    return points, times


def deskew_points(points, times, t_ref, sensor_pose_at, steps):
    """Points expressed in the sensor frame at t_ref, each moved with the sensor pose at its own time.

    The scan's time span is split into ``steps`` equal bins and every point in a bin uses the
    pose at the bin's middle; ``sensor_pose_at(t)`` returns T_odom_sensor.
    """
    t0, t1 = times.min(), times.max()
    if steps < 1 or t1 <= t0:
        return points
    T_ref_inv = np.linalg.inv(sensor_pose_at(t_ref))
    bins = np.minimum(((times - t0) / (t1 - t0) * steps).astype(int), steps - 1)
    out = np.empty_like(points)
    for k in range(steps):
        mask = bins == k
        if not mask.any():
            continue
        T = T_ref_inv @ sensor_pose_at(t0 + (k + 0.5) / steps * (t1 - t0))
        out[mask] = points[mask] @ T[:3, :3].T + T[:3, 3]
    return out


class RomanMapNode(Node):

    def __init__(self):
        super().__init__('roman_map_node')
        self.up = True

        # ros params
        self.declare_parameters(
            namespace='',
            parameters=[
                ("robot_id", 0),                        # robot id for this node
                ("config_path", ""),                    # non-ros ROMAN mapper.yaml param file
                ("submap_config_path", ""),             # submap descriptor settings from ROMAN alignment yaml
                ("submap_num_segments", 40),
                ("submap_overlapping_segments", 20),
                ("visualize", False),                   # whether to publish annotated images
                ("output_roman_map", ""),               # output file to save the map to (must kill node with ctrl-c)
                ("odom_frame_id", "odom"),              # odometry frame id to map objects in
                ("base_flu_frame_id", "base_link"),     # robot coordinate frame with xyz = forward left up
                ("object_ref", "center"),               # bottom_middle or center, what reference point to use for objects
                ("publish_active_segments", False),     # whether to wait till segments are inactive before publishing them
                ("nickname", "roman_map"),              # nickname for this node
                ("use_multiple_cams", False),           # whether mapper will get FastSAM measurements from multiple cameras
                ("timing_window", 10),                  # timing window used for report processing speed
                ("depth_pcd_stride", 4),                # depth pixels sampled for cloud aggregation
                ("pcd_source", "depth"),                # aggregated clouds from: depth (depth/image_raw)
                                                        # or pointcloud (PointCloud2 on pointcloud, e.g. a lidar)
                ("pointcloud_T_base_sensor", Parameter.Type.DOUBLE_ARRAY),  # optional row-major 4x4
                                                        # base_flu -> sensor mount, used instead of TF
                ("pointcloud_min_range", 0.0),          # m, drop closer points (e.g. the robot itself)
                ("pointcloud_deskew_steps", 0),         # >0: deskew with per-point times, this many poses per scan
                ("viz_num_objs", 20),                   # maximum number of objects to visualize
                ("viz_pts_per_obj", 250),               # number of points to visualize per object
                ("viz_min_viz_dt", 0.25),               # minimum time between visualizations
                ("viz_rotate_img", ""),                 # rotate image for visualization, options: CW, CCW, 180
                ("viz_pointcloud", False),               # whether to publish pointcloud of objects
                ("viz_object_map", False),              # whether to publish OBB markers of object map
                ("viz_object_map_dt", 1.0),             # how often to publish object map markers (seconds)
                ("viz_object_map_time_window", 60.0),   # only show objects seen in the last N seconds (-1 for all)
                ("viz_object_map_alpha", 0.5),          # transparency of object map markers
            ]
        )

        self.robot_id = self.get_parameter("robot_id").value
        self.visualize = self.get_parameter("visualize").value
        self.output_file = self.get_parameter("output_roman_map").value
        self.object_ref = self.get_parameter("object_ref").value
        self.base_flu_frame_id = self.get_parameter("base_flu_frame_id").value
        self.odom_frame_id = self.get_parameter("odom_frame_id").value
        self.publish_active_segments = self.get_parameter("publish_active_segments").value
        self.nickname = self.get_parameter("nickname").value
        self.use_multiple_cams = self.get_parameter("use_multiple_cams").value
        timing_window = self.get_parameter("timing_window").value
        self.timing_fifo = TimingFifo(timing_window)
        self.depth_pcd_stride = self.get_parameter("depth_pcd_stride").value
        if self.depth_pcd_stride < 1:
            raise ValueError("depth_pcd_stride must be positive")
        self.pcd_source = self.get_parameter("pcd_source").value
        if self.pcd_source not in ("depth", "pointcloud"):
            raise ValueError("pcd_source must be depth or pointcloud")
        try:
            T_base_sensor = self.get_parameter("pointcloud_T_base_sensor").value
        except ParameterUninitializedException:     # not set: use the cloud's frame from TF
            T_base_sensor = None
        self.T_base_pcd_sensor = None
        if T_base_sensor is not None and len(T_base_sensor) > 0:
            if len(T_base_sensor) != 16:
                raise ValueError("pointcloud_T_base_sensor must have 16 values (row-major 4x4)")
            self.T_base_pcd_sensor = np.asarray(T_base_sensor, dtype=np.float64).reshape(4, 4)
        self.pointcloud_min_range = self.get_parameter("pointcloud_min_range").value
        self.pointcloud_deskew_steps = self.get_parameter("pointcloud_deskew_steps").value

        config_path = self.get_parameter("config_path").value
        submap_config_path = self.get_parameter("submap_config_path").value
        align_params = (SubmapAlignParams.from_yaml(submap_config_path)
                        if submap_config_path else SubmapAlignParams())
        self.submap_params = SubmapParams.from_submap_align_params(align_params)
        self.submap_builder = SubmapBuilder(
            self.get_parameter("submap_num_segments").value,
            self.get_parameter("submap_overlapping_segments").value,
            self.submap_params)
        
        assert self.base_flu_frame_id != "", "base_flu_frame_id must be set"

        self.viz_object_map = self.get_parameter("viz_object_map").value
        if self.visualize:
            self.viz_num_objs = self.get_parameter("viz_num_objs").value
            self.viz_pts_per_obj = self.get_parameter("viz_pts_per_obj").value
            self.min_viz_dt = self.get_parameter("viz_min_viz_dt").value
            self.viz_rotate_img = self.get_parameter("viz_rotate_img").value
            self.viz_pointcloud = self.get_parameter("viz_pointcloud").value
            if self.viz_rotate_img == "":
                self.viz_rotate_img = None
        if self.viz_object_map:
            self.viz_object_map_dt = self.get_parameter("viz_object_map_dt").value
            self.viz_object_map_time_window = self.get_parameter("viz_object_map_time_window").value
            self.viz_object_map_alpha = self.get_parameter("viz_object_map_alpha").value
        if self.output_file != "":
            self.output_file = os.path.expanduser(self.output_file)
            output_file_parent = Path(self.output_file).parent
            output_file_parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Output file: {self.output_file}")
        else:
            self.output_file = None

        # mapper
        self.status_pub = self.create_publisher(NodeInfoMsg, "roman/roman_map/status", qos_profile=QoSProfile(depth=10))
        self.log_and_send_status("RomanMapNode setting up mapping...", status=NodeInfoMsg.STARTUP)
        self.log_and_send_status("RomanMapNode waiting for color camera info messages...", status=NodeInfoMsg.STARTUP)
        color_info_msg = self._wait_for_message("color/camera_info", sensor_msgs.CameraInfo)
        color_params = CameraParams.from_msg(color_info_msg)
        self.log_and_send_status("RomanMapNode received for color camera info messages...", status=NodeInfoMsg.STARTUP)

        if config_path != "":
            mapper_params = MapperParams.from_yaml(config_path)
        else:
            mapper_params = MapperParams()
        self.mapper = Mapper(
           mapper_params,
           camera_params=color_params
        )
        
        # handle transforming segments in ROS node to support listening to multiple cameras.
        # internally, the mapper assumes a single camera, we will say that the camera is pointed 
        # xyz = forward-right-up
        self.mapper.set_T_camera_flu(np.eye(4))
        self.T_camera_flu_set = False
        
        self._shutdown_complete = False
        self._shutdown_requested = False
        self._shutdown_timer = None

        self.setup_ros()

    def setup_ros(self):

        # ros publishers
        self.segments_pub = self.create_publisher(roman_msgs.Segment, "roman/segment_update", qos_profile=10)
        self.submap_pub = self.create_publisher(roman_msgs.Submap, "roman/submap", qos_profile=20)
        self.pulse_pub = self.create_publisher(std_msgs.Empty, "roman/pulse", qos_profile=10)

        # ros tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # separate callback groups so obs_cb and viz_cb can run in parallel
        self.obs_cb_group = MutuallyExclusiveCallbackGroup()
        self.viz_cb_group = MutuallyExclusiveCallbackGroup()
        self.shutdown_srv = self.create_service(
            Empty, "roman/shutdown", self.shutdown_cb, callback_group=self.obs_cb_group)

        # ros subscribers
        self.create_subscription(roman_msgs.ObservationArray, "roman/observations", self.obs_cb, 10,
                                 callback_group=self.obs_cb_group)

        if self.mapper.params.store_aggregated_pcds and self.pcd_source == "pointcloud":
            self.last_pointcloud_time = -np.inf
            self.pointcloud_sub = self.create_subscription(
                sensor_msgs.PointCloud2, "pointcloud", self.pointcloud_cb,
                qos_profile_sensor_data, callback_group=self.obs_cb_group)
        elif self.mapper.params.store_aggregated_pcds:
            self.depth_camera_info = None
            self.depth_bridge = cv_bridge.CvBridge()
            self.last_depth_time = -np.inf
            self.depth_info_sub = self.create_subscription(
                sensor_msgs.CameraInfo, "depth/camera_info", self.depth_info_cb,
                qos_profile_sensor_data, callback_group=self.obs_cb_group)
            self.depth_sub = self.create_subscription(
                sensor_msgs.Image, "depth/image_raw", self.depth_cb,
                qos_profile_sensor_data, callback_group=self.obs_cb_group)

        # visualization
        if self.visualize:
            self.last_viz_t = -np.inf

            self.create_subscription(sensor_msgs.Image, "color/image_raw", self.viz_cb, 10,
                                     callback_group=self.viz_cb_group)
            self.bridge = cv_bridge.CvBridge()
            self.annotated_img_pub = self.create_publisher(sensor_msgs.Image, "roman/annotated_img", qos_profile=10)
            self.object_points_pub = self.create_publisher(sensor_msgs.PointCloud, "roman/object_points", qos_profile=10)

        if self.viz_object_map:
            self.object_map_pub = self.create_publisher(visualization_msgs.MarkerArray,
                "roman/object_map", qos_profile=10)
            self.create_timer(self.viz_object_map_dt, self.object_map_timer_cb,
                              callback_group=self.viz_cb_group)

        self.log_and_send_status("ROMAN Map Node setup complete.", status=NodeInfoMsg.STARTUP)
        self.log_and_send_status("Waiting for observation.", status=NodeInfoMsg.STARTUP)

    def set_T_camera_flu(self, obs_array_msg):
        """
        Gets transform to set mapper T_camera_flu
        """
        # get transform from camera to base_link
        transform_stamped_msg = self.tf_buffer.lookup_transform(self.base_flu_frame_id,  obs_array_msg.header.frame_id, obs_array_msg.header.stamp, rclpy.duration.Duration(seconds=2.0))
        T_baselink_camera = rnp.numpify(transform_stamped_msg.transform).astype(np.float64)
        self.mapper.set_T_camera_flu(np.linalg.inv(T_baselink_camera))
        self.T_camera_flu_set = True

    def obs_cb(self, obs_array_msg):
        """
        Triggered by incoming observation messages
        """
        if not self.up:
            return

        # publish pulse
        map_size = len(self.mapper.segments) + \
                len(self.mapper.inactive_segments) + \
                len(self.mapper.segment_graveyard)
        self.log_and_send_status((f"Map size: {map_size}, "
            f"Avg time (ms): {np.round(self.timing_fifo.mean()*1000)}")
            if len(self.timing_fifo) > 0 else f"Map size: {map_size}",)
        self.pulse_pub.publish(std_msgs.Empty())

        frame_descriptor = descriptor_from_array_msg(obs_array_msg.frame_descriptor)
        self.submap_builder.update_frame(
            time_stamp_to_float(obs_array_msg.header.stamp),
            rnp.numpify(obs_array_msg.pose_flu).astype(np.float64), frame_descriptor)
        
        if len(obs_array_msg.observations) == 0:
            return
        
        start_t = time.time()
        observations = []
        for obs_msg in obs_array_msg.observations:
            observations.append(observation_from_msg(obs_msg))

        t = observations[0].time
        assert all([obs.time == t for obs in observations])

        inactive_ids = [segment.id for segment in self.mapper.inactive_segments]
        
        if not self.use_multiple_cams:
            if not self.T_camera_flu_set:
                self.set_T_camera_flu(obs_array_msg)
            pose = rnp.numpify(obs_array_msg.pose) # camera frame (T_camera_flu applied internally)
        else:
            pose = rnp.numpify(obs_array_msg.pose_flu) # base link frame

        self.mapper.update(time_stamp_to_float(obs_array_msg.header.stamp), pose, observations, frame_descriptor)
        updated_inactive_ids = [segment.id for segment in self.mapper.inactive_segments]
        new_inactive_ids = [seg_id for seg_id in updated_inactive_ids if seg_id not in inactive_ids]

        # publish segments
        segment: Segment
        for segment in self.mapper.inactive_segments:
            # TODO: this does not include a way to notify of a deleted segment
            if segment.last_seen == t or segment.id in new_inactive_ids:
                self.publish_segment(segment)
        if self.publish_active_segments:
            for segment in self.mapper.segments:
                self.publish_segment(segment)
        self.timing_fifo.update(time.time() - start_t)

    def depth_info_cb(self, msg: sensor_msgs.CameraInfo):
        self.depth_camera_info = msg

    def depth_cb(self, msg: sensor_msgs.Image):
        """Project a depth image and feed the mapper's independent scan window."""
        if not self.up or self.depth_camera_info is None:
            return
        t = time_stamp_to_float(msg.header.stamp)
        if t <= self.last_depth_time:
            return
        info = self.depth_camera_info
        if (info.width != msg.width or info.height != msg.height or
                (info.header.frame_id and info.header.frame_id != msg.header.frame_id)):
            self.get_logger().warning("Depth image and camera info dimensions/frames do not match.")
            return
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
            self.get_logger().warning("Invalid depth camera intrinsics.")
            return
        if msg.encoding not in ('16UC1', '32FC1'):
            self.get_logger().warning(f"Unsupported depth encoding: {msg.encoding}")
            return
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.odom_frame_id, msg.header.frame_id, msg.header.stamp,
                rclpy.duration.Duration(seconds=0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warning(f"Depth TF lookup failed: {ex}")
            return
        depth = self.depth_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        intrinsics = o3d.camera.PinholeCameraIntrinsic(info.width, info.height, fx, fy, cx, cy)
        metadata = DepthProjectionMetadata(self.depth_pcd_stride,
            1000.0 if msg.encoding == '16UC1' else 1.0, intrinsics)
        self.mapper.update_point_cloud(t, rnp.numpify(tf_msg.transform).astype(np.float64),
            depth, is_depth_img=True, depth_projection_metadata=metadata)
        self.last_depth_time = t

    def pointcloud_sensor_pose(self, frame_id, t):
        """T_odom_sensor at time t (float s), using pointcloud_T_base_sensor for the mount if set."""
        stamp = Time(nanoseconds=int(round(t * 1e9))).to_msg()
        timeout = rclpy.duration.Duration(seconds=0.5)
        if self.T_base_pcd_sensor is None:
            return rnp.numpify(self.tf_buffer.lookup_transform(
                self.odom_frame_id, frame_id, stamp, timeout).transform).astype(np.float64)
        T_odom_base = rnp.numpify(self.tf_buffer.lookup_transform(
            self.odom_frame_id, self.base_flu_frame_id, stamp, timeout).transform).astype(np.float64)
        return T_odom_base @ self.T_base_pcd_sensor

    def pointcloud_cb(self, msg: sensor_msgs.PointCloud2):
        """Feed a point cloud (e.g. a lidar scan) to the mapper's scan window."""
        if not self.up:
            return
        t = time_stamp_to_float(msg.header.stamp)
        if t <= self.last_pointcloud_time:
            return
        try:
            points, times = pointcloud2_to_xyz_time(msg)
        except ValueError as ex:
            self.get_logger().warning(f"Unusable point cloud: {ex}", throttle_duration_sec=10.0)
            return
        keep = np.all(np.isfinite(points), axis=1)
        if self.pointcloud_min_range > 0.0:
            keep &= np.linalg.norm(points, axis=1) >= self.pointcloud_min_range
        points = points[keep]
        try:
            T_odom_sensor = self.pointcloud_sensor_pose(msg.header.frame_id, t)
            if self.pointcloud_deskew_steps > 0 and times is not None and len(points):
                points = deskew_points(points, times[keep], t,
                    lambda ti: self.pointcloud_sensor_pose(msg.header.frame_id, ti),
                    self.pointcloud_deskew_steps)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warning(f"Point cloud TF lookup failed: {ex}", throttle_duration_sec=10.0)
            return
        self.mapper.update_point_cloud(t, T_odom_sensor, points)
        self.last_pointcloud_time = t

    def publish_segment(self, segment: Segment):
        if self.object_ref == 'bottom_middle':
            segment.set_center_ref('bottom_middle')
        segment_msg = segment_to_msg(self.robot_id, segment)
        self.segments_pub.publish(segment_msg)
        submap = self.submap_builder.update_segment(msg_to_segment(segment_msg))
        if submap is not None:
            if self.submap_params.submap_descriptor is not None and submap.descriptor is None:
                self.get_logger().warning(f"Submap {submap.id} has no descriptor; check FastSAM descriptor settings.")
            if self.mapper.params.store_aggregated_pcds:
                aggregate_submap_point_cloud(submap, self.mapper.aggregated_point_clouds,
                                             self.submap_params.point_cloud_voxel_size)
            self.submap_pub.publish(submap_to_msg(self.robot_id, submap, self.odom_frame_id, self.base_flu_frame_id))

    def viz_cb(self, img_msg):
        """
        Triggered by incoming odometry and image messages
        """
        if not self.up:
            return

        # rospy.logwarn("Received messages")
        t = time_stamp_to_float(img_msg.header.stamp)
        if t - self.last_viz_t < self.min_viz_dt:
            return
        else:
            self.last_viz_t = t
            
        cam_frame_id = img_msg.header.frame_id

        try:
            transform_stamped_msg = self.tf_buffer.lookup_transform(self.odom_frame_id, cam_frame_id, img_msg.header.stamp, rclpy.duration.Duration(seconds=2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warning("tf lookup failed")
            self.get_logger().warning(str(ex))
            return

        pose = rnp.numpify(transform_stamped_msg.transform).astype(np.float64)

        # snapshot and shallow-copy segments so visualization doesn't interfere
        # with the mapper (avoids get_segment_map which resets all memoized caches)
        segments = [copy.copy(seg) for seg in
                    self.mapper.segment_graveyard + self.mapper.inactive_segments + self.mapper.segments
                    if seg.num_points > 0]

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        graveyard_time = self.mapper.params.segment_graveyard_time
        for segment in segments:
            if segment.last_seen < t - graveyard_time - 10:
                continue
            img = visualize_segment_on_img(segment, pose, img)
            
        if self.viz_rotate_img is not None:
            if self.viz_rotate_img == "CW":
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            elif self.viz_rotate_img == "CCW":
                img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            elif self.viz_rotate_img == "180":
                img = cv.rotate(img, cv.ROTATE_180)
        
        annotated_img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        annotated_img_msg.header = img_msg.header
        self.annotated_img_pub.publish(annotated_img_msg)

    def object_map_timer_cb(self):
        """
        Publishes OBB markers for the object map at a fixed rate.
        """
        if not self.up:
            return
        now = self.get_clock().now()
        t = time_stamp_to_float(now.to_msg())

        # snapshot segments (same copy approach as viz_cb)
        segments = [copy.copy(seg) for seg in
                    self.mapper.segment_graveyard + self.mapper.inactive_segments + self.mapper.segments
                    if seg.num_points > 4]

        markers = []
        # first marker deletes all previous markers
        delete_marker = visualization_msgs.Marker()
        delete_marker.action = visualization_msgs.Marker.DELETEALL
        markers.append(delete_marker)

        for segment in segments:
            if self.viz_object_map_time_window > 0 and \
                    segment.last_seen < t - self.viz_object_map_time_window:
                continue

            try:
                obb = segment.obb
            except Exception:
                continue

            marker = visualization_msgs.Marker()
            marker.header.frame_id = self.odom_frame_id
            marker.header.stamp = now.to_msg()
            marker.ns = "object_map"
            marker.id = segment.id
            marker.type = visualization_msgs.Marker.CUBE
            marker.action = visualization_msgs.Marker.ADD

            center = np.asarray(obb.center)
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])

            quat = Rot.from_matrix(np.asarray(obb.R)).as_quat()  # [x, y, z, w]
            marker.pose.orientation.x = float(quat[0])
            marker.pose.orientation.y = float(quat[1])
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])

            extent = np.asarray(obb.extent)
            marker.scale.x = float(extent[0])
            marker.scale.y = float(extent[1])
            marker.scale.z = float(extent[2])

            color = segment.viz_color
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = self.viz_object_map_alpha

            markers.append(marker)

        self.object_map_pub.publish(visualization_msgs.MarkerArray(markers=markers))

    def log_and_send_status(self, note, status=NodeInfoMsg.NOMINAL):
        """
        Log a message and send it to the status topic.
        """
        self.get_logger().info(note)
        status_msg = NodeInfoMsg()
        status_msg.nickname = self.nickname
        status_msg.node_name = self.get_fully_qualified_name()
        status_msg.status = status
        status_msg.notes = note
        self.status_pub.publish(status_msg)

        return
    
    def shutdown_cb(self, request, response):
        self.shutdown()
        if self._shutdown_timer is None:
            # This shares the service's callback group, so it runs only after
            # the service callback returns and ROS has sent the response.
            self._shutdown_timer = self.create_timer(
                0.01, self._request_shutdown, callback_group=self.obs_cb_group)
        return response

    def _request_shutdown(self):
        self._shutdown_timer.cancel()
        self._shutdown_requested = True

    def shutdown(self):
        if self._shutdown_complete:
            return
        self.up = False
        if self.output_file is None:
            print("No file to save to.")
        if self.output_file is not None:
            print(f"Saving map to {self.output_file}...")
            self.mapper.make_pickle_compatible()
            with open(self.output_file, 'wb') as pkl_file:
                pickle.dump(self.mapper.get_roman_map(), pkl_file, -1)
            self.get_logger().info(f"\033[1;32mROMANMap saved to {self.output_file}!\033[0m")
        self._shutdown_complete = True

    def _wait_for_message(self, topic, msg_type):
        """
        Wait for a message on topic of type msg_type
        """
        subscription = self.create_subscription(msg_type, topic, self._wait_for_message_cb, 1)
        
        self._wait_for_message_msg = None
        while self._wait_for_message_msg is None:
            rclpy.spin_once(self)
        msg = self._wait_for_message_msg
        # subscription.destroy()

        return msg
    
    def _wait_for_message_cb(self, msg):
        self._wait_for_message_msg = msg
        return

def main():

    rclpy.init()
    node = RomanMapNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node._shutdown_requested:
            executor.spin_once(timeout_sec=0.1)
    finally:
        node.up = False
        # This Jazzy executor does not join its workers in shutdown(). Drain
        # them first, while both executor guard conditions and node handles exist.
        executor._executor.shutdown(wait=True)
        executor.shutdown()
        try:
            node.shutdown()
        finally:
            node.destroy_node()
            rclpy.try_shutdown()

if __name__ == "__main__":
    main()
