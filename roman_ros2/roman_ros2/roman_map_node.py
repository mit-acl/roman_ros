#!/usr/bin/env python3

import numpy as np
import os
import cv2 as cv
import copy
import struct
import pickle
import time
import signal
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

# ROS imports
import rclpy
from rclpy.node import Node
import cv_bridge
import message_filters
import ros2_numpy as rnp
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import tf2_ros
from rclpy.executors import MultiThreadedExecutor
import time

# ROS msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import roman_msgs.msg as roman_msgs
import visualization_msgs.msg as visualization_msgs
from ros_system_monitor_msgs.msg import NodeInfoMsg

# robot_utils
from robotdatapy.camera import CameraParams

# ROMAN
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.mapper import Mapper, MapperParams
from roman.map.map import ROMANMap
from roman.object.segment import Segment
from roman.viz import visualize_segment_on_img

# relative
from roman_ros2.utils import observation_from_msg, descriptor_from_array_msg, segment_to_msg, \
    time_stamp_to_float, TimingFifo

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
                ("visualize", False),                   # whether to publish annotated images
                ("output_roman_map", ""),               # output file to save the map to (must kill node with ctrl-c)
                ("odom_frame_id", "odom"),              # odometry frame id to map objects in
                ("base_flu_frame_id", "base_link"),     # robot coordinate frame with xyz = forward left up
                ("object_ref", "center"),               # bottom_middle or center, what reference point to use for objects
                ("publish_active_segments", False),     # whether to wait till segments are inactive before publishing them
                ("nickname", "roman_map"),              # nickname for this node
                ("use_multiple_cams", False),           # whether mapper will get FastSAM measurements from multiple cameras
                ("timing_window", 10),                  # timing window used for report processing speed
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
        self.publish_active_segments = self.get_parameter("publish_active_segments").value
        self.nickname = self.get_parameter("nickname").value
        self.use_multiple_cams = self.get_parameter("use_multiple_cams").value
        timing_window = self.get_parameter("timing_window").value
        self.timing_fifo = TimingFifo(timing_window)
        config_path = self.get_parameter("config_path").value
        
        assert self.base_flu_frame_id != "", "base_flu_frame_id must be set"

        self.viz_object_map = self.get_parameter("viz_object_map").value
        if self.visualize or self.viz_object_map:
            self.odom_frame_id = self.get_parameter("odom_frame_id").value
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

        self.setup_ros()

    def setup_ros(self):

        # ros publishers
        self.segments_pub = self.create_publisher(roman_msgs.Segment, "roman/segment_updates", qos_profile=10)
        self.pulse_pub = self.create_publisher(std_msgs.Empty, "roman/pulse", qos_profile=10)

        # ros tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # separate callback groups so obs_cb and viz_cb can run in parallel
        self.obs_cb_group = MutuallyExclusiveCallbackGroup()
        self.viz_cb_group = MutuallyExclusiveCallbackGroup()

        # ros subscribers
        self.create_subscription(roman_msgs.ObservationArray, "roman/observations", self.obs_cb, 10,
                                 callback_group=self.obs_cb_group)

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

        frame_descriptor = descriptor_from_array_msg(obs_array_msg.frame_descriptor)

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
        
    def publish_segment(self, segment: Segment):
        if self.object_ref == 'bottom_middle':
            segment.set_center_ref('bottom_middle')
        self.segments_pub.publish(segment_to_msg(self.robot_id, segment))

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

        # Point cloud publishing
        # points_msg = sensor_msgs.PointCloud()
        # points_msg.header = img_msg.header
        # points_msg.header.frame_id = self.odom_frame_id
        # points_msg.points = []
        # # points_msg.channels = [sensor_msgs.ChannelFloat32(name=channel, values=[]) for channel in ['r', 'g', 'b']]
        # points_msg.channels = [sensor_msgs.ChannelFloat32(name='rgb', values=[])]
        

        # most_recently_seen_segments = sorted(
        #     self.mapper.segments + self.mapper.inactive_segments + self.mapper.segment_graveyard, 
        #     key=lambda x: x.last_seen if len(x.points) > 10 else 0, reverse=True)[:self.viz_num_objs]

        # for segment in most_recently_seen_segments:
        #     # color
        #     np.random.seed(segment.id)
        #     color_unpacked = np.random.rand(3)*256
        #     color_raw = int(color_unpacked[0]*256**2 + color_unpacked[1]*256 + color_unpacked[2])
        #     color_packed = struct.unpack('f', struct.pack('i', color_raw))[0]
            
        #     points = segment.points
        #     sampled_points = np.random.choice(len(points), min(len(points), 1000), replace=True)
        #     points = [points[i] for i in sampled_points]
        #     points_msg.points += [geometry_msgs.Point32(x=p[0], y=p[1], z=p[2]) for p in points]
        #     points_msg.channels[0].values += [color_packed for _ in points]
        
        # self.object_points_pub.publish(points_msg)

        return
    
    def shutdown(self):
        if self.output_file is None:
            print(f"No file to save to.")
        if self.output_file is not None:
            self.up = False
            print(f"Saving map to {self.output_file}...")
            time.sleep(1.0)
            self.mapper.make_pickle_compatible()
            pkl_file = open(self.output_file, 'wb')
            pickle.dump(self.mapper.get_roman_map(), pkl_file, -1)
            pkl_file.close()
        self.destroy_node()

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

    # signal.signal(signal.SIGINT, node.shutdown)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()