#!/usr/bin/env python3

import numpy as np
import os
import cv2 as cv
import struct
import pickle
import time

# ROS imports
import rclpy
from rclpy.node import Node
import cv_bridge
import message_filters
import ros2_numpy as rnp
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile
import tf2_ros

# ROS msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import roman_msgs.msg as roman_msgs

# robot_utils
from robotdatapy.camera import CameraParams

# ROMAN
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.tracker import Tracker
from roman.object.segment import Segment

# relative
from roman_ros2.utils import observation_from_msg, segment_to_msg, time_stamp_to_float

class RomanMapNode(Node):

    def __init__(self):
        super().__init__('roman_map_node')

        # ros params
        self.declare_parameters(
            namespace='',
            parameters=[
                ("robot_id", 0),
                ("min_iou", 0.25),
                ("min_sightings", 2),
                ("max_t_no_sightings", 0.25),
                ("mask_downsample_factor", 8),
                ("visualize", False),
                ("output_map", None),
                ("cam_frame_id", "camera_link"),
                ("map_frame_id", "map"),
                ("viz/num_objs", 20),
                ("viz/pts_per_obj", 250),
                ("viz/min_viz_dt", 2.0),
                ("viz/rotate_img", None),
                ("viz/pointcloud", False)
            ]
        )

        self.robot_id = self.get_parameter("robot_id").value
        min_iou = self.get_parameter("min_iou").value
        min_sightings = self.get_parameter("min_sightings").value
        max_t_no_sightings = self.get_parameter("max_t_no_sightings").value
        mask_downsample_factor = self.get_parameter("mask_downsample_factor").value
        self.visualize = self.get_parameter("visualize").value
        self.output_file = self.get_parameter("output_map").value
        if self.visualize:
            self.cam_frame_id = self.get_parameter("cam_frame_id").value
            self.map_frame_id = self.get_parameter("map_frame_id").value
            self.viz_num_objs = self.get_parameter("viz/num_objs").value
            self.viz_pts_per_obj = self.get_parameter("viz/pts_per_obj").value
            self.min_viz_dt = self.get_parameter("viz/min_viz_dt").value
            self.viz_rotate_img = self.get_parameter("viz/rotate_img").value
            self.viz_pointcloud = self.get_parameter("viz/pointcloud").value
        if self.output_file is not None and self.output_file != "":
            self.output_file = os.path.expanduser(self.output_file)
            self.pose_history = []
            self.time_history = []
            self.get_logger().info(f"Output file: {self.output_file}")
        elif self.output_file == "":
            self.output_file = None

        # mapper
        self.get_logger().info("RomanMapNode setting up mapping...")
        self.get_logger().info("RomanMapNode waiting for color camera info messages...")
        color_info_msg = self._wait_for_message("color/camera_info", sensor_msgs.CameraInfo)
        color_params = CameraParams.from_msg(color_info_msg)
        self.get_logger().info("RomanMapNode received for color camera info messages...")

        self.tracker = Tracker(
            camera_params=color_params,
            min_iou=min_iou,
            min_sightings=min_sightings,
            max_t_no_sightings=max_t_no_sightings,
            mask_downsample_factor=mask_downsample_factor,
        )

        self.setup_ros()

    def setup_ros(self):
        
        # ros subscribers
        self.create_subscription(roman_msgs.ObservationArray, "roman/observations", self.obs_cb, 10)

        # ros publishers
        self.segments_pub = self.create_publisher(roman_msgs.Segment, "roman/segment_updates", qos_profile=10)
        self.pulse_pub = self.create_publisher(std_msgs.Empty, "roman/pulse", qos_profile=10)

        # visualization
        if self.visualize:
            self.last_viz_t = -np.inf
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            
            self.create_subscription(sensor_msgs.Image, "color/image_raw", self.viz_cb, 10)
            self.bridge = cv_bridge.CvBridge()
            self.annotated_img_pub = self.create_publisher(sensor_msgs.Image, "roman/annotated_img", qos_profile=10)
            self.object_points_pub = self.create_publisher(sensor_msgs.PointCloud, "roman/object_points", qos_profile=10)

        self.get_logger().info("ROMAN Map Node setup complete.")
        self.get_logger().info("Waiting for observation.")

    def obs_cb(self, obs_array_msg):
        """
        Triggered by incoming observation messages
        """
        # publish pulse
        self.get_logger().info("Received messages")
        self.pulse_pub.publish(std_msgs.Empty())
        
        if len(obs_array_msg.observations) == 0:
            return
        
        observations = []
        for obs_msg in obs_array_msg.observations:
            observations.append(observation_from_msg(obs_msg))

        t = observations[0].time
        assert all([obs.time == t for obs in observations])

        inactive_ids = [segment.id for segment in self.tracker.inactive_segments]
        self.tracker.update(time_stamp_to_float(obs_array_msg.header.stamp), rnp.numpify(obs_array_msg.pose), observations)
        updated_inactive_ids = [segment.id for segment in self.tracker.inactive_segments]
        new_inactive_ids = [seg_id for seg_id in updated_inactive_ids if seg_id not in inactive_ids]

        # publish segments
        segment: Segment
        for segment in self.tracker.inactive_segments:
            if segment.last_seen == t or segment.id in new_inactive_ids:
                self.segments_pub.publish(segment_to_msg(self.robot_id, segment))
        
        if self.output_file is not None:
            self.pose_history.append(rnp.numpify(obs_array_msg.pose_flu))
            self.time_history.append(t)

    def viz_cb(self, img_msg):
        """
        Triggered by incoming odometry and image messages
        """

        # rospy.logwarn("Received messages")
        t = time_stamp_to_float(img_msg.header.stamp)
        if t - self.last_viz_t < self.min_viz_dt:
            return
        else:
            self.last_viz_t = t

        try:
            transform_stamped_msg = self.tf_buffer.lookup_transform(self.map_frame_id, self.cam_frame_id, img_msg.header.stamp, rclpy.duration.Duration(seconds=2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warning("tf lookup failed")
            print(ex)
            return

        pose = rnp.numpify(transform_stamped_msg.transform).astype(np.float64)

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        
        segment: Segment
        for i, segment in enumerate(self.tracker.segments + self.tracker.inactive_segments + self.tracker.segment_graveyard):
            # only draw segments seen in the last however many seconds
            if segment.last_seen < t - self.tracker.segment_graveyard_time - 10:
                continue
            try:
                bbox = segment.reprojected_bbox(pose @ self.T_BC)
            except:
                continue
            if bbox is None:
                continue
            if i < len(self.tracker.segments):
                color = (0, 255, 0)
            elif i < len(self.tracker.segments) + len(self.tracker.inactive_segments):
                color = (255, 0, 0)
            else:
                color = (180, 0, 180)
            img = cv.rectangle(img, np.array([bbox[0][0], bbox[0][1]]).astype(np.int32), 
                        np.array([bbox[1][0], bbox[1][1]]).astype(np.int32), color=color, thickness=2)
            img = cv.putText(img, str(segment.id), (np.array(bbox[0]) + np.array([10., 10.])).astype(np.int32), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        if self.viz_rotate_img is not None:
            if self.viz_rotate_img == "CW":
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            elif self.viz_rotate_img == "CCW":
                img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            elif self.viz_rotate_img == "180":
                img = cv.rotate(img, cv.ROTATE_180)
        
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.annotated_img_pub.publish(img_msg)

        # Point cloud publishing
        # points_msg = sensor_msgs.PointCloud()
        # points_msg.header = img_msg.header
        # points_msg.header.frame_id = self.map_frame_id
        # points_msg.points = []
        # # points_msg.channels = [sensor_msgs.ChannelFloat32(name=channel, values=[]) for channel in ['r', 'g', 'b']]
        # points_msg.channels = [sensor_msgs.ChannelFloat32(name='rgb', values=[])]
        

        # most_recently_seen_segments = sorted(
        #     self.tracker.segments + self.tracker.inactive_segments + self.tracker.segment_graveyard, 
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
            print(f"Saving map to {self.output_file}...")
            time.sleep(5.0)
            self.tracker.make_pickle_compatible()
            pkl_file = open(self.output_file, 'wb')
            pickle.dump([self.tracker, self.pose_history, self.time_history], pkl_file, -1)
            pkl_file.close()

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
    rclpy.spin(node)

    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()