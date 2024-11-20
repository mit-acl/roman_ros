#!/usr/bin/env python3
import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
import struct
import open3d as o3d

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
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import roman_msgs.msg as roman_msgs

# robot_utils
from robotdatapy.camera import CameraParams

# ROMAN
from roman.map.fastsam_wrapper import FastSAMWrapper

# relative
from roman_ros2.utils import observation_to_msg

class FastSAMNode(Node):

    def __init__(self):
        super().__init__('fastsam_node')
        
        # internal variables
        self.bridge = cv_bridge.CvBridge()

        # required ros parameters
        self.declare_parameter("fastsam_weights")
        self.declare_parameter("fastsam_device")

        # ros params
        self.declare_parameters(
            namespace='',
            parameters=[
                ("cam_frame_id", None),
                ("map_frame_id", "map"),
                ("odom_base_frame_id", "base"),
                ("fastsam_imgsz", 256),
                ("fastsam_mask_downsample", 8),
                ("fastsam_rotate_img", None),
                ("fastsam_ignore_people", True),
                ("fastsam_allow_edges", True),
                ("fastsam_min_area_div", 30),
                ("fastsam_max_area_div", 3),
                ("fastsam_erosion_size", 3),
                ("fastsam_min_dt", 0.1),
                ("fastsam_viz", False),
                ("fastsam_max_depth", 8.0),
            ]
        )

        self.cam_frame_id = self.get_parameter("cam_frame_id").value
        self.map_frame_id = self.get_parameter("map_frame_id").value
        self.odom_base_frame_id = self.get_parameter("odom_base_frame_id").value
        
        fastsam_weights_path = self.get_parameter("fastsam_weights").value
        fastsam_imgsz = self.get_parameter("fastsam_imgsz").value
        fastsam_device = self.get_parameter("fastsam_device").value
        fastsam_mask_downsample = self.get_parameter("fastsam_mask_downsample").value
        fastsam_rotate_img = self.get_parameter("fastsam_rotate_img").value

        fastsam_ignore_people = self.get_parameter("fastsam_ignore_people").value
        fastsam_allow_edges = self.get_parameter("fastsam_allow_edges").value
        fastsam_min_area_div = self.get_parameter("fastsam_min_area_div").value
        fastsam_max_area_div = self.get_parameter("fastsam_max_area_div").value
        fastsam_erosion_size = self.get_parameter("fastsam_erosion_size").value
        fastsam_max_depth = self.get_parameter("fastsam_max_depth").value
        self.min_dt = self.get_parameter("fastsam_min_dt").value

        self.visualize = self.get_parameter("fastsam_viz").value
        self.last_t = -np.inf

        assert fastsam_weights_path is not None, "fastsam_weights parameter must be set"
        assert fastsam_device is not None, "fastsam_device parameter must be set"

        # fastsam wrapper
        self.fastsam = FastSAMWrapper(
            weights=os.path.expanduser(os.path.expandvars(fastsam_weights_path)),
            imgsz=fastsam_imgsz,
            device=fastsam_device,
            mask_downsample_factor=fastsam_mask_downsample,
            rotate_img=fastsam_rotate_img
        )

        # FastSAM set up after camera info can be retrieved
        self.get_logger().info("Waiting for depth camera info messages...")
        depth_info_msg = self._wait_for_message("depth/camera_info", sensor_msgs.CameraInfo)
        self.get_logger().info("Received for depth camera info messages...")
        self.get_logger().info("Waiting for color camera info messages...")
        color_info_msg = self._wait_for_message("color/camera_info", sensor_msgs.CameraInfo)
        self.get_logger().info("Received for color camera info messages...")
        self.depth_params = CameraParams.from_msg(depth_info_msg)
        color_params = CameraParams.from_msg(color_info_msg)
        
        self.fastsam.setup_rgbd_params(
            depth_cam_params=self.depth_params, 
            max_depth=fastsam_max_depth,
            depth_scale=1e3,
            voxel_size=0.05,
            erosion_size=fastsam_erosion_size
        )
        img_area = self.depth_params.width * self.depth_params.height
        self.fastsam.setup_filtering(
            ignore_labels=['person'] if fastsam_ignore_people else [],
            yolo_det_img_size=(128, 128) if fastsam_ignore_people else None,
            allow_tblr_edges=[True, True, True, True] if fastsam_allow_edges else [False, False, False, False],
            area_bounds=[img_area / (fastsam_min_area_div**2), img_area / (fastsam_max_area_div**2)]
        )

        self.setup_ros()

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

    def setup_ros(self):

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ros subscribers
        subs = [
            message_filters.Subscriber(self, sensor_msgs.Image, "color/image_raw"),
            message_filters.Subscriber(self, sensor_msgs.Image, "depth/image_raw"),
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.1)
        self.ts.registerCallback(self.cb) # registers incoming messages to callback

        # ros publishers
        self.obs_pub = self.create_publisher(roman_msgs.ObservationArray, "roman/observations", qos_profile=QoSProfile(depth=10))

        if self.visualize:
            self.ptcld_pub = self.create_publisher(sensor_msgs.PointCloud, "roman/observations/ptcld")

        self.get_logger().info("FastSAM node setup complete.")

    def cb(self, *msgs):
        """
        This function gets called every time synchronized odometry, image message, and 
        depth image message are received.
        """
        
        self.get_logger().info("Received messages")
        img_msg, depth_msg = msgs

        # check that enough time has passed since last observation (to not overwhelm GPU)
        t = rclpy.time.Time.from_msg(img_msg.header.stamp).nanoseconds * 1e-9
        if t - self.last_t < self.min_dt:
            return
        else:
            self.last_t = t

        try:
            # self.tf_buffer.waitForTransform(self.map_frame_id, self.cam_frame_id, img_msg.header.stamp, rospy.Duration(0.5))
            transform_stamped_msg = self.tf_buffer.lookup_transform(self.map_frame_id, self.cam_frame_id, img_msg.header.stamp, rclpy.duration.Duration(seconds=2.0))
            flu_transformed_stamped_msg = self.tf_buffer.lookup_transform(self.map_frame_id, self.odom_base_frame_id, img_msg.header.stamp, rclpy.duration.Duration(seconds=0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warning("tf lookup failed")
            print(ex)
            return       
         
        pose = rnp.numpify(transform_stamped_msg.transform).astype(np.float64)
        pose_flu = rnp.numpify(flu_transformed_stamped_msg.transform).astype(np.float64)

        

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)

        observations = self.fastsam.run(t, pose, img, img_depth=depth)

        observation_msgs = [observation_to_msg(obs) for obs in observations]
        
        observation_array = roman_msgs.ObservationArray(
            header=img_msg.header,
            pose=rnp.msgify(geometry_msgs.Pose, pose),
            pose_flu=rnp.msgify(geometry_msgs.Pose, pose_flu),
            observations=observation_msgs
        )
        self.obs_pub.publish(observation_array)

        # if self.visualize:
        #     self.pub_ptclds(observations, img_msg.header, depth)

        return
    
    # def pub_ptclds(self, observations, header, depth):
    #     points_msg = sensor_msgs.PointCloud()
    #     points_msg.header = header
    #     points_msg.header.frame_id = self.cam_frame_id
    #     points_msg.points = []
    #     points_msg.channels = [sensor_msgs.ChannelFloat32(name='rgb', values=[])]

    #     for i, obs in enumerate(observations):
    #         # color
    #         color_unpacked = np.random.rand(3)*256
    #         color_raw = int(color_unpacked[0]*256**2 + color_unpacked[1]*256 + color_unpacked[2])
    #         color_packed = struct.unpack('f', struct.pack('i', color_raw))[0]
            
    #         points = obs.point_cloud
    #         sampled_points = np.random.choice(len(points), min(len(points), 100), replace=True)
    #         points = [points[i] for i in sampled_points]
    #         points_msg.points += [geometry_msgs.Point32(x=p[0], y=p[1], z=p[2]) for p in points]
    #         points_msg.channels[0].values += [color_packed for _ in points]
        
    #     self.ptcld_pub.publish(points_msg)

def main():

    rclpy.init()
    node = FastSAMNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

