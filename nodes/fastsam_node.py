#!/usr/bin/env python3

import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
import struct
import open3d as o3d

# ROS imports
import ros_numpy as rnp
import rospy
import cv_bridge
import message_filters
import tf2_ros

# ROS msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import segment_slam_msgs.msg as segment_slam_msgs

# robot_utils
from robot_utils.camera import CameraParams

# segment_track
from segment_track.fastsam_wrapper import FastSAMWrapper

# relative
from utils import observation_to_msg

class FastSAMNode():

    def __init__(self):
        
        # internal variables
        self.bridge = cv_bridge.CvBridge()

        # ros params
        # self.T_BC = np.array(rospy.get_param("~T_BC", np.eye(4).tolist())).reshape((4, 4)).astype(np.float64)
        self.cam_frame_id = rospy.get_param("~cam_frame_id", None)
        self.map_frame_id = rospy.get_param("~map_frame_id", "map")
        
        fastsam_weights_path = rospy.get_param("~fastsam_weights")
        fastsam_imgsz = rospy.get_param("~fastsam_imgsz")
        fastsam_device = rospy.get_param("~fastsam_device")
        fastsam_mask_downsample = rospy.get_param("~fastsam_mask_downsample")

        fastsam_ignore_people = rospy.get_param("~fastsam_ignore_people")
        fastsam_allow_edges = rospy.get_param("~fastsam_allow_edges")
        fastsam_min_area_div = rospy.get_param("~fastsam_min_area_div")
        fastsam_max_area_div = rospy.get_param("~fastsam_max_area_div")
        fastsam_erosion_size = rospy.get_param("~fastsam_erosion_size")
        self.min_dt = rospy.get_param("~fastsam_min_dt", 0.1)
        self.last_t = -np.inf

        self.visualize = rospy.get_param("~fastsam_viz", False)

        # fastsam wrapper
        self.fastsam = FastSAMWrapper(
            weights=os.path.expanduser(os.path.expandvars(fastsam_weights_path)),
            imgsz=fastsam_imgsz,
            device=fastsam_device,
            mask_downsample_factor=fastsam_mask_downsample
        )

        # FastSAM set up after camera info can be retrieved
        rospy.loginfo("Waiting for depth camera info messages...")
        depth_info_msg = rospy.wait_for_message("depth/camera_info", sensor_msgs.CameraInfo)
        rospy.loginfo("Received for depth camera info messages...")
        rospy.loginfo("Waiting for color camera info messages...")
        color_info_msg = rospy.wait_for_message("color/camera_info", sensor_msgs.CameraInfo)
        rospy.loginfo("Received for color camera info messages...")
        self.depth_params = CameraParams.from_msg(depth_info_msg)
        color_params = CameraParams.from_msg(color_info_msg)
        
        self.fastsam.setup_rgbd_params(
            depth_cam_params=self.depth_params, 
            max_depth=8,
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

    def setup_ros(self):

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # ros subscribers
        subs = [
            message_filters.Subscriber("color/image_raw", sensor_msgs.Image),
            message_filters.Subscriber("depth/image_raw", sensor_msgs.Image),
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=.1)
        self.ts.registerCallback(self.cb) # registers incoming messages to callback

        # ros publishers
        self.obs_pub = rospy.Publisher("segment_track/observations", segment_slam_msgs.ObservationArray, queue_size=5)

        if self.visualize:
            self.ptcld_pub = rospy.Publisher("segment_track/observations/ptcld", sensor_msgs.PointCloud, queue_size=5)

        rospy.loginfo("FastSAM node setup complete.")

    def cb(self, *msgs):
        """
        This function gets called every time synchronized odometry, image message, and 
        depth image message are received.
        """
        
        rospy.loginfo("Received messages")
        img_msg, depth_msg = msgs
        try:
            transform_stamped_msg = self.tf_buffer.lookup_transform(self.map_frame_id, self.cam_frame_id, img_msg.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("tf lookup failed")
            return
        t = img_msg.header.stamp.to_sec()
        
        pose = rnp.numpify(transform_stamped_msg.transform).astype(np.float64)

        # check that enough time has passed since last observation (to not overwhelm GPU)
        if t - self.last_t < self.min_dt:
            return
        else:
            self.last_t = t

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)

        observations = self.fastsam.run(t, pose, img, img_depth=depth)

        observation_msgs = [observation_to_msg(obs) for obs in observations]
        
        observation_array = segment_slam_msgs.ObservationArray(
            header=img_msg.header,
            pose=rnp.msgify(geometry_msgs.Pose, pose),
            observations=observation_msgs
        )
        self.obs_pub.publish(observation_array)

        if self.visualize:
            self.pub_ptclds(observations, img_msg.header, depth)

        return
    
    def pub_ptclds(self, observations, header, depth):
        points_msg = sensor_msgs.PointCloud()
        points_msg.header = header
        points_msg.header.frame_id = self.cam_frame_id
        points_msg.points = []
        points_msg.channels = [sensor_msgs.ChannelFloat32(name='rgb', values=[])]

        for i, obs in enumerate(observations):
            # color
            color_unpacked = np.random.rand(3)*256
            color_raw = int(color_unpacked[0]*256**2 + color_unpacked[1]*256 + color_unpacked[2])
            color_packed = struct.unpack('f', struct.pack('i', color_raw))[0]
            
            points = obs.point_cloud
            sampled_points = np.random.choice(len(points), min(len(points), 100), replace=True)
            points = [points[i] for i in sampled_points]
            points_msg.points += [geometry_msgs.Point32(x=p[0], y=p[1], z=p[2]) for p in points]
            points_msg.channels[0].values += [color_packed for _ in points]
        
        self.ptcld_pub.publish(points_msg)

def main():

    rospy.init_node('fastsam_node')
    node = FastSAMNode()
    rospy.spin()

if __name__ == "__main__":
    main()

