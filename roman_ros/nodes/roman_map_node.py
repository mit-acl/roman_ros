#!/usr/bin/env python3
"""
ROS node for creating ROMAN open-set object map.

Author: Mason Peterson
"""

import numpy as np
import os
import ros_numpy as rnp
import cv2 as cv
import struct
import pickle
import time

# ROS imports
import rospy
import cv_bridge
import message_filters
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
from roman.map.mapper import Mapper, MapperParams
from roman.object.segment import Segment

# relative
from utils import observation_from_msg, segment_to_msg

class ROMANMapNode():

    def __init__(self):

        # ros params
        self.robot_id = rospy.get_param("~robot_id", 0)
        min_iou = rospy.get_param("~min_iou", 0.25)
        min_sightings = rospy.get_param("~min_sightings", 2)
        max_t_no_sightings = rospy.get_param("~max_t_no_sightings", 0.25)
        mask_downsample_factor = rospy.get_param("~mask_downsample_factor", 8)
        self.visualize = rospy.get_param("~visualize", False)
        self.output_file = rospy.get_param("~output_roman_map", None)
        if self.visualize:
            self.cam_frame_id = rospy.get_param("~cam_frame_id", "camera_link")
            self.map_frame_id = rospy.get_param("~map_frame_id", "map")
            self.odom_base_frame_id = rospy.get_param("~odom_base_frame_id", "base")
            self.viz_num_objs = rospy.get_param("~viz/num_objs", 20)
            self.viz_pts_per_obj = rospy.get_param("~viz/pts_per_obj", 250)
            self.min_viz_dt = rospy.get_param("~viz/min_viz_dt", 2.0)
            self.viz_rotate_img = rospy.get_param("~viz/rotate_img", None)
        if self.output_file is not None and self.output_file != "":
            self.output_file = os.path.expanduser(self.output_file)
            self.pose_history = [] # list of poses
            self.time_history = []
            rospy.loginfo(f"Output file: {self.output_file}")
        elif self.output_file == "":
            self.output_file = None
        print(self.output_file)

        # tracker
        rospy.loginfo("ROMANMapNode waiting for color camera info messages...")
        color_info_msg = rospy.wait_for_message("color/camera_info", sensor_msgs.CameraInfo)
        color_params = CameraParams.from_msg(color_info_msg)
        rospy.loginfo("ROMANMapNode received for color camera info messages...")

        mapper_params = MapperParams(
            camera_params=color_params,
            min_iou=min_iou,
            min_sightings=min_sightings,
            max_t_no_sightings=max_t_no_sightings,
            mask_downsample_factor=mask_downsample_factor,
        )
        self.tracker = Mapper(mapper_params)

        self.setup_ros()

    def setup_ros(self):

        # ros publishers
        self.segments_pub = rospy.Publisher("roman/segment_updates", roman_msgs.Segment, queue_size=5)
        self.pulse_pub = rospy.Publisher("roman/pulse", std_msgs.Empty, queue_size=1)

        # visualization
        if self.visualize:
            self.last_viz_t = -np.inf
            if self.cam_frame_id is not None:
                tf_buffer = tf2_ros.Buffer()
                tf_listener = tf2_ros.TransformListener(tf_buffer)
                odom_msg = rospy.wait_for_message("odom", nav_msgs.Odometry)
                # transform_stamped_msg = tf_buffer.lookup_transform(odom_msg.child_frame_id, self.cam_frame_id, rospy.Time(0))
                transform_stamped_msg = tf_buffer.lookup_transform(self.odom_base_frame_id, self.cam_frame_id, rospy.Time(0))
                self.T_BC = rnp.numpify(transform_stamped_msg.transform)
            else:
                self.T_BC = np.eye(4)
            
            self.bridge = cv_bridge.CvBridge()
            subs = [
                message_filters.Subscriber("odom", nav_msgs.Odometry),
                message_filters.Subscriber("color/image_raw", sensor_msgs.Image),
            ]
            self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=20, slop=.1)
            self.ts.registerCallback(self.viz_cb) # registers incoming messages to callback
            self.annotated_img_pub = rospy.Publisher("roman/annotated_img", sensor_msgs.Image, queue_size=5)
            self.object_points_pub = rospy.Publisher("roman/object_points", sensor_msgs.PointCloud, queue_size=5)

        # ros subscribers
        rospy.Subscriber("roman/observations", roman_msgs.ObservationArray, self.obs_cb)

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Segment Mapper Node setup complete.")
        rospy.loginfo("Waiting for observation.")

    def obs_cb(self, obs_array_msg):
        """
        Triggered by incoming observation messages
        """
        # publish pulse
        rospy.logwarn("Received messages")
        self.pulse_pub.publish(std_msgs.Empty())
        
        if len(obs_array_msg.observations) == 0:
            return
        
        observations = []
        for obs_msg in obs_array_msg.observations:
            observations.append(observation_from_msg(obs_msg))

        t = observations[0].time
        assert all([obs.time == t for obs in observations])

        inactive_ids = [segment.id for segment in self.tracker.inactive_segments]
        self.tracker.update(obs_array_msg.header.stamp.to_sec(), rnp.numpify(obs_array_msg.pose), observations)
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

    def viz_cb(self, odom_msg, img_msg):
        """
        Triggered by incoming odometry and image messages
        """

        # rospy.logwarn("Received messages")
        t = img_msg.header.stamp.to_sec()
        if t - self.last_viz_t < self.min_viz_dt:
            return
        else:
            self.last_viz_t = t

        pose = rnp.numpify(odom_msg.pose.pose)

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        
        segment: Segment
        for i, segment in enumerate(self.tracker.segments + self.tracker.inactive_segments + self.tracker.segment_graveyard):
            # only draw segments seen in the last however many seconds
            if segment.last_seen < t - self.tracker.params.segment_graveyard_time - 10:
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
        points_msg = sensor_msgs.PointCloud()
        points_msg.header = img_msg.header
        points_msg.header.frame_id = self.map_frame_id
        points_msg.points = []
        # points_msg.channels = [sensor_msgs.ChannelFloat32(name=channel, values=[]) for channel in ['r', 'g', 'b']]
        points_msg.channels = [sensor_msgs.ChannelFloat32(name='rgb', values=[])]
        

        most_recently_seen_segments = sorted(
            self.tracker.segments + self.tracker.inactive_segments + self.tracker.segment_graveyard, 
            key=lambda x: x.last_seen if len(x.points) > 10 else 0, reverse=True)[:self.viz_num_objs]

        for segment in most_recently_seen_segments:
            # color
            np.random.seed(segment.id)
            color_unpacked = np.random.rand(3)*256
            color_raw = int(color_unpacked[0]*256**2 + color_unpacked[1]*256 + color_unpacked[2])
            color_packed = struct.unpack('f', struct.pack('i', color_raw))[0]
            
            points = segment.points
            sampled_points = np.random.choice(len(points), min(len(points), 1000), replace=True)
            points = [points[i] for i in sampled_points]
            points_msg.points += [geometry_msgs.Point32(x=p[0], y=p[1], z=p[2]) for p in points]
            points_msg.channels[0].values += [color_packed for _ in points]
        
        self.object_points_pub.publish(points_msg)

        return
    
    def shutdown(self):
        if self.output_file is None:
            print(f"No file to save to.")
        if self.output_file is not None:
            print(f"Saving map to {self.output_file}...")
            time.sleep(5.0)
            pkl_file = open(self.output_file, 'wb')
            pickle.dump(self.tracker.get_roman_map(), pkl_file, -1)
            pkl_file.close()

def main():

    rospy.init_node('segment_tracker_node')
    node = ROMANMapNode()
    rospy.spin()

if __name__ == "__main__":
    main()