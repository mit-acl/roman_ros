#!/usr/bin/env python3

import numpy as np
import os
import ros_numpy as rnp
import cv2 as cv

# ROS imports
import rospy
import cv_bridge
import message_filters

# ROS msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import segment_slam_msgs.msg as segment_slam_msgs

# robot_utils
from robot_utils.camera import CameraParams

# segment_track
from segment_track.fastsam_wrapper import FastSAMWrapper
from segment_track.tracker import Tracker
from segment_track.segment import Segment

# relative
from utils import observation_from_msg, segment_to_msg

class SegmentTrackerNode():

    def __init__(self):

        # ros params
        self.robot_id = rospy.get_param("~robot_id", 0)
        pixel_std_dev = rospy.get_param("~pixel_std_dev", 20.0)
        min_iou = rospy.get_param("~min_iou", 0.25)
        min_sightings = rospy.get_param("~min_sightings", 5)
        max_t_no_sightings = rospy.get_param("~max_t_no_sightings", 0.5)
        mask_downsample_factor = rospy.get_param("~mask_downsample_factor", 8)
        self.visualize = rospy.get_param("~visualize", False)
        if self.visualize:
            self.T_BC = np.array(rospy.get_param("~T_BC", np.eye(4).tolist())).reshape((4, 4))
    

        # tracker
        rospy.loginfo("Waiting for color camera info messages...")
        color_info_msg = rospy.wait_for_message("color/camera_info", sensor_msgs.CameraInfo)
        color_params = CameraParams.from_msg(color_info_msg)

        self.tracker = Tracker(
            camera_params=color_params,
            pixel_std_dev=pixel_std_dev,
            min_iou=min_iou,
            min_sightings=min_sightings,
            max_t_no_sightings=max_t_no_sightings,
            mask_downsample_factor=mask_downsample_factor,
        )

        self.setup_ros()

    def setup_ros(self):
        
        # ros subscribers
        rospy.Subscriber("segment_track/observations", segment_slam_msgs.ObservationArray, self.obs_cb)

        # ros publishers
        self.segments_pub = rospy.Publisher("segment_track/segment_updates", segment_slam_msgs.Segment, queue_size=5)

        # visualization
        if self.visualize:
            self.bridge = cv_bridge.CvBridge()
            subs = [
                message_filters.Subscriber("odom", nav_msgs.Odometry),
                message_filters.Subscriber("color/image_raw", sensor_msgs.Image),
            ]
            self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=20, slop=.1)
            self.ts.registerCallback(self.viz_cb) # registers incoming messages to callback
            self.annotated_img_pub = rospy.Publisher("segment_track/annotated_img", sensor_msgs.Image, queue_size=5)

        rospy.loginfo("Segment Tracker Node setup complete.")

    def obs_cb(self, obs_array_msg):
        """
        Triggered by incoming observation messages
        """
        if len(obs_array_msg.observations) == 0:
            return
        
        observations = []
        for obs_msg in obs_array_msg.observations:
            observations.append(observation_from_msg(obs_msg))

        t = observations[0].time
        assert all([obs.time == t for obs in observations])

        graveyard_ids = [segment.id for segment in self.tracker.segment_graveyard]
        self.tracker.update(obs_array_msg.header.stamp.to_sec(), rnp.numpify(obs_array_msg.pose), observations)
        updated_graveyard_ids = [segment.id for segment in self.tracker.segment_graveyard]
        new_graveyard_ids = [seg_id for seg_id in updated_graveyard_ids if seg_id not in graveyard_ids]

        # publish segments
        segment: Segment
        for segment in self.tracker.segment_graveyard:
            if segment.last_seen == t or segment.id in new_graveyard_ids:
                self.segments_pub.publish(segment_to_msg(self.robot_id, segment))

    def viz_cb(self, odom_msg, img_msg):
        """
        Triggered by incoming odometry and image messages
        """

        rospy.logwarn("Received messages")
        t = img_msg.header.stamp.to_sec()
        pose = rnp.numpify(odom_msg.pose.pose)

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        
        segment: Segment
        for i, segment in enumerate(self.tracker.segments + self.tracker.segment_graveyard):
            # only draw segments seen in the last however many seconds
            if segment.last_seen < t - 50:
                continue
            try:
                bbox = segment.reprojected_bbox(pose) # @ self.T_BC)
            except:
                continue
            if bbox is None:
                continue
            if i < len(self.tracker.segments):
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            img = cv.rectangle(img, np.array([bbox[0][0], bbox[0][1]]).astype(np.int32), 
                        np.array([bbox[1][0], bbox[1][1]]).astype(np.int32), color=color, thickness=2)
            img = cv.putText(img, str(segment.id), (np.array(bbox[0]) + np.array([10., 10.])).astype(np.int32), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.annotated_img_pub.publish(img_msg)

        return

def main():

    rospy.init_node('segment_tracker_node')
    node = SegmentTrackerNode()
    rospy.spin()

if __name__ == "__main__":
    main()

