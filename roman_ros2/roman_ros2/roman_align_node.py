#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass

# ROS imports
import rclpy
from rclpy.node import Node
import ros2_numpy as rnp

# ROS msgs
import roman_msgs.msg as roman_msgs
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as visualization_msgs

# Custom modules
from robotdatapy.transform import transform_to_xyzrpy

from roman.object.segment import Segment, SegmentMinimalData
from roman.align.roman import ROMAN, ROMANParams
from roman.utils import transform_rm_roll_pitch

# Local imports
from roman_ros2.utils import msg_to_segment, time_stamp_to_float


class RecentROMANMap():

    def __init__(self, time_pass_thresh=30.0):
        self.time = -np.inf
        self.segments = dict()
        self.time_pass_thresh = time_pass_thresh

    def update(self, segment: Segment):
        self.time = segment.last_seen
        
        if segment.id in self.segments:
            segment.first_seen = self.segments[segment.id].first_seen
        else:
            segment.first_seen = segment.last_seen

        # update segment
        self.segments[segment.id] = segment

        # remove segments that have not been seen for a while
        to_delete = []
        for seg_id, seg in self.segments.items():
            if self.time - seg.last_seen > self.time_pass_thresh:
                to_delete.append(seg_id)

        for seg_id in to_delete:
            del self.segments[seg_id]
            
    @property
    def segment_list(self):
        return list(self.segments.values())


class ROMANAlignNode(Node):

    def __init__(self):
        
        super().__init__('roman_align_node')

        # ros params
        self.declare_parameters(
            namespace='',
            parameters=[
                ("robot1", None),
                ("robot2", None),
                ("time_pass_thresh", 60.0),
                ("align_dt", 0.5),
                ("clipper/sigma", 0.3),
                ("clipper/epsilon", 0.5),
                ("clipper/mindist", 0.1),
                ("clipper/volume_epsilon", 0.0),
                ("clipper/min_associations", 4)
            ]
        )
        
        self.robot1 = self.get_parameter("robot1").value
        self.robot2 = self.get_parameter("robot2").value
        
        self.declare_parameter("frame1", f"{self.robot1}")
        self.declare_parameter("frame2", f"{self.robot2}")
        
        self.frame1 = self.get_parameter("frame1").value
        self.frame2 = self.get_parameter("frame2").value
        time_pass_thresh = self.get_parameter("time_pass_thresh").value
        self.align_dt = self.get_parameter("align_dt").value
        clipper_sigma = self.get_parameter("clipper/sigma").value
        clipper_epsilon = self.get_parameter("clipper/epsilon").value
        clipper_mindist = self.get_parameter("clipper/mindist").value
        clipper_volume_epsilon = self.get_parameter("clipper/volume_epsilon").value
        self.clipper_min_associations = self.get_parameter("clipper/min_associations").value
        
        # internal variables
        self.seg_maps = {robot: RecentROMANMap(time_pass_thresh=time_pass_thresh) for robot in [self.robot1, self.robot2]}
        roman_registration_params = ROMANParams(
            sigma=clipper_sigma,
            epsilon=clipper_epsilon,
            mindist=clipper_mindist,
            gravity=True,
            volume=True,
            pca=True,
            epsilon_shape=clipper_volume_epsilon,
        )
        self.roman_registration = ROMAN(roman_registration_params)
        
        self.setup_ros()
        
    def setup_ros(self):
        
        # ros subscribers
        self.segment_subs = [
            self.create_subscription(roman_msgs.Segment, f"/{self.robot1}/roman/segment_updates", 
                                     lambda msg: self.seg_cb(msg, self.robot1), 20),
            self.create_subscription(roman_msgs.Segment, f"/{self.robot2}/roman/segment_updates", 
                                      lambda msg: self.seg_cb(msg, self.robot2), 20),
        ]
        
        self.timer = self.create_timer(self.align_dt, self.timer_cb)

        # ros publishers
        self.transform_pub = self.create_publisher(geometry_msgs.TransformStamped, 
                                    f"/{self.robot1}/roman/frame_align/{self.robot2}", 10)
        # self.map_association_pub = self.create_publisher(visualization_msgs.MarkerArray, 
        #                             f"/{self.robot1}/roman/frame_align/{self.robot2}/markers", 10)

    def seg_cb(self, seg_msg: roman_msgs.Segment, robot: str):
        """
        Callback function for segment messages.
        """
        segment = msg_to_segment(seg_msg)
        self.seg_maps[robot].update(segment)

    def timer_cb(self):
        """
        Performs object map registration when timer is triggered.

        Args:
            timer (_type_): _description_
        """
        # self.pub_map(self.robot1)
        # self.pub_map(self.robot2)
        map1 = self.seg_maps[self.robot1].segment_list
        map2 = self.seg_maps[self.robot2].segment_list
        
        print(f"Map1: {len(map1)} segments, Map2: {len(map2)} segments.")
        if len(map1) < self.clipper_min_associations or len(map2) < self.clipper_min_associations:
            print("Not enough segments to align.")
            # print(f"Map1: {len(map1)} segments, Map2: {len(map2)} segments.")
            return
        
        # compute object-to-object associations
        associations = self.roman_registration.register(map1, map2)
        print(f"Found {len(associations)} associations.")
        
        if len(associations) < self.clipper_min_associations:
            return

        # compute frame alignment (the pose of frame2 with respect to frame1)
        T_frame1_frame2 = self.roman_registration.T_align(map1, map2, associations)
        T_frame1_frame2 = transform_rm_roll_pitch(T_frame1_frame2)
        xyzrpy = transform_to_xyzrpy(T_frame1_frame2, degrees=True)
        print(f"T^{self.frame1}_{self.frame2} = ")
        print(f"x: {xyzrpy[0]:.2f}, y: {xyzrpy[1]:.2f}, z: {xyzrpy[2]:.2f}, ")
        print(f"roll: {xyzrpy[3]:.2f}, pitch: {xyzrpy[4]:.2f}, yaw: {xyzrpy[5]:.2f}")

        # publish transform
        tf_msg = geometry_msgs.TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.frame1
        tf_msg.child_frame_id = self.frame2
        tf_msg.transform = rnp.msgify(geometry_msgs.Transform, T_frame1_frame2)
        self.transform_pub.publish(tf_msg)

        return
    
    def pub_map(self, robot):
        pass
        # # publish map array
        # map_array = geometry_msgs.PoseArray()
        # map_array.header.stamp = rospy.Time.now()
        # map_array.header.frame_id = "world"
        # map_array.poses = []
        # for obj in self.seg_maps[robot].as_object_list():
        #     pose = geometry_msgs.Pose()
        #     pose.position.x = obj.center[0]
        #     pose.position.y = obj.center[1]
        #     pose.position.z = obj.center[2]
        #     map_array.poses.append(pose)
        
        # self.map_array_pub[robot].publish(map_array)  
        
def main():

    rclpy.init()
    node = ROMANAlignNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

