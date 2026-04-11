#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass
from typing import List
from copy import deepcopy
from pathlib import Path
import cv2 as cv

# ROS imports
import rclpy
from rclpy.node import Node
import ros2_numpy as rnp
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from rclpy.qos import QoSProfile
import cv_bridge

# ROS msgs
import roman_msgs.msg as roman_msgs
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as visualization_msgs
import sensor_msgs.msg as sensor_msgs 
import std_msgs.msg as std_msgs
from ros_system_monitor_msgs.msg import NodeInfoMsg
from pose_graph_tools_msgs.msg import PoseGraph, PoseGraphEdge

# Custom modules
from robotdatapy.transform import transform_to_xyzrpy
from robotdatapy.data import PoseData

from roman.object.segment import Segment, SegmentMinimalData
from roman.align.roman_registration import ROMANRegistration, ROMANParams
from roman.align.object_registration import InsufficientAssociationsException
from roman.utils import expandvars_recursive

from roman.params.submap_align_params import SubmapAlignParams
from roman.utils import transform_rm_roll_pitch
from roman.map.map import Submap, ROMANMap, submaps_from_roman_map, SubmapParams

# Local imports
from roman_ros2.roman_loop_closure_node import ROMANLoopClosureNodeBaseClass
from roman_ros2.utils import msg_to_segment, float_to_ros_time, lc_to_pose_graph_msg, \
    lc_to_msg, time_stamp_to_float


class LCVizNode(ROMANLoopClosureNodeBaseClass):

    def __init__(self):
        
        super().__init__('lc_viz_node')

        # additional parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ("viz_robot_ids", [-1]),
                ("aligned_traj_viz", True),
                ("lc_marker_viz", True),
                ("output_dir", "~/.roman_ros2/loop_closures"),
                ("img_dim", 300),
                ("img_border_frac", .1),
                ("pose_update_dt", 0.25),
                ("traj_num_pts", 400),
                ("lc_marker_color", "#7fffd4"),
                ("lc_marker_cube_size", 0.5),
            ]
        )

        self.aligned_traj_viz = self.get_parameter("aligned_traj_viz").value
        self.lc_marker_viz = self.get_parameter("lc_marker_viz").value
        self.output_dir = Path(expandvars_recursive(self.get_parameter("output_dir").value))
        self.img_dim = self.get_parameter("img_dim").value
        self.img_border_frac = self.get_parameter("img_border_frac").value
        self.pose_update_dt = self.get_parameter("pose_update_dt").value
        self.traj_num_pts = self.get_parameter("traj_num_pts").value
        self.lc_marker_cube_size = self.get_parameter("lc_marker_cube_size").value

        # parse hex color to (r, g, b) floats
        hex_color = self.get_parameter("lc_marker_color").value.lstrip('#')
        self.lc_marker_color = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        # parse viz_robot_ids: [-1] means show all
        viz_robot_ids_param = self.get_parameter("viz_robot_ids").value
        if viz_robot_ids_param == [-1]:
            self.viz_robot_ids = None
        else:
            self.viz_robot_ids = set(viz_robot_ids_param)

        assert self.aligned_traj_viz, "Only aligned trajectory visualizer currently supported."
        self.get_logger().info(f"Saving loop closure visualizations to {str(self.output_dir )}")

        # internal variables
        self.trajectories = {id_i: None for id_i in self.all_ids}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = cv_bridge.CvBridge()
        
        # load prior session maps
        self.get_logger().info("Loading prior maps.")
        for prior_id, map_path in zip(self.prior_session_ids, self.prior_session_maps):
            prior_map = ROMANMap.from_pickle(map_path)
            self.trajectories[prior_id] = PoseData.from_times_and_poses(prior_map.times, prior_map.trajectory, time_tol=np.inf)
            self.get_logger().info(f"Loaded map for robot {prior_id} from {map_path}.")
        
        self.setup_ros()
        self.get_logger().info("ROMAN Loop Closure Visualizer Setup complete.",)
        
    def setup_ros(self):

        # ros publishers
        if self.aligned_traj_viz:
            self.traj_img_pub = self.create_publisher(sensor_msgs.Image,
                "roman/roman_lc/aligned_traj", qos_profile=QoSProfile(depth=10))
        if self.lc_marker_viz:
            self.lc_marker_pub = self.create_publisher(visualization_msgs.MarkerArray,
                "roman/roman_lc/lc_markers", qos_profile=QoSProfile(depth=10))
            self.lc_marker_id = 0
            self.lc_markers = []

        # tf buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            
        # ros subscribers
        self.create_subscription(roman_msgs.LoopClosure, 
                "roman/roman_lc/loop_closure", self.lc_cb, qos_profile=QoSProfile(depth=10))
        self.timer = self.create_timer(self.pose_update_dt, self.timer_cb)


    def _should_viz_lc(self, robot1_id, robot2_id):
        if self.viz_robot_ids is None:
            return True
        lc_robots = {robot1_id, robot2_id}
        return lc_robots <= (self.viz_robot_ids | {self.ego_id}) and bool(lc_robots & self.viz_robot_ids)

    def lc_cb(self, lc_msg: roman_msgs.LoopClosure):

        if not self._should_viz_lc(lc_msg.robot1_id, lc_msg.robot2_id):
            return

        # publish cube marker at ego robot's pose
        if self.lc_marker_viz:
            if lc_msg.robot1_id == self.ego_id:
                ego_time = time_stamp_to_float(lc_msg.robot1_time)
            elif lc_msg.robot2_id == self.ego_id:
                ego_time = time_stamp_to_float(lc_msg.robot2_time)
            else:
                ego_time = None

            ego_traj = self.trajectories[self.ego_id]
            if ego_time is not None and ego_traj is not None:
                T_odom_ego = ego_traj.pose(ego_time)
                marker = visualization_msgs.Marker()
                marker.header.frame_id = self.ego_odom_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "lc_markers"
                marker.id = self.lc_marker_id
                self.lc_marker_id += 1
                marker.type = visualization_msgs.Marker.CUBE
                marker.action = visualization_msgs.Marker.ADD
                marker.pose.position = rnp.msgify(geometry_msgs.Point, T_odom_ego[:3, 3].reshape(-1))
                marker.pose.orientation = rnp.msgify(geometry_msgs.Quaternion, Rot.from_matrix(T_odom_ego[:3, :3]).as_quat())
                marker.scale.x = self.lc_marker_cube_size
                marker.scale.y = self.lc_marker_cube_size
                marker.scale.z = self.lc_marker_cube_size
                marker.color.a = 1.0
                marker.color.r = self.lc_marker_color[0]
                marker.color.g = self.lc_marker_color[1]
                marker.color.b = self.lc_marker_color[2]
                self.lc_markers.append(marker)
                self.lc_marker_pub.publish(visualization_msgs.MarkerArray(markers=self.lc_markers))

        # align the trajectories
        traj1 = deepcopy(self.trajectories[lc_msg.robot1_id])
        traj2 = deepcopy(self.trajectories[lc_msg.robot2_id])
        t1 = time_stamp_to_float(lc_msg.robot1_time)
        t2 = time_stamp_to_float(lc_msg.robot2_time)
        T_robot1t1_robot2t2 = rnp.numpify(lc_msg.transform_robot1_robot2).astype(np.float64)
        traj1.T_premulitply = None
        traj2.T_premulitply = None
        
        T_odom1_robot1t1 = traj1.pose(t1)
        T_odom2_robot2t2 = traj2.pose(t2)
        T_odom1_odom2 = T_odom1_robot1t1 @ T_robot1t1_robot2t2 @ np.linalg.inv(T_odom2_robot2t2)
        traj2.T_premultiply = T_odom1_odom2

        # collect the trajectoriesalong
        points = []
        for i, traj in enumerate([traj1, traj2]):
            times = np.linspace(traj.t0, traj.tf, self.traj_num_pts)
            points.append(np.array([traj.position(t)[:2] for t in times]))
            points[-1][:,1] *= -1 # flip along y axis since this will be in image coordinates
        # loop closure points
        points.append(np.array([traj1.position(t1)[:2], traj2.position(t2)[:2]]))
        points[-1][:,1] *= -1

        # scale trajectories
        minx = np.min(np.concatenate([points[0][:,0], points[1][:,0]]))
        maxx = np.max(np.concatenate([points[0][:,0], points[1][:,0]]))
        miny = np.min(np.concatenate([points[0][:,1], points[1][:,1]]))
        maxy = np.max(np.concatenate([points[0][:,1], points[1][:,1]]))
        dx = maxx - minx
        dy = maxy - miny
        maxx += 0.5*dx*self.img_border_frac
        minx -= 0.5*dx*self.img_border_frac
        maxy += 0.5*dy*self.img_border_frac
        miny -= 0.5*dy*self.img_border_frac
        max_dim = np.max([maxx - minx, maxy - miny])

        for i in range(3):
            points[i][:,0] -= minx
            points[i][:,1] -= miny
            points[i][:,0] *= self.img_dim / max_dim
            points[i][:,1] *= self.img_dim / max_dim

        # draw trajectories
        img = np.ones((self.img_dim, self.img_dim, 3)).astype(np.uint8)*255
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
        for i in range(3):
            points_i = points[i].astype(np.int32).reshape((-1,1,2))
            img = cv.polylines(img, [points_i], isClosed=False, color=colors[i], thickness=2)

        img = cv.putText(img, f"loop closure between {lc_msg.robot1_id} and {lc_msg.robot2_id}", (10, self.img_dim - 2*15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        img = cv.putText(img, f"{lc_msg.num_associations} matched objects", (10, self.img_dim - 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        f_name = f"robots_{lc_msg.robot1_id}_{lc_msg.robot2_id}_submaps_{lc_msg.submap1_id}_{lc_msg.submap2_id}.png"
        cv.imwrite(str(self.output_dir / f_name), img)
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "overhead"
        self.traj_img_pub.publish(img_msg)

    def timer_cb(self):
        # record times and poses
        curr_time = self.get_clock().now()
        curr_time_float = time_stamp_to_float(curr_time.to_msg())

        for robot_id in self.live_ids:
                
            try:
                T_odom_flu_msg = self.tf_buffer.lookup_transform(self.odom_frames[robot_id], 
                    self.flu_ref_frames[robot_id], curr_time, rclpy.duration.Duration(seconds=0.5))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
                # self.get_logger().warning("tf lookup failed")
                # print(ex)
                return
            T_odom_flu = rnp.numpify(T_odom_flu_msg.transform).astype(np.float64)
            if self.trajectories[robot_id] is None:
                self.trajectories[robot_id] = PoseData.from_times_and_poses(np.array([curr_time_float]), np.array([T_odom_flu]), time_tol=np.inf)
            else:
                self.trajectories[robot_id].times = np.concatenate([self.trajectories[robot_id].times, [curr_time_float]])
                self.trajectories[robot_id].positions = np.concatenate([self.trajectories[robot_id].positions, [T_odom_flu[:3,3]]])
                self.trajectories[robot_id].orientations = np.concatenate([self.trajectories[robot_id].orientations, [Rot.as_quat(Rot.from_matrix(T_odom_flu[:3,:3]))]])


def main():

    rclpy.init()
    node = LCVizNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

