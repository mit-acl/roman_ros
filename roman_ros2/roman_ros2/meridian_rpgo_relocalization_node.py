#!/usr/bin/env python3
"""
Relocalize the live robot against fixed prior maps from ROMAN loop closures with Meridian's
RPGO (CLIPPER outlier rejection + pose graph), publishing map_frame <- odom_frame on /tf and the
optimized trajectory. Replaces Hydra-Multi's map <- odom estimate; see
meridian_rpgo_relocalization.py for the solver.
"""

import signal
import threading
import time

import numpy as np
from numpy.linalg import inv

# ROS imports
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerOptions
import tf2_ros
import ros2_numpy as rnp

# ROS msgs
import roman_msgs.msg as roman_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
from ros_system_monitor_msgs.msg import NodeInfoMsg

# Local imports
from roman_ros2.meridian_rpgo_relocalization import (
    LoopClosureCandidate, MeridianRPGOParams, MeridianRPGORelocalizer, OdometryKeyframes)


def ros_time_to_float(t) -> float:
    return t.sec + t.nanosec * 1e-9


class MeridianRPGORelocalizationNode(Node):

    def __init__(self):
        super().__init__("meridian_rpgo_relocalization")

        defaults = MeridianRPGOParams()
        self.declare_parameters(
            namespace='',
            parameters=[
                ("nickname", "meridian_rpgo"), # node nickname for status messages
                ("status_rate_hz", 1.0),
                ("ego_id", 0), # robot id of the live robot in the loop closures; all others are prior maps
                ("map_frame", "map"), # fixed frame of the prior maps (published parent)
                ("odom_frame", "odom"), # live odometry frame (published child)
                ("base_frame", "base_link"), # live robot frame the ROMAN submaps are in
                ("tf_rate_hz", 10.0),
                ("odom_rate_hz", 10.0), # rate odom <- base is sampled for keyframes
                ("keyframe_trans_m", 0.5),
                ("keyframe_rot_deg", 10.0),
                ("rot_consistency_sigma_deg", defaults.rot_consistency_sigma_deg),
                ("rot_consistency_eps_deg", defaults.rot_consistency_eps_deg),
                ("trans_consistency_sigma_m", defaults.trans_consistency_sigma_m),
                ("trans_consistency_eps_m", defaults.trans_consistency_eps_m),
                ("added_trans_noise_m_per_m", defaults.added_trans_noise_m_per_m),
                ("added_rot_noise_deg_per_m", defaults.added_rot_noise_deg_per_m),
                ("single_lc_per_live_submap", defaults.single_lc_per_live_submap),
                ("fuse_lc_score", defaults.fuse_lc_score),
                ("lc_score_associations_ref", defaults.lc_score_associations_ref),
                ("min_num_inliers", defaults.min_num_inliers),
                ("optimization_method", defaults.optimization_method), # "pgo" or "frame_align"
                ("odom_rot_sigma_deg", defaults.odom_rot_sigma_deg),
                ("odom_trans_sigma_m", defaults.odom_trans_sigma_m),
                ("prior_rot_sigma_deg", defaults.prior_rot_sigma_deg),
                ("prior_trans_sigma_m", defaults.prior_trans_sigma_m),
            ]
        )
        gp = lambda name: self.get_parameter(name).value
        self.nickname = gp("nickname")
        self.ego_id = gp("ego_id")
        self.map_frame = gp("map_frame")
        self.odom_frame = gp("odom_frame")
        self.base_frame = gp("base_frame")
        params = MeridianRPGOParams(**{name: gp(name) for name in MeridianRPGOParams.__dataclass_fields__})
        self.relocalizer = MeridianRPGORelocalizer(params)

        # state shared with the solver thread
        self.lock = threading.Lock()
        self.candidates = []
        self.odometry = OdometryKeyframes(gp("keyframe_trans_m"), gp("keyframe_rot_deg"))
        self.T_map_odom = None # None until localized: no transform is published before that
        self.result = None
        self.num_solves = 0
        self.last_solve_time_s = None
        self.new_lc = threading.Event()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.lc_sub = self.create_subscription(roman_msgs.LoopClosure, "roman/roman_lc/loop_closure",
                                               self.lc_cb, QoSProfile(depth=100))
        self.path_pub = self.create_publisher(nav_msgs.Path, "meridian_rpgo/trajectory",
                                              QoSProfile(depth=1))
        self.status_pub = self.create_publisher(NodeInfoMsg, "meridian_rpgo/status",
                                                QoSProfile(depth=10))
        # each timer in its own callback group so a slow one never delays the others
        self.odom_timer = self.create_timer(1.0 / gp("odom_rate_hz"), self.odom_timer_cb,
                                            callback_group=MutuallyExclusiveCallbackGroup())
        self.tf_timer = self.create_timer(1.0 / gp("tf_rate_hz"), self.tf_timer_cb,
                                          callback_group=MutuallyExclusiveCallbackGroup())
        self.status_timer = self.create_timer(1.0 / gp("status_rate_hz"), self.status_timer_cb,
                                              callback_group=MutuallyExclusiveCallbackGroup())

        self.running = True
        self.solver_thread = threading.Thread(target=self.solver_loop, daemon=True)
        self.solver_thread.start()
        self.log_and_send_status(
            f"Meridian RPGO relocalization started ({self.map_frame} <- {self.odom_frame}, "
            f"live robot id {self.ego_id}).", status=NodeInfoMsg.STARTUP)

    def odom_timer_cb(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame,
                                                     rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return
        T_odom_base = rnp.numpify(tf_msg.transform)
        with self.lock:
            self.odometry.add(ros_time_to_float(tf_msg.header.stamp), T_odom_base)

    def lc_cb(self, msg: roman_msgs.LoopClosure):
        T_1_2 = rnp.numpify(msg.transform_robot1_robot2)
        T_odom1_1 = rnp.numpify(msg.robot1_submap_pose)
        T_odom2_2 = rnp.numpify(msg.robot2_submap_pose)
        if msg.robot1_id == self.ego_id and msg.robot2_id != self.ego_id:
            candidate = LoopClosureCandidate(
                live_submap_id=msg.submap1_id, prior_key=(msg.robot2_id, msg.submap2_id),
                live_time=ros_time_to_float(msg.robot1_time), T_odom_live=T_odom1_1,
                T_map_prior=T_odom2_2, T_prior_live=inv(T_1_2),
                num_associations=msg.num_associations)
        elif msg.robot2_id == self.ego_id and msg.robot1_id != self.ego_id:
            candidate = LoopClosureCandidate(
                live_submap_id=msg.submap2_id, prior_key=(msg.robot1_id, msg.submap1_id),
                live_time=ros_time_to_float(msg.robot2_time), T_odom_live=T_odom2_2,
                T_map_prior=T_odom1_1, T_prior_live=T_1_2,
                num_associations=msg.num_associations)
        else:
            # intra-robot (live-live) or prior-prior loop closures: the prior maps are fixed and
            # the live odometry is trusted, so neither constrains map <- odom
            return
        with self.lock:
            self.candidates.append(candidate)
        self.new_lc.set()

    def solver_loop(self):
        """Re-solve with all candidates whenever new loop closures arrive; loop closures that
        arrive during a solve are picked up by the next one."""
        while self.running:
            if not self.new_lc.wait(timeout=0.5):
                continue
            self.new_lc.clear()
            with self.lock:
                candidates = list(self.candidates)
                odometry = self.odometry.copy()
            t0 = time.time()
            try:
                result = self.relocalizer.solve(candidates, odometry)
            except Exception as ex:
                self.log_and_send_status(f"solve failed: {ex}", status=NodeInfoMsg.ERROR)
                continue
            self.last_solve_time_s = time.time() - t0
            self.num_solves += 1
            with self.lock:
                self.result = result
                if result.localized:
                    self.T_map_odom = result.T_map_odom
            if result.localized:
                self.get_logger().info(
                    f"localized: {len(result.inlier_indices)} / {result.num_candidates} inliers "
                    f"({self.last_solve_time_s:.2f} s)")
                self.publish_path(result, odometry)

    def tf_timer_cb(self):
        with self.lock:
            T_map_odom = self.T_map_odom
        if T_map_odom is None:
            return
        tf_msg = geometry_msgs.TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.map_frame
        tf_msg.child_frame_id = self.odom_frame
        tf_msg.transform = rnp.msgify(geometry_msgs.Transform, T_map_odom)
        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_path(self, result, odometry: OdometryKeyframes):
        if result.T_map_keyframes is not None:
            times, poses = result.keyframe_times, result.T_map_keyframes
        else:
            times, poses = odometry.times, [result.T_map_odom @ T for T in odometry.poses]
        path = nav_msgs.Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame
        for t, T in zip(times, poses):
            pose = geometry_msgs.PoseStamped()
            pose.header.frame_id = self.map_frame
            pose.header.stamp = rclpy.time.Time(nanoseconds=int(t * 1e9)).to_msg()
            pose.pose = rnp.msgify(geometry_msgs.Pose, T)
            path.poses.append(pose)
        self.path_pub.publish(path)

    def status_timer_cb(self):
        with self.lock:
            result = self.result
            num_candidates = len(self.candidates)
            num_keyframes = len(self.odometry)
            localized = self.T_map_odom is not None
        solve_time = "N/A" if self.last_solve_time_s is None else f"{self.last_solve_time_s:.2f} s"
        num_inliers = 0 if result is None else len(result.inlier_indices)
        note = (f"{num_inliers} / {num_candidates} inliers, {num_keyframes} odom keyframes, "
                f"solves: {self.num_solves}, last solve: {solve_time}")
        if num_keyframes == 0:
            self.send_status_msg(f"no {self.odom_frame} <- {self.base_frame} TF yet; " + note,
                                 NodeInfoMsg.WARNING)
        elif not localized:
            self.send_status_msg("not localized; " + note, NodeInfoMsg.STARTUP)
        elif result is not None and not result.localized:
            # keeps publishing the last localization
            self.send_status_msg("latest solve below min_num_inliers, holding last estimate; "
                                 + note, NodeInfoMsg.WARNING)
        else:
            self.send_status_msg("localized; " + note, NodeInfoMsg.NOMINAL)

    def log_and_send_status(self, note, status=NodeInfoMsg.NOMINAL):
        self.get_logger().info(note)
        self.send_status_msg(note, status)

    def send_status_msg(self, note, status=NodeInfoMsg.NOMINAL):
        status_msg = NodeInfoMsg()
        status_msg.nickname = self.nickname
        status_msg.node_name = self.get_fully_qualified_name()
        status_msg.status = status
        status_msg.notes = note
        self.status_pub.publish(status_msg)

    def destroy_node(self):
        self.running = False
        self.new_lc.set()
        self.solver_thread.join(timeout=5.0)
        super().destroy_node()


def main():

    # handle Ctrl-C here rather than in rclpy's signal handler, so the solver thread is stopped
    # before the context shuts down under it
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # also when started with SIGINT ignored (e.g., as a background job)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = MeridianRPGORelocalizationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
