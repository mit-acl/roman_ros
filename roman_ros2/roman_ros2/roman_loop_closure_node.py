#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv
import time
import os
import threading
from collections import deque

# ROS imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile

# ROS msgs
import roman_msgs.msg as roman_msgs
from ros_system_monitor_msgs.msg import NodeInfoMsg
from pose_graph_tools_msgs.msg import PoseGraph

# Custom modules
from roman.align.object_registration import InsufficientAssociationsException

from roman.params.submap_align_params import SubmapAlignParams
from roman.map.map import Submap, ROMANMap, submaps_from_roman_map, SubmapParams

# Local imports
from roman_ros2.utils import lc_to_pose_graph_msg, lc_to_msg, msg_to_submap

class ROMANLoopClosureNodeBaseClass(Node):

    def __init__(self, node_name):
        super().__init__(node_name)

        # ros parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ("config_path", ""),
                ("lc_required_associations", 6), # minimum number of associations to accept a loop closure
                ("nickname", "roman_lc"), # node nickname for status messages
                ("status_rate_hz", 1.0), # rate of the loop closure summary status messages
                ("lc_std_dev_rotation_deg", 1.0), # standard deviation for rotation in degrees
                ("lc_std_dev_translation_m", 0.5), # standard deviation for translation in meters
                ('submap_knn', -1), # number of nearest neighbor submaps to consider for registration, -1 for all
                ('k_latest_submaps', -1), # only register against the latest k submaps of each robot, -1 for all
                ("ego_id", 0), # robot id of the ego robot
                ("ego_name", ""), # robot name of the ego robot
                # NOTE: for the following prameters, [''] is a placeholder for [] because ROS2 does not allow empty lists as default params
                ("team_ids", [1]), # integer ids for online robots
                ("team_names", ['']), # string names for online robots
                ("prior_session_maps", ['']), # Prior session map paths or a single directory with numbered prior maps
                ("prior_session_ids", [-1]) # Prior session map ids. Not needed if prior session maps is a directory
            ]
        )
        
        self.config_path = self.get_parameter("config_path").value
        self.lc_required_associations = self.get_parameter("lc_required_associations").value
        self.lc_std_dev_rotation_deg = self.get_parameter("lc_std_dev_rotation_deg").value
        self.lc_std_dev_translation_m = self.get_parameter("lc_std_dev_translation_m").value
        self.submap_knn = self.get_parameter("submap_knn").value
        self.k_latest_submaps = self.get_parameter("k_latest_submaps").value
        self.nickname = self.get_parameter("nickname").value
        self.status_rate_hz = self.get_parameter("status_rate_hz").value
        self.ego_id = self.get_parameter("ego_id").value
        self.ego_name = self.get_parameter("ego_name").value
        self.team_ids = self.get_parameter("team_ids").value
        self.team_names = self.get_parameter("team_names").value
        self.prior_session_ids = self.get_parameter("prior_session_ids").value
        self.prior_session_maps = self.get_parameter("prior_session_maps").value

        # make default lists empty
        if self.team_names == ['']:
            self.team_ids = []
            self.team_names = []
        if self.prior_session_maps == ['']:
            self.prior_session_ids = []
            self.prior_session_maps = []
        # if prior session maps is a directory, load all .pkl files in the directory
        if len(self.prior_session_maps) > 0 and \
                os.path.isdir(self.prior_session_maps[0]):
            prior_map_dir = self.prior_session_maps[0]
            self.prior_session_maps = []
            self.prior_session_ids = []
            for file_name in os.listdir(prior_map_dir):
                if file_name.endswith('.roman_map.pkl'):
                    map_path = os.path.join(prior_map_dir, file_name)
                    try:
                        robot_id = int(file_name.split('.roman_map.pkl')[0])
                    except ValueError:
                        self.get_logger().warning(f"Could not parse robot id from prior map file name: {file_name}. Skipping.")
                        continue
                    self.prior_session_maps.append(map_path)
                    self.prior_session_ids.append(robot_id)

        # organize multi-robot metadata
        self.live_names = [self.ego_name] + self.team_names
        self.live_ids = [self.ego_id] + self.team_ids
        self.all_ids = self.live_ids + self.prior_session_ids

        if self.submap_knn < 0:
            self.submap_knn = None

        # check params
        assert self.ego_name != "", "ERROR: ego_robot_name param must be set."
        assert len(self.team_ids) == len(self.team_names), "ERROR: team_ids and team_names must be the same length."
        assert len(self.prior_session_maps) == len(self.prior_session_ids), \
            "ERROR: prior_session_maps and prior_session_ids must be the same length."
        assert len(set(self.all_ids)) == len(self.all_ids), \
            f"ERROR: all robot ids must be unique. Current team_ids: {self.team_ids}, " + \
            f"prior_session_ids: {self.prior_session_ids}."

class ROMANLoopClosureNode(ROMANLoopClosureNodeBaseClass):

    def __init__(self):
        
        super().__init__('roman_loop_closure_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                # publish only the loop closure with the most associations (ties: submap
                # similarity) for each new submap, across all robots it was registered against
                ("best_lc_only", False),
                # do not register a robot's submaps against its own earlier submaps
                ("no_self_lcs", False),
            ]
        )
        self.best_lc_only = self.get_parameter("best_lc_only").value
        self.no_self_lcs = self.get_parameter("no_self_lcs").value
        self.status_pub = self.create_publisher(NodeInfoMsg, "roman/roman_lc/status",
                                                qos_profile=QoSProfile(depth=10))

        # internal variables
        self.covariance = np.diag([np.deg2rad(self.lc_std_dev_rotation_deg)**2]*3 + 
                                  [self.lc_std_dev_translation_m**2]*3)
        # loop closure counts and the duration of the latest detection, for the status timer
        self.num_inter_robot_lcs = 0
        self.num_intra_robot_lcs = 0
        self.num_ego_submaps = 0  # submaps received from the ego robot
        self.last_detection_time_s = None
        self.last_lc_stamp = None  # ROS time the latest loop closure was found
        self.new_lc_flag = False
         
        self.submaps = {live_id: [] for live_id in self.live_ids}

        # Prior maps load in a background thread after ROS setup so that live submaps arriving
        # meanwhile are not dropped: they wait in this queue and are processed, in order, once
        # loading finishes.
        self.prior_maps_loaded = False
        self.prior_load_error = None
        self.submap_queue = deque()
        self.queue_lock = threading.Lock()

        if self.config_path == "":
            self.submap_align_params = SubmapAlignParams()
        else:
            self.submap_align_params = SubmapAlignParams.from_yaml(self.config_path)

        self.submap_params = SubmapParams.from_submap_align_params(self.submap_align_params)
        
        self.roman_reg = self.submap_align_params.get_object_registration()

        # Check params
        assert self.submap_knn is None or self.submap_align_params.submap_descriptor is not None

        # submap callbacks and the queue drain share this group so submaps are handled one at a time
        self.lc_callback_group = MutuallyExclusiveCallbackGroup()
        self.setup_ros()
        self.log_and_send_status("ROMAN Loop Closure Node ROS setup complete.", status=NodeInfoMsg.STARTUP)

        # own callback group so the status keeps publishing while a detection is running
        self.status_timer = self.create_timer(1.0 / self.status_rate_hz, self.status_timer_cb,
                                              callback_group=MutuallyExclusiveCallbackGroup())

        # load prior session maps last, without blocking ROS
        self.drain_timer = None
        self.prior_load_thread = threading.Thread(target=self.load_prior_maps, daemon=True)
        self.prior_load_thread.start()

    def load_prior_maps(self):
        """
        Load the prior session maps (background thread), then hand any submaps queued meanwhile to
        the loop closure callback group.
        """
        self.log_and_send_status(f"Loading {len(self.prior_session_maps)} prior maps.",
                                 status=NodeInfoMsg.STARTUP)
        start_t = time.time()
        for prior_id, map_path in zip(self.prior_session_ids, self.prior_session_maps):
            try:
                prior_map = ROMANMap.from_pickle(map_path)
                submaps = submaps_from_roman_map(prior_map, self.submap_params)
            except Exception as e:
                # submaps stay queued; the status timer keeps reporting this error
                self.prior_load_error = f"Failed to load prior map {map_path}: {e!r}"
                self.get_logger().error(self.prior_load_error)
                return
            self.submaps[prior_id] = submaps
            self.log_and_send_status(f"Loaded {len(submaps)} submaps for robot {prior_id} from {map_path}.",
                                     status=NodeInfoMsg.STARTUP)
        with self.queue_lock:
            self.prior_maps_loaded = True
            num_queued = len(self.submap_queue)
        self.log_and_send_status(f"Prior maps loaded in {time.time() - start_t:.1f} s; processing "
                                 f"{num_queued} queued submaps.", status=NodeInfoMsg.STARTUP)
        if num_queued > 0:
            # drain from the executor so queued submaps never run concurrently with submap_cb
            self.drain_timer = self.create_timer(0.0, self.drain_timer_cb,
                                                 callback_group=self.lc_callback_group)

    def drain_timer_cb(self):
        self.drain_timer.cancel()
        self.drain_submap_queue()

    def drain_submap_queue(self):
        """
        Process the submaps queued while prior maps were loading, oldest first.
        """
        while True:
            with self.queue_lock:
                if not self.submap_queue:
                    return
                submap, robot_id = self.submap_queue.popleft()
            self.handle_new_submap(submap, robot_id)

    def setup_ros(self):
        
        # ros publishers
        self.loop_closure_pub = self.create_publisher(roman_msgs.LoopClosure, 
            "roman/roman_lc/loop_closure", qos_profile=QoSProfile(depth=10))
        self.pgt_lc_pub = self.create_publisher(PoseGraph, "roman/roman_lc/pose_graph_update", 
                                                qos_profile=QoSProfile(depth=10))
        
        # Each live robot's mapper publishes its own submaps, including ego.
        self.submap_subs = [
            self.create_subscription(roman_msgs.Submap, f"/{robot_name}/roman/submap",
                                     lambda msg, rid=robot_id: self.submap_cb(msg, rid), 20,
                                     callback_group=self.lc_callback_group)
            for robot_name, robot_id in zip(self.live_names, self.live_ids)
        ]

    def submap_cb(self, submap_msg: roman_msgs.Submap, robot_id: int):
        """
        Receive mapper-built submaps from ego or a teammate and trigger loop closure.
        """
        submap = msg_to_submap(submap_msg)
        if robot_id == self.ego_id:
            self.num_ego_submaps += 1
        if submap_msg.robot_id != robot_id:
            self.get_logger().warning(f"Submap message on robot {robot_id}'s topic says robot id "
                                      f"{submap_msg.robot_id}; using {robot_id}.")
        if self.submap_align_params.submap_descriptor is not None and submap.descriptor is None:
            self.get_logger().warning(f"Submap {submap.id} from robot {robot_id} has no descriptor; skipping it.")
            return
        with self.queue_lock:
            if not self.prior_maps_loaded:
                self.submap_queue.append((submap, robot_id))
                self.get_logger().info(f"Prior maps still loading; queued submap {submap.id} from robot "
                                       f"{robot_id} ({len(self.submap_queue)} queued).")
                return
        # keep arrival order: anything still queued goes before this submap
        self.drain_submap_queue()
        self.handle_new_submap(submap, robot_id)

    def handle_new_submap(self, submap: Submap, robot_id: int):
        """
        Store a received submap and run loop closure detection with it.
        """
        self.submaps[robot_id].append(submap)

        start_t = time.time()

        # run submap registration
        candidates = []
        if robot_id == self.ego_id:
            for r2 in self.all_ids:
                if self.no_self_lcs and r2 == robot_id:
                    continue
                candidates += self.run_submap_registration(submap, robot_id, r2)
        else:
            # run registration with ego robot
            candidates += self.run_submap_registration(submap, robot_id, self.ego_id)

        if self.best_lc_only and len(candidates) > 1:
            candidates = [max(candidates, key=lambda c: (len(c[2]), self.lc_similarity(submap, c[1])))]

        icp_target = None
        for other_id, submap2, associations, T_submapgrav1_submapgrav2 in candidates:
            icp_target = self.publish_loop_closure(submap, robot_id, other_id, submap2, associations,
                                                   T_submapgrav1_submapgrav2, icp_target)

        end_t = time.time()

        self.last_detection_time_s = end_t - start_t
        detection_time_ms = np.round(self.last_detection_time_s * 1e3, 1)
        if self.new_lc_flag:
            self.get_logger().info(f"Loop closure found for robot {robot_id}, submap {submap.id}. Detection time: {detection_time_ms} ms")
        else:
            self.get_logger().info(f"No loop closures found for robot {robot_id}, submap {submap.id}. Detection time: {detection_time_ms} ms")
        self.new_lc_flag = False
            
    def lc_similarity(self, submap: Submap, submap2: Submap) -> float:
        return Submap.similarity(submap, submap2) \
            if self.submap_align_params.submap_descriptor is not None else float('nan')

    def run_submap_registration(self, submap: Submap, robot_id: int, other_id: int):
        """
        Register a submap against another robot's submaps.

        Returns:
            list of (other_id, submap2, associations, T_submapgrav1_submapgrav2) for each
            pair with enough associations to become a loop closure
        """
        submap2: Submap

        other_submaps = self.submaps[other_id]
        
        # prohibit removing a submap with itself
        if robot_id == other_id:
            other_submaps = [submap2 for submap2 in other_submaps if submap2.id != submap.id]

        # bound to the most recent k submaps of the other robot
        if self.k_latest_submaps > 0:
            other_submaps = other_submaps[-self.k_latest_submaps:]

        # get k nearest neighbors in terms of submap similarity
        if self.submap_knn is not None and self.submap_align_params.submap_descriptor is not None:
            other_submaps = sorted(other_submaps, 
                key=lambda sm: Submap.similarity(submap, sm),
                reverse=True
            )[:self.submap_knn]
        
        # reject submaps with similarity lower than threshold
        if self.submap_align_params.submap_descriptor is not None:
            other_submaps = [submap2 for submap2 in other_submaps 
                             if Submap.similarity(submap, submap2) > 
                             self.submap_align_params.submap_descriptor_thresh]

        self.get_logger().info(f"Attempting to register {len(other_submaps)} " +
                               f"submaps between robot ids: ({robot_id}, {other_id})")

        candidates = []
        for submap2 in other_submaps:

            # run registration
            try:
                segments1 = submap.segments
                segments2 = submap2.segments
                # remove same robot same segments
                if robot_id == other_id:
                    segments1_ids = [seg.id for seg in segments1]
                    segments2_ids = [seg.id for seg in segments2]
                    segments1 = [seg for seg in segments1 if seg.id not in segments2_ids]
                    segments2 = [seg for seg in segments2 if seg.id not in segments1_ids]
                associations = self.roman_reg.register(segments1, segments2)
                T_submapgrav1_submapgrav2 = self.roman_reg.T_align(
                    segments1, segments2, associations,
                    xyz_yaw_only=self.submap_align_params.force_rm_lc_roll_pitch)
            except InsufficientAssociationsException:
                continue

            if len(associations) < self.lc_required_associations:
                continue

            candidates.append((other_id, submap2, associations, T_submapgrav1_submapgrav2))

        return candidates

    def publish_loop_closure(self, submap: Submap, robot_id: int, other_id: int, submap2: Submap,
                             associations: np.ndarray, T_submapgrav1_submapgrav2: np.ndarray,
                             icp_target=None):
        """
        Refine (optionally) and publish one loop closure. Returns the ICP target cloud so it
        can be reused for the next loop closure of the same submap.
        """
        # ICP only for pairs that will become loop closures; the target (this submap's
        # cloud with normals) is prepared once and reused for every candidate
        if self.submap_align_params.icp_on_submap_pcds:
            if icp_target is None:
                icp_target = self.roman_reg.prepare_icp_cloud(submap.point_cloud)
            t0 = time.time()
            T_object = T_submapgrav1_submapgrav2
            T_submapgrav1_submapgrav2 = self.roman_reg.refine_with_icp(
                T_object, submap.point_cloud, submap2.point_cloud,
                xyz_yaw_only=self.submap_align_params.force_rm_lc_roll_pitch, target_pcd=icp_target)
            dT = T_submapgrav1_submapgrav2 @ inv(T_object)
            kept = np.allclose(dT, np.eye(4))
            self.get_logger().info(
                f"ICP ({robot_id}, {other_id}) submaps ({submap.id}, {submap2.id}): " +
                ("kept object alignment" if kept else
                 f"refined by {np.linalg.norm(dT[:3, 3]):.3f} m, "
                 f"{np.degrees(np.arctan2(dT[1, 0], dT[0, 0])):+.2f} deg yaw") +
                f" ({1000 * (time.time() - t0):.0f} ms)")

        self.get_logger().info(f"Found {len(associations)} associations between robot_ids: " +
            f"({robot_id}, {other_id}) and submaps: ({submap.id}, {submap2.id}).")
        
        T_odom1_submapgrav1 = submap.pose_gravity_aligned
        T_odom2_submapgrav2 = submap2.pose_gravity_aligned
        T_odom1_submap1 = submap.pose_flu
        T_odom2_submap2 = submap2.pose_flu
        T_submap1_submap2 = inv(T_odom1_submap1) @ T_odom1_submapgrav1 @ \
            T_submapgrav1_submapgrav2 @ inv(T_odom2_submapgrav2) @ T_odom2_submap2
            
        pg_msg = lc_to_pose_graph_msg(robot_id, other_id, submap, submap2, T_submap1_submap2,
                                      self.covariance, self.get_clock().now().to_msg())
        self.pgt_lc_pub.publish(pg_msg)
        
        similarity = self.lc_similarity(submap, submap2)
        lc_msg = lc_to_msg(robot_id, other_id, submap, submap2, associations, 
                           T_submap1_submap2, self.covariance, self.get_clock().now().to_msg(),
                           submap_similarity=similarity)
        self.loop_closure_pub.publish(lc_msg)

        if robot_id == other_id:
            self.num_intra_robot_lcs += 1
        else:
            self.num_inter_robot_lcs += 1
        self.last_lc_stamp = self.get_clock().now()
        self.new_lc_flag = True
        return icp_target

    def status_timer_cb(self):
        """
        Send a summary of the loop closures found so far to the status topic.
        """
        if self.prior_load_error is not None:
            self.send_status_msg(self.prior_load_error, status=NodeInfoMsg.ERROR)
            return
        if not self.prior_maps_loaded:
            self.send_status_msg(f"Loading prior maps; {len(self.submap_queue)} submaps queued, "
                                 f"sms received: {self.num_ego_submaps}", status=NodeInfoMsg.STARTUP)
            return
        detection_time = "N/A" if self.last_detection_time_s is None \
            else f"{self.last_detection_time_s:.2f} s"
        time_since_lc = "N/A" if self.last_lc_stamp is None \
            else f"{(self.get_clock().now() - self.last_lc_stamp).nanoseconds * 1e-9:.1f} s"
        self.send_status_msg(f"Inter-robot lcs: {self.num_inter_robot_lcs}, "
                             f"Intra-robot lcs: {self.num_intra_robot_lcs}, "
                             f"sms processed: {self.num_ego_submaps}, "
                             f"last detection time: {detection_time}, "
                             f"time since last lc: {time_since_lc}")

    def log_and_send_status(self, note, status=NodeInfoMsg.NOMINAL):
        """
        Log a message and send it to the status topic.
        """
        self.get_logger().info(note)
        self.send_status_msg(note, status)
        

    def send_status_msg(self, note, status=NodeInfoMsg.NOMINAL):
        status_msg = NodeInfoMsg()
        status_msg.nickname = self.nickname
        status_msg.node_name = self.get_fully_qualified_name()
        status_msg.status = status
        status_msg.notes = note
        self.status_pub.publish(status_msg)


def main():

    rclpy.init()
    node = ROMANLoopClosureNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

