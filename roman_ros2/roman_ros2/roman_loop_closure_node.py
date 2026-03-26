#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
from typing import List
from copy import deepcopy
import time
import os

# ROS imports
import rclpy
from rclpy.node import Node
import ros2_numpy as rnp
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from rclpy.qos import QoSProfile

# ROS msgs
import roman_msgs.msg as roman_msgs
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as visualization_msgs
from ros_system_monitor_msgs.msg import NodeInfoMsg
from pose_graph_tools_msgs.msg import PoseGraph, PoseGraphEdge

# Custom modules
from robotdatapy.transform import transform_to_xyzrpy
from robotdatapy.data import PoseData

from roman.object.segment import Segment, SegmentMinimalData
from roman.align.roman_registration import ROMANRegistration, ROMANParams
from roman.align.object_registration import InsufficientAssociationsException

from roman.params.submap_align_params import SubmapAlignParams
from roman.utils import transform_rm_roll_pitch
from roman.map.map import Submap, ROMANMap, submaps_from_roman_map, extract_submap_descriptors, SubmapParams

# Local imports
from roman_ros2.utils import msg_to_segment, frame_descriptor_from_msg, float_to_ros_time, lc_to_pose_graph_msg, \
    lc_to_msg

@dataclass
class FrameDescriptor():
    descriptor: np.ndarray
    pose: np.ndarray
    time: float

@dataclass
class SegmentUpdateResult():
    """
    Class to store the result of a segment update.
    """
    submap_created: bool
    submap_segments: List[SegmentMinimalData] = None

class SegmentQueue():

    def __init__(self, submap_num_segments: int, submap_overlapping_segments: int):
        self.segments = dict()
        self.submap_num_segments = submap_num_segments
        self.submap_overlapping_segments = submap_overlapping_segments

    def update(self, segment: Segment):
        # update existing segment
        if segment.id in self.segments:
            segment.first_seen = self.segments[segment.id].first_seen
            self.segments[segment.id] = segment
            return SegmentUpdateResult(False)

        # create new segment
        segment.first_seen = segment.last_seen
        self.segments[segment.id] = segment

        # check if we have enough segments to create a submap
        if len(self.segments) < self.submap_num_segments:
            return SegmentUpdateResult(False)
        
        assert len(self.segments) == self.submap_num_segments, \
            "ERROR: Segment queue is incorrect size."

        # create submap
        submap = []
        for seg_id, seg in self.segments.items():
            submap.append(seg)

        segments_sorted_by_time = sorted(submap, key=lambda seg: np.mean([seg.first_seen, seg.last_seen]))
        rm_segment_ids = [seg.id for seg in segments_sorted_by_time[:-self.submap_overlapping_segments]]

        # remove segments that will not be include in subsequent submaps
        for seg_id in rm_segment_ids:
            del self.segments[seg_id]

        return SegmentUpdateResult(True, submap)

    @property
    def earliest_seen(self):
        return np.min([seg.first_seen for seg in self.segments.values()])
    
@dataclass
class LoopClosureInfo:
    robot_id1: int = None
    robot_id2: int = None
    submap_id1: int = None
    submap_id2: int = None

    @property
    def initialized(self):
        return self.robot_id1 is not None and \
               self.robot_id2 is not None and \
               self.submap_id1 is not None and \
               self.submap_id2 is not None

    def __str__(self):
        if not self.initialized:
            return "No loop closure found yet." 
        else:
            return f"Last loop closure between robot_ids: " + \
                f"({self.robot_id1}, {self.robot_id2})" + \
                f" and submaps: ({self.submap_id1}, {self.submap_id2})."
            
class ROMANLoopClosureNodeBaseClass(Node):

    def __init__(self, node_name):
        super().__init__(node_name)

        # ros parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ("config_path", ""),
                ("submap_num_segments", 40), # number of segments to create a submap from
                ("submap_overlapping_segments", 20), # number of overlapping segments between submaps
                ("lc_required_associations", 6), # minimum number of associations to accept a loop closure
                ("nickname", "roman_lc"), # node nickname for status messages
                ("lc_std_dev_rotation_deg", 1.0), # standard deviation for rotation in degrees
                ("lc_std_dev_translation_m", 0.5), # standard deviation for translation in meters
                ('submap_knn', -1), # number of nearest neighbor submaps to consider for registration, -1 for all
                ("ego_id", 0), # robot id of the ego robot
                ("ego_name", ""), # robot name of the ego robot
                ("ego_flu_ref_frame", ""), # ego robot body reference frame that generally has Z point roughly up
                ("ego_odom_frame", ""), # ego robot odometry frame
                # NOTE: for the following prameters, [''] is a placeholder for [] because ROS2 does not allow empty lists as default params
                ("team_ids", [1]), # integer ids for online robots
                ("team_names", ['']), # string names for online robots
                ("team_flu_ref_frames", ['']), # robot body reference frames that generally have Z point roughly up
                ("team_odom_frames", ['']), # robot odometry frames
                ("prior_session_maps", ['']), # Prior session map paths or a single directory with numbered prior maps
                ("prior_session_ids", [-1]) # Prior session map ids. Not needed if prior session maps is a directory
            ]
        )
        
        self.config_path = self.get_parameter("config_path").value
        self.submap_num_segments = self.get_parameter("submap_num_segments").value
        self.submap_overlapping_segments = self.get_parameter("submap_overlapping_segments").value
        self.lc_required_associations = self.get_parameter("lc_required_associations").value
        self.lc_std_dev_rotation_deg = self.get_parameter("lc_std_dev_rotation_deg").value
        self.lc_std_dev_translation_m = self.get_parameter("lc_std_dev_translation_m").value
        self.submap_knn = self.get_parameter("submap_knn").value
        self.nickname = self.get_parameter("nickname").value
        self.ego_id = self.get_parameter("ego_id").value
        self.ego_name = self.get_parameter("ego_name").value
        self.ego_flu_ref_frame = self.get_parameter("ego_flu_ref_frame").value
        self.ego_odom_frame = self.get_parameter("ego_odom_frame").value
        self.team_ids = self.get_parameter("team_ids").value
        self.team_names = self.get_parameter("team_names").value
        self.team_flu_ref_frames = self.get_parameter("team_flu_ref_frames").value
        self.team_odom_frames = self.get_parameter("team_odom_frames").value
        self.prior_session_ids = self.get_parameter("prior_session_ids").value
        self.prior_session_maps = self.get_parameter("prior_session_maps").value

        # make default lists empty
        if self.team_names == ['']:
            self.team_ids = []
            self.team_names = []
            self.team_flu_ref_frames = []
            self.team_odom_frames = []
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

        self.flu_ref_frames = {robot_id: robot_flu_ref_frame 
            for robot_id, robot_flu_ref_frame in 
            zip(self.live_ids, [self.ego_flu_ref_frame] + self.team_flu_ref_frames)}
        self.odom_frames = {robot_id: robot_odom_frame
            for robot_id, robot_odom_frame in 
            zip(self.live_ids, [self.ego_odom_frame] + self.team_odom_frames)}

        if self.submap_knn < 0:
            self.submap_knn = None

        # check params
        assert self.ego_name != "", "ERROR: ego_robot_name param must be set."
        assert self.ego_flu_ref_frame != "", "ERROR: ego_robot_flu_ref_frame param must be set."
        assert self.ego_odom_frame != "", "ERROR: ego_robot_odom_frame param must be set."
        assert len(self.team_ids) == len(self.team_names), "ERROR: team_ids and team_names must be the same length."
        assert len(self.team_flu_ref_frames) == len(self.team_ids), "ERROR: team_flu_ref_frames param must be set."
        assert len(self.team_odom_frames) == len(self.team_ids), "ERROR: team_odom_frames param must be set."
        assert len(self.prior_session_maps) == len(self.prior_session_ids), \
            "ERROR: prior_session_maps and prior_session_ids must be the same length."
        assert len(set(self.all_ids)) == len(self.all_ids), \
            f"ERROR: all robot ids must be unique. Current team_ids: {self.team_ids}, " + \
            f"prior_session_ids: {self.prior_session_ids}."

class ROMANLoopClosureNode(ROMANLoopClosureNodeBaseClass):

    def __init__(self):
        
        super().__init__('roman_loop_closure_node')
        self.status_pub = self.create_publisher(NodeInfoMsg, "roman/roman_lc/status", 
                                                qos_profile=QoSProfile(depth=10))

        # internal variables
        self.time_eps = 1.0
        self.covariance = np.diag([np.deg2rad(self.lc_std_dev_rotation_deg)**2]*3 + 
                                  [self.lc_std_dev_translation_m**2]*3)
        # time taken, id1, id2, submap_id1, submap_id2
        self.last_lc_info = LoopClosureInfo()
        self.last_lc_time = None
        self.new_lc_flag = False
         
        self.segment_queues = {
            live_id: SegmentQueue(self.submap_num_segments, self.submap_overlapping_segments)
        for live_id in self.live_ids}
        self.submaps = {live_id: [] for live_id in self.live_ids}
        self.times = {live_id: [] for live_id in self.live_ids}
        self.poses = {live_id: [] for live_id in self.live_ids}
        self.descriptors = {live_id: [] for live_id in self.live_ids}

        if self.config_path == "":
            self.submap_align_params = SubmapAlignParams()
        else:
            self.submap_align_params = SubmapAlignParams.from_yaml(self.config_path)

        self.submap_params = SubmapParams.from_submap_align_params(self.submap_align_params)
        
        self.roman_reg = self.submap_align_params.get_object_registration()

        # load prior session maps
        self.log_and_send_status("Loading prior maps.", status=NodeInfoMsg.STARTUP)
        for prior_id, map_path in zip(self.prior_session_ids, self.prior_session_maps):
            prior_map = ROMANMap.from_pickle(map_path)
            submaps = submaps_from_roman_map(prior_map, self.submap_params)
            self.submaps[prior_id] = submaps
            self.log_and_send_status(f"Loaded {len(submaps)} submaps for robot {prior_id} from {map_path}.", 
                                     status=NodeInfoMsg.STARTUP)
        
        # Check params
        assert self.submap_knn is None or self.submap_align_params.submap_descriptor is not None
        
        self.setup_ros()
        self.log_and_send_status("ROMAN Loop Closure Node setup complete.", status=NodeInfoMsg.STARTUP)
        
    def setup_ros(self):
        
        # ros publishers
        self.loop_closure_pub = self.create_publisher(roman_msgs.LoopClosure, 
            "roman/roman_lc/loop_closure", qos_profile=QoSProfile(depth=10))
        self.pgt_lc_pub = self.create_publisher(PoseGraph, "roman/roman_lc/pose_graph_update", 
                                                qos_profile=QoSProfile(depth=10))
        
        # ros subscribers
        self.segment_subs = [
            self.create_subscription(roman_msgs.Segment, f"/{robot_name}/roman/segment_updates", 
                                     lambda msg: self.seg_cb(msg, robot_id), 20)
        for robot_name, robot_id in zip(self.live_names, self.live_ids)]

        if self.submap_align_params.submap_descriptor is not None:
            self.descriptor_subs = [
                self.create_subscription(roman_msgs.FrameDescriptor, f"/{robot_name}/roman/frame_descriptor", 
                                        lambda msg: self.descriptor_cb(msg, robot_id), 20)
            for robot_name, robot_id in zip(self.live_names, self.live_ids)]

        # tf buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def seg_cb(self, seg_msg: roman_msgs.Segment, robot_id: int):
        """
        Callback function for segment messages.
        """
        # Send pulse
        detection_time_info_str = f"Last detection time: {self.last_lc_time} ms" \
            if self.last_lc_time is not None else "No loop closure detection run yet."
        self.send_status_msg(detection_time_info_str + " " + str(self.last_lc_info), 
                             status=NodeInfoMsg.NOMINAL)
        
        segment = msg_to_segment(seg_msg)
        seg_update_res = self.segment_queues[robot_id].update(segment)

        # record times and poses
        try:
            T_odom_flu_msg = self.tf_buffer.lookup_transform(self.odom_frames[robot_id], 
                self.flu_ref_frames[robot_id], float_to_ros_time(segment.last_seen), rclpy.duration.Duration(seconds=0.25))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            # self.get_logger().warning("tf lookup failed")
            # print(ex)
            return

        T_odom_flu = rnp.numpify(T_odom_flu_msg.transform).astype(np.float64)
        
        if segment.last_seen not in self.times[robot_id]:
            self.times[robot_id].append(segment.last_seen)
            self.poses[robot_id].append(T_odom_flu)

        if not seg_update_res.submap_created:
            return
        
        # create submap
        submap_ref_time = np.mean([np.mean([seg.first_seen, seg.last_seen]) for seg in seg_update_res.submap_segments])
        submap_id = len(self.submaps[robot_id])

        # sort times and poses
        sorted_times, sorted_poses = zip(*sorted(zip(self.times[robot_id], self.poses[robot_id])))
        sorted_times = list(sorted_times)
        sorted_poses = [pose for pose in sorted_poses]
        pd = PoseData.from_times_and_poses(sorted_times, sorted_poses, interp=False, time_tol=1e3)
        pd_idx = pd.idx(submap_ref_time, force_single=True)
        submap_ref_time = sorted_times[pd_idx]
        submap_ref_pose = sorted_poses[pd_idx]

        # refresh times and poses buffer
        pd.causal = True
        earliest_seen_idx = pd.idx(self.segment_queues[robot_id].earliest_seen + self.time_eps)
        self.times[robot_id] = sorted_times[earliest_seen_idx:]
        self.poses[robot_id] = sorted_poses[earliest_seen_idx:]
        
        submap = Submap(
            id=submap_id,
            time=submap_ref_time,
            pose_flu=submap_ref_pose,
            segments=[],
            segment_frame='submap_gravity_aligned'
        )
        for seg_i in seg_update_res.submap_segments:
            segment = deepcopy(seg_i)
            segment.transform(np.linalg.inv(submap.pose_gravity_aligned))
            submap.segments.append(segment)
        if self.submap_align_params.submap_descriptor == 'mean_semantic':
            submap.descriptor = \
                np.mean([seg.semantic_descriptor.flatten() for seg in submap.segments], axis=0)
            submap.descriptor /= np.linalg.norm(submap.descriptor)
        elif self.submap_align_params.submap_descriptor is not None:
            # sort descriptor buffer
            sorted_descriptors = sorted(self.descriptors[robot_id])
            descriptor_times, descriptor_poses, descriptors = zip(*sorted_descriptors)

            # extract submap descriptor
            extract_submap_descriptors(
                submaps=[submap],
                descriptor_times=descriptor_times,
                descriptors=descriptors,
                poses=descriptor_poses,
                submap_params=self.submap_params
            )

            # refresh descriptor buffer
            pd = PoseData.from_times_and_poses(descriptor_times, descriptor_poses, interp=False, time_tol=1e3, causal=True)
            earliest_seen_idx = pd.idx(self.segment_queues[robot_id].earliest_seen + self.time_eps)
            self.descriptors[robot_id] = list(sorted_descriptors)[earliest_seen_idx:]

        self.submaps[robot_id].append(submap)

        start_t = time.time()

        # run submap registration
        if robot_id == self.ego_id:
            for r2 in self.all_ids:
                self.run_submap_registration(submap, robot_id, r2)
        else:
            # run registration with ego robot
            self.run_submap_registration(submap, robot_id, self.ego_id)

        end_t = time.time()

        self.last_lc_time = np.round((end_t - start_t) * 1e3, 1)
        if self.new_lc_flag:
            self.log_and_send_status(f"Loop closure found for robot {robot_id}, submap {submap.id}. Detection time: {self.last_lc_time} ms")
        else:
            self.log_and_send_status(f"No loop closures found for robot {robot_id}, submap {submap.id}. Detection time: {self.last_lc_time} ms")
        self.new_lc_flag = False
            
    def descriptor_cb(self, descriptor_msg: roman_msgs.FrameDescriptor, robot_id: int):
        """
        Callback function for frame descriptor messages.
        """
        self.descriptors[robot_id].append(frame_descriptor_from_msg(descriptor_msg)) # (time, pose, descriptor)

    def run_submap_registration(self, submap: Submap, robot_id: int, other_id: int):
        submap2: Submap

        other_submaps = self.submaps[other_id]
        
        # prohibit removing a submap with itself
        if robot_id == other_id:
            other_submaps = [submap2 for submap2 in other_submaps if submap2.id != submap.id]

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

        self.log_and_send_status(f"Attempting to register {len(other_submaps)} " + 
                                 f"submaps between robot ids: ({robot_id}, {other_id})")

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
                    segments1, segments2, associations)
                if self.submap_align_params.force_rm_lc_roll_pitch:
                    T_submapgrav1_submapgrav2 = transform_rm_roll_pitch(T_submapgrav1_submapgrav2)
            except InsufficientAssociationsException:
                continue

            if len(associations) < self.lc_required_associations:
                continue

            self.log_and_send_status(f"Found {len(associations)} associations between robot_ids: " + 
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
            
            lc_msg = lc_to_msg(robot_id, other_id, submap, submap2, associations, 
                               T_submap1_submap2, self.covariance, self.get_clock().now().to_msg())
            self.loop_closure_pub.publish(lc_msg)

            # store info about last loop closure
            self.last_lc_info = LoopClosureInfo(robot_id1=robot_id, robot_id2=other_id,
                                                submap_id1=submap.id, submap_id2=submap2.id)
            self.new_lc_flag = True

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

