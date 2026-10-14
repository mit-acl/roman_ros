import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from roman.map.map import SubmapParams
from roman.object.segment import SegmentMinimalData
from roman_ros2.submap_builder import SegmentQueue, SubmapBuilder


def segment(identifier, t0=0., tf=2., center=None, descriptor=None):
    return SegmentMinimalData(identifier, np.zeros(3) if center is None else center,
                              1., 0.1, 0.2, 0.3, None, descriptor, t0, tf)


class SegmentQueueTests(unittest.TestCase):
    def test_updates_replace_segments_without_advancing_window(self):
        queue = SegmentQueue(2, 1)
        self.assertFalse(queue.update(segment(0)).submap_created)
        self.assertFalse(queue.update(segment(0, tf=4)).submap_created)
        result = queue.update(segment(1, tf=5))
        self.assertTrue(result.submap_created)
        self.assertEqual(result.submap_segments[0].last_seen, 4)
        self.assertEqual(list(queue.segments), [1])
        self.assertFalse(queue.update(segment(1, tf=6)).submap_created)
        self.assertTrue(queue.update(segment(2, tf=7)).submap_created)

    def test_zero_overlap_empties_queue_and_allows_next_window(self):
        queue = SegmentQueue(2, 0)
        for identifier in range(4):
            result = queue.update(segment(identifier))
            self.assertEqual(result.submap_created, identifier % 2 == 1)
        self.assertEqual(queue.segments, {})
        self.assertIsNone(queue.earliest_seen)

    def test_overlap_retains_most_recent_segments(self):
        queue = SegmentQueue(3, 1)
        queue.update(segment(0, t0=8, tf=10))
        queue.update(segment(1, t0=0, tf=2))
        queue.update(segment(2, t0=4, tf=6))
        self.assertEqual(list(queue.segments), [0])
        self.assertEqual(queue.earliest_seen, 8)

    def test_invalid_window_parameters(self):
        for count, overlap in ((0, 0), (2, -1), (2, 2), (2, 3)):
            with self.subTest(count=count, overlap=overlap), self.assertRaises(ValueError):
                SegmentQueue(count, overlap)


class SubmapBuilderTests(unittest.TestCase):
    def test_reference_pose_transform_and_snapshot_ownership(self):
        builder = SubmapBuilder(2, 1, SubmapParams())
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('ZYX', [np.pi / 2, 0.2, 0.1]).as_matrix()
        pose[:3, 3] = [2, 3, 0]
        builder.update_frame(1., pose)
        builder.update_frame(3., np.eye(4))
        first = segment(0, t0=0, tf=2, center=np.array([3., 4., 1.]))
        second = segment(1, t0=2, tf=4, center=np.array([4., 4., 1.]))
        self.assertIsNone(builder.update_segment(first))
        submap = builder.update_segment(second)
        self.assertEqual((submap.id, submap.time), (0, 1.))
        self.assertEqual(submap.segment_frame, 'submap_gravity_aligned')
        np.testing.assert_allclose(submap.segments[0].center.flatten(), [1, -1, 1], atol=1e-12)
        np.testing.assert_equal(submap.pose_flu, pose)
        np.testing.assert_equal(first.center.flatten(), [3, 4, 1])
        moved = np.eye(4)
        moved[:3, 3] = 5
        second.transform(moved)
        np.testing.assert_allclose(submap.segments[1].center.flatten(), [1, -2, 1], atol=1e-12)
        builder.update_frame(5., pose)
        self.assertEqual(builder.update_segment(segment(2, t0=4, tf=6)).id, 1)

    def test_mean_semantic_descriptor(self):
        builder = SubmapBuilder(2, 0, SubmapParams(submap_descriptor='mean_semantic'))
        builder.update_frame(1., np.eye(4))
        builder.update_segment(segment(0, descriptor=np.array([1., 0.])))
        submap = builder.update_segment(segment(1, descriptor=np.array([0., 1.])))
        np.testing.assert_allclose(submap.descriptor, np.ones(2) / np.sqrt(2))

    def test_frame_descriptors_sorted_and_selected_by_segment_time_span(self):
        for mode in ('mean_frame_descriptor', 'stacked_frame_descriptors'):
            with self.subTest(mode=mode):
                builder = SubmapBuilder(2, 0, SubmapParams(submap_descriptor=mode))
                for time, descriptor in ((3., [99., 99.]), (2., [0., 2.]), (0., [2., 0.])):
                    builder.update_frame(time, np.eye(4), np.array(descriptor))
                builder.update_segment(segment(0))
                submap = builder.update_segment(segment(1))
                expected = [1, 1] if mode == 'mean_frame_descriptor' else [[2, 0], [0, 2]]
                np.testing.assert_equal(submap.descriptor, expected)
                self.assertEqual(list(builder.poses), [3.])

    def test_distance_gated_frame_descriptors(self):
        builder = SubmapBuilder(2, 1, SubmapParams(
            submap_descriptor='stacked_frame_descriptors', frame_descriptor_dist=10))
        for time, distance in ((0., 0), (1., 1), (2., 12)):
            pose = np.eye(4)
            pose[0, 3] = distance
            builder.update_frame(time, pose, np.array([time, 1.]))
        builder.update_segment(segment(0))
        submap = builder.update_segment(segment(1))
        np.testing.assert_equal(submap.descriptor, [[0, 1], [2, 1]])

    def test_pose_required_and_empty_descriptor_buffer(self):
        builder = SubmapBuilder(1, 0, SubmapParams(submap_descriptor='mean_frame_descriptor'))
        with self.assertRaises(ValueError):
            builder.update_segment(segment(0))
        self.assertEqual(builder.segment_queue.segments, {})
        builder.update_frame(1., np.eye(4))
        self.assertIsNone(builder.update_segment(segment(0)).descriptor)


class SubmapCallbackTests(unittest.TestCase):
    def test_mapper_publishes_segments_and_local_submap(self):
        from roman_ros2.roman_map_node import RomanMapNode
        from roman_ros2.utils import msg_to_submap

        builder = SubmapBuilder(2, 0, SubmapParams())
        builder.update_frame(1., np.eye(4))
        node = SimpleNamespace(object_ref='center', robot_id=7, odom_frame_id='robot/odom', base_flu_frame_id='robot/base_link',
                               segments_pub=Mock(), submap_pub=Mock(), submap_builder=builder,
                               submap_params=builder.params,
                               mapper=SimpleNamespace(params=SimpleNamespace(store_aggregated_pcds=False)))
        RomanMapNode.publish_segment(node, segment(0))
        RomanMapNode.publish_segment(node, segment(1))
        self.assertEqual(node.segments_pub.publish.call_count, 2)
        node.submap_pub.publish.assert_called_once()
        msg = node.submap_pub.publish.call_args.args[0]
        self.assertEqual((msg.robot_id, msg.submap_id, msg.header.frame_id), (7, 0, 'robot/base_link'))
        self.assertEqual(msg.odom_frame_id, 'robot/odom')
        self.assertEqual(len(msg_to_submap(msg).segments), 2)

    def test_empty_observations_still_record_pose_and_descriptor(self):
        import geometry_msgs.msg as geometry_msgs
        import roman_msgs.msg as roman_msgs
        import ros2_numpy as rnp
        from roman_ros2.roman_map_node import RomanMapNode
        from roman_ros2.utils import descriptor_to_array_msg

        builder = SubmapBuilder(2, 1, SubmapParams())
        node = SimpleNamespace(up=True, mapper=SimpleNamespace(
            segments=[], inactive_segments=[], segment_graveyard=[]), log_and_send_status=Mock(),
            timing_fifo=[], pulse_pub=Mock(), submap_builder=builder)
        msg = roman_msgs.ObservationArray(
            pose_flu=rnp.msgify(geometry_msgs.Pose, np.eye(4)),
            frame_descriptor=descriptor_to_array_msg(np.array([1., 2.])), observations=[])
        RomanMapNode.obs_cb(node, msg)
        np.testing.assert_equal(builder.poses[0.], np.eye(4))
        np.testing.assert_equal(builder.descriptors[0.], [1., 2.])

    def test_loop_closure_subscribes_only_to_ego_and_team_submaps(self):
        import roman_msgs.msg as roman_msgs
        from roman_ros2.roman_loop_closure_node import ROMANLoopClosureNode

        node = SimpleNamespace(live_names=['ego', 'team'], live_ids=[7, 8],
                               create_publisher=Mock(), create_subscription=Mock(), submap_cb=Mock())
        ROMANLoopClosureNode.setup_ros(node)
        calls = node.create_subscription.call_args_list
        self.assertEqual([call.args[1] for call in calls], ['/ego/roman/submap', '/team/roman/submap'])
        self.assertTrue(all(call.args[0] is roman_msgs.Submap for call in calls))
        for call, robot_id in zip(calls, node.live_ids):
            call.args[2]('message')
            node.submap_cb.assert_called_with('message', robot_id)

    def test_received_submap_triggers_registration_and_is_stored(self):
        from roman.map.map import Submap
        from roman_ros2.roman_loop_closure_node import ROMANLoopClosureNode, LoopClosureInfo
        from roman_ros2.utils import submap_to_msg

        node = SimpleNamespace(last_lc_time=None, last_lc_info=LoopClosureInfo(),
                               send_status_msg=Mock(), submap_align_params=SimpleNamespace(submap_descriptor=None),
                               ego_id=7, all_ids=[7, 8, 9], submaps={7: [], 8: [], 9: []},
                               run_submap_registration=Mock(), log_and_send_status=Mock(), new_lc_flag=False)
        node.handle_new_submap = lambda submap, robot_id: ROMANLoopClosureNode.handle_new_submap(
            node, submap, robot_id)
        for robot_id, expected_peers in ((7, [7, 8, 9]), (8, [7])):
            node.run_submap_registration.reset_mock()
            msg = submap_to_msg(robot_id, Submap(0, 1., [segment(0)], np.eye(4)), 'odom', 'base_link')
            ROMANLoopClosureNode.submap_cb(node, msg, robot_id)
            self.assertEqual(len(node.submaps[robot_id]), 1)
            self.assertEqual([call.args[2] for call in node.run_submap_registration.call_args_list],
                             expected_peers)

    def test_registration_keeps_flu_poses_when_converting_gravity_frame_result(self):
        from roman.map.map import Submap
        from roman.utils import transform_rm_roll_pitch
        from roman_ros2.roman_loop_closure_node import ROMANLoopClosureNode

        poses = []
        for yaw, pitch, roll in ((0.8, 0.2, 0.1), (-0.1, -0.3, 0.2)):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
            poses.append(pose)
        submaps = [Submap(0, 1., [segment(i) for i in range(4)], pose.copy()) for pose in poses]
        registration_transform = np.eye(4)
        registration_transform[0, 3] = 0.4
        registration = Mock()
        registration.register.return_value = np.column_stack((np.arange(4), np.arange(4)))
        registration.T_align.return_value = registration_transform
        node = SimpleNamespace(submaps={0: [submaps[0]], 1: [submaps[1]]}, k_latest_submaps=-1,
                               submap_knn=None, submap_align_params=SimpleNamespace(
                                   submap_descriptor=None, icp_on_submap_pcds=False,
                                   force_rm_lc_roll_pitch=False), roman_reg=registration,
                               lc_required_associations=4, log_and_send_status=Mock(),
                               covariance=np.eye(6), get_clock=Mock(), pgt_lc_pub=Mock(),
                               loop_closure_pub=Mock())
        with patch('roman_ros2.roman_loop_closure_node.lc_to_pose_graph_msg') as to_pose_graph, \
                patch('roman_ros2.roman_loop_closure_node.lc_to_msg'):
            ROMANLoopClosureNode.run_submap_registration(node, submaps[0], 0, 1)
        expected = (np.linalg.inv(poses[0]) @ transform_rm_roll_pitch(poses[0]) @
                    registration_transform @ np.linalg.inv(transform_rm_roll_pitch(poses[1])) @ poses[1])
        np.testing.assert_allclose(to_pose_graph.call_args.args[4], expected)
        for submap, pose in zip(submaps, poses):
            np.testing.assert_equal(submap.pose_flu, pose)


if __name__ == '__main__':
    unittest.main()
