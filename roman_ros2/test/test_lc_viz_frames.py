import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import geometry_msgs.msg as geometry_msgs
import tf2_ros

from roman.map.map import Submap
from roman_ros2.lc_viz_node import LCVizNode
from roman_ros2.utils import submap_to_msg, msg_to_submap


class LCVizFramesTest(unittest.TestCase):
    def test_frame_metadata_and_pose_round_trip(self):
        pose = np.eye(4)
        pose[0, 3] = 4.
        msg = submap_to_msg(2, Submap(0, 1., [], pose), 'robot/odom', 'robot/base_link')
        node = SimpleNamespace(odom_frames={}, flu_ref_frames={}, get_logger=Mock())
        LCVizNode.submap_cb(node, msg, 2)
        self.assertEqual(node.odom_frames, {2: 'robot/odom'})
        self.assertEqual(node.flu_ref_frames, {2: 'robot/base_link'})
        np.testing.assert_equal(msg_to_submap(msg).pose_flu, pose)
        LCVizNode.submap_cb(node, msg, 3)
        self.assertNotIn(3, node.odom_frames)

    def test_gravity_pose_is_derived_without_mutating_submap_pose(self):
        from scipy.spatial.transform import Rotation
        from roman.utils import transform_rm_roll_pitch
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
        submap = Submap(0, 1., [], pose)
        np.testing.assert_equal(submap.pose_flu, pose)
        np.testing.assert_allclose(submap.pose_gravity_aligned,
                                   transform_rm_roll_pitch(pose))
        msg = submap_to_msg(2, submap, 'odom', 'base_link')
        restored = msg_to_submap(msg)
        np.testing.assert_allclose(restored.pose_flu, pose)
        np.testing.assert_allclose(restored.pose_gravity_aligned, submap.pose_gravity_aligned)
        np.testing.assert_equal(submap.pose_flu, pose)
        np.testing.assert_allclose(restored.pose_flu, pose)

    def test_message_centroids_use_true_submap_frame(self):
        import ros2_numpy as rnp
        from scipy.spatial.transform import Rotation
        from roman.object.segment import SegmentMinimalData
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('xyz', [.3, -.4, .6]).as_matrix()
        pose[:3, 3] = [3., -2., 1.]
        center = np.array([1., 2., 3.])
        segment = SegmentMinimalData(4, center.copy(), 1., .1, .2, .3, None, None, 0., 2.)
        submap = Submap(0, 1., [segment], pose.copy())
        msg = submap_to_msg(2, submap, 'odom', 'base_link')
        wire_center = rnp.numpify(msg.segments[0].position).reshape(3)
        world_center = submap.pose_gravity_aligned @ np.r_[center, 1.]
        np.testing.assert_allclose(pose @ np.r_[wire_center, 1.], world_center)
        self.assertFalse(np.allclose(wire_center, center))
        restored = msg_to_submap(msg)
        np.testing.assert_allclose(restored.segments[0].center.reshape(3), center)
        np.testing.assert_equal(segment.center.reshape(3), center)
        np.testing.assert_equal(submap.pose_flu, pose)
        np.testing.assert_allclose(rnp.numpify(msg.segments[0].position).reshape(3), wire_center)

    def test_timer_skips_unknown_and_unavailable_robots_and_duplicate_tf(self):
        transform = geometry_msgs.TransformStamped()
        transform.header.stamp.sec = 5
        transform.transform.rotation.w = 1.
        buffer = Mock()
        buffer.lookup_transform.side_effect = [tf2_ros.LookupException('unavailable'), transform]
        node = SimpleNamespace(lc_marker_viz=False, live_ids=[0, 1, 2],
            odom_frames={1: 'one/odom', 2: 'two/odom'},
            flu_ref_frames={1: 'one/base_link', 2: 'two/base_link'},
            trajectories={0: None, 1: None, 2: None}, tf_buffer=buffer)
        LCVizNode.timer_cb(node)
        self.assertIsNone(node.trajectories[1])
        np.testing.assert_equal(node.trajectories[2].times, [5.])
        buffer.lookup_transform.side_effect = [tf2_ros.LookupException('unavailable'), transform]
        LCVizNode.timer_cb(node)
        self.assertEqual(len(node.trajectories[2].times), 1)
