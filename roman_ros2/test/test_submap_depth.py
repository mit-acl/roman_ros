import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv_bridge
import geometry_msgs.msg as geometry_msgs
import numpy as np
import sensor_msgs.msg as sensor_msgs
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation

from roman.map.map import Submap, SubmapParams, AggregatedPointCloud
from roman.map.mapper import Mapper
from roman.params.mapper_params import MapperParams
from roman.object.segment import SegmentMinimalData
from roman_ros2.roman_map_node import RomanMapNode
from roman_ros2.submap_builder import SubmapBuilder
from roman_ros2.utils import submap_to_msg, msg_to_submap, float_to_ros_time


def segment():
    return SegmentMinimalData(0, np.zeros(3), 1., .1, .2, .3, None, None, 1., 2.)


class SubmapDepthTests(unittest.TestCase):
    def test_optional_cloud_and_tilted_frame_round_trip(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('xyz', [.3, -.4, .6]).as_matrix()
        pose[:3, 3] = [1., 2., 3.]
        submap = Submap(0, 1., [segment()], pose)
        self.assertEqual(len(submap_to_msg(0, submap, 'odom', 'base').point_cloud), 0)
        self.assertIsNone(msg_to_submap(submap_to_msg(0, submap, 'odom', 'base')).point_cloud)
        for points in [np.empty((0, 3)), np.array([[1., 2., 3.], [4., 5., 6.]])]:
            submap.point_cloud = points
            msg = submap_to_msg(0, submap, 'odom', 'base')
            self.assertEqual(len(msg.point_cloud), 1)
            self.assertEqual(msg.point_cloud[0].header.frame_id, 'base')
            restored = msg_to_submap(msg)
            np.testing.assert_allclose(restored.point_cloud, points, atol=1e-6)
            wire = point_cloud2.read_points_numpy(msg.point_cloud[0]).reshape(-1, 3)
            if len(points):
                np.testing.assert_allclose(wire @ pose[:3, :3].T + pose[:3, 3],
                    points @ submap.pose_gravity_aligned[:3, :3].T + submap.pose_gravity_aligned[:3, 3], atol=1e-6)

    def test_published_submap_merges_overlapping_windows_only_when_enabled(self):
        for enabled in [False, True]:
            builder = SubmapBuilder(1, 0, SubmapParams())
            pose = np.eye(4)
            pose[0, 3] = 1.
            builder.update_frame(1., pose)
            mapper = SimpleNamespace(params=SimpleNamespace(store_aggregated_pcds=enabled),
                aggregated_point_clouds=[
                    AggregatedPointCloud(0., 1., np.array([[2., 0., 0.]])),
                    AggregatedPointCloud(2., 3., np.array([[2.001, 0., 0.]])),
                    AggregatedPointCloud(3., 4., np.array([[99., 0., 0.]]))])
            node = SimpleNamespace(object_ref='center', robot_id=0, odom_frame_id='odom',
                base_flu_frame_id='base', segments_pub=Mock(), submap_pub=Mock(),
                submap_builder=builder, submap_params=builder.params, mapper=mapper)
            RomanMapNode.publish_segment(node, segment())
            msg = node.submap_pub.publish.call_args.args[0]
            self.assertEqual(len(msg.point_cloud), int(enabled))
            if enabled:
                points = msg_to_submap(msg).point_cloud
                self.assertEqual(points.shape, (1, 3))
                self.assertAlmostEqual(points[0, 0], 1.0005, places=5)

    def test_depth_subscriptions_are_conditional_and_share_mapper_callback_group(self):
        for enabled, keyframe in [(False, False), (True, False), (False, True), (True, True)]:
            node = SimpleNamespace(mapper=SimpleNamespace(params=SimpleNamespace(store_aggregated_pcds=enabled)),
                create_publisher=Mock(), create_subscription=Mock(), create_service=Mock(),
                obs_cb=Mock(), depth_cb=Mock(), depth_info_cb=Mock(), keyframe_cb=Mock(), shutdown_cb=Mock(),
                reference_submap_by_keyframe=keyframe,
                visualize=False, viz_object_map=False, log_and_send_status=Mock())
            with patch('roman_ros2.roman_map_node.tf2_ros.Buffer'), \
                 patch('roman_ros2.roman_map_node.tf2_ros.TransformListener'):
                RomanMapNode.setup_ros(node)
            calls = node.create_subscription.call_args_list
            self.assertEqual([call.args[1] for call in calls],
                ['roman/observations'] + (['keyframe'] if keyframe else []) +
                (['depth/camera_info', 'depth/image_raw'] if enabled else []))
            for call in calls:
                self.assertIs(call.kwargs['callback_group'], node.obs_cb_group)

    def test_depth_units_projection_tf_and_windowing(self):
        bridge = cv_bridge.CvBridge()
        for encoding, depth in [('16UC1', np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)),
                                ('32FC1', np.array([[1., np.nan], [2., 3.]], dtype=np.float32))]:
            mapper = Mapper(MapperParams(store_aggregated_pcds=True, pcd_window_num_scans=1,
                pcd_window_overlap=0., pcd_dist_thresh_m=0., pcd_ang_thresh_deg=0.), camera_params=None)
            info = sensor_msgs.CameraInfo(width=2, height=2, k=[2., 0., 0., 0., 2., 0., 0., 0., 1.])
            info.header.frame_id = 'camera'
            transform = geometry_msgs.TransformStamped()
            transform.transform.rotation.w = 1.
            transform.transform.translation.x = 10.
            node = SimpleNamespace(up=True, depth_camera_info=info, last_depth_time=-np.inf,
                odom_frame_id='odom', depth_bridge=bridge, mapper=mapper, depth_pcd_stride=1,
                tf_buffer=Mock(), get_logger=Mock())
            node.tf_buffer.lookup_transform.return_value = transform
            msg = bridge.cv2_to_imgmsg(depth, encoding=encoding)
            msg.header.frame_id = 'camera'
            msg.header.stamp = float_to_ros_time(1.)
            RomanMapNode.depth_cb(node, msg)
            self.assertEqual(len(mapper.aggregated_point_clouds), 1)
            actual = mapper.aggregated_point_clouds[0].point_cloud
            expected = np.array([[10., 0., 1.], [10., 1., 2.], [11.5, 1.5, 3.]])
            np.testing.assert_allclose(actual[np.argsort(actual[:, 2])], expected)
            self.assertEqual(mapper.times_history, [])
            RomanMapNode.depth_cb(node, msg)
            self.assertEqual(len(mapper.aggregated_point_clouds), 1)
            node.depth_camera_info = None
            RomanMapNode.depth_cb(node, msg)
