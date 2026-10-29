import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import sensor_msgs.msg as sensor_msgs
from scipy.spatial.transform import Rotation

from roman_ros2.roman_map_node import RomanMapNode, deskew_points, pointcloud2_to_xyz_time
from roman_ros2.utils import float_to_ros_time


def cloud_msg(points, times=None, time_field='timestamp', stamp=100.0):
    """PointCloud2 laid out like a Livox (x, y, z, intensity, timestamp f8 ns) or Ouster (t u4 ns)."""
    fields = [sensor_msgs.PointField(name=n, offset=4 * i, datatype=sensor_msgs.PointField.FLOAT32, count=1)
              for i, n in enumerate(['x', 'y', 'z', 'intensity'])]
    formats = ['<f4'] * 4
    if times is not None:
        fmt = '<f8' if time_field == 'timestamp' else '<u4'
        datatype = sensor_msgs.PointField.FLOAT64 if fmt == '<f8' else sensor_msgs.PointField.UINT32
        fields.append(sensor_msgs.PointField(name=time_field, offset=16, datatype=datatype, count=1))
        formats.append(fmt)
    dtype = np.dtype({'names': [f.name for f in fields], 'formats': formats,
                      'offsets': [f.offset for f in fields], 'itemsize': 32})
    data = np.zeros(len(points), dtype=dtype)
    for i, n in enumerate('xyz'):
        data[n] = points[:, i]
    if times is not None:
        data[time_field] = times * 1e9 if time_field == 'timestamp' else (times - stamp) * 1e9
    msg = sensor_msgs.PointCloud2(height=1, width=len(points), fields=fields, is_bigendian=False,
                                  point_step=32, row_step=32 * len(points), data=data.tobytes(), is_dense=True)
    msg.header.stamp = float_to_ros_time(stamp)
    msg.header.frame_id = 'lidar'
    return msg


def pose(x=0.0, yaw=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    T[0, 3] = x
    return T


class PointCloudSourceTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1., 2., 3.], [4., 5., 6.], [-1., 0., 2.]])
        self.times = np.array([100.0, 100.05, 100.1])

    def test_parses_livox_and_ouster_times(self):
        for field in ['timestamp', 't']:
            points, times = pointcloud2_to_xyz_time(cloud_msg(self.points, self.times, field))
            np.testing.assert_allclose(points, self.points, atol=1e-6)
            np.testing.assert_allclose(times, self.times, atol=1e-6)
        points, times = pointcloud2_to_xyz_time(cloud_msg(self.points))
        np.testing.assert_allclose(points, self.points, atol=1e-6)
        self.assertIsNone(times)

    def test_deskew_moves_each_point_with_the_sensor(self):
        # sensor moves +1 m in x and turns 0.2 rad over the scan; the world points are fixed
        pose_at = lambda t: pose(x=(t - 100.0) * 10.0, yaw=(t - 100.0) * 2.0)
        world = np.array([[5., 0., 0.], [5., 1., 0.], [5., 2., 0.], [5., 3., 0.]])
        times = np.array([100.0, 100.033, 100.066, 100.1])
        steps = 1000
        # what the sensor measured: each world point in the sensor frame at its own time
        raw = np.array([np.linalg.inv(pose_at(t)) @ np.append(p, 1.) for p, t in zip(world, times)])[:, :3]
        out = deskew_points(raw, times, 100.0, pose_at, steps)
        np.testing.assert_allclose(out, world, atol=0.02)   # sensor frame at t_ref = odom here
        np.testing.assert_allclose(deskew_points(raw, times, 100.0, pose_at, 0), raw)

    def test_callback_uses_mount_override_and_min_range(self):
        T_base_sensor = np.eye(4)
        T_base_sensor[:3, :3] = Rotation.from_euler('x', np.pi).as_matrix()   # upside-down lidar
        T_base_sensor[:3, 3] = [-0.2, 0., 0.45]
        T_odom_base = pose(x=3.0, yaw=0.5)
        tf_buffer = Mock()
        tf_buffer.lookup_transform.return_value = SimpleNamespace(transform=None)
        node = SimpleNamespace(up=True, last_pointcloud_time=-np.inf, pointcloud_min_range=1.5,
                               pointcloud_deskew_steps=4, T_base_pcd_sensor=T_base_sensor,
                               odom_frame_id='odom', base_flu_frame_id='base', tf_buffer=tf_buffer,
                               mapper=Mock(), get_logger=Mock())
        node.pointcloud_sensor_pose = lambda frame, t: RomanMapNode.pointcloud_sensor_pose(node, frame, t)
        # every TF lookup returns T_odom_base: a stationary robot
        with patch('roman_ros2.roman_map_node.rnp.numpify', return_value=T_odom_base):
            RomanMapNode.pointcloud_cb(node, cloud_msg(self.points, self.times))
            t, T_odom_sensor, points = node.mapper.update_point_cloud.call_args.args
            self.assertEqual(t, 100.0)
            np.testing.assert_allclose(T_odom_sensor, T_odom_base @ T_base_sensor)
            self.assertEqual(tf_buffer.lookup_transform.call_args.args[:2], ('odom', 'base'))
            # ranges are 3.7, 8.8 and 2.2 m: all kept at 1.5 m, and deskewing a still robot changes nothing
            np.testing.assert_allclose(points, self.points, atol=1e-6)
            node.pointcloud_min_range = 3.0
            node.last_pointcloud_time = -np.inf
            RomanMapNode.pointcloud_cb(node, cloud_msg(self.points, self.times))
            np.testing.assert_allclose(node.mapper.update_point_cloud.call_args.args[2], self.points[:2], atol=1e-6)

if __name__ == '__main__':
    unittest.main()
