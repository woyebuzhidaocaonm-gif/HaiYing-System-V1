#!/usr/bin/env python3
"""
V9.2 视觉链 5 项离线测试（纯 numpy，无 ROS 依赖）
=================================================

覆盖 PM 冻结交付物⑦要求的 5 项：
  1. ar0234_camera_link(FLU) → optical(RDF) 坐标转换正确性，
     并与 V9.2 静态 TF rpy(-pi/2, 0, -pi/2) 的旋转矩阵做等价性断言
  2. CameraInfo 内参 K 投影与相机模型一致性（合成 K）
  3. lidar 深度查询（墙面点云取中位数 / 空区域返回 None / 离群点稳健）
  4. 图像-点云 0.5s 时间差拒绝（stamps_within_skew 边界）
  5. PointCloud2 解析（字段乱序 / gz 12 字节布局 / 非对齐慢路径 / 异常防御）

运行（无需 ROS，仅需 numpy）:
    python3 tests/test_v9_2_localization.py
    python3 tests/test_v9_2_localization.py -v
"""
import math
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_projection import (  # noqa: E402
    LINK_TO_OPTICAL_R,
    link_to_optical,
    optical_to_link,
    pc2_to_xyz,
    project_optical_to_pixel,
    query_lidar_depth,
    stamps_within_skew,
)


# ---------------------------------------------------------------------------
# 离线替身：模仿 sensor_msgs 消息的最小接口（鸭子类型）
# ---------------------------------------------------------------------------
class FakeField:
    def __init__(self, name, offset, datatype=7):  # 7 = FLOAT32
        self.name = name
        self.offset = offset
        self.datatype = datatype


class FakePC2:
    def __init__(self, data, point_step, fields):
        self.data = data
        self.point_step = point_step
        self.fields = fields


class FakeStamp:
    def __init__(self, sec=0, nanosec=0):
        self.sec = sec
        self.nanosec = nanosec


class TestLinkToOptical(unittest.TestCase):
    """测试 1: link(FLU) → optical(RDF) 坐标转换"""

    def test_axis_mapping(self):
        # FLU 前(+X) → RDF 前(+Z)
        np.testing.assert_allclose(
            link_to_optical(np.array([[1., 0., 0.]])), [[0., 0., 1.]])
        # FLU 左(+Y) → RDF 右(-X)
        np.testing.assert_allclose(
            link_to_optical(np.array([[0., 1., 0.]])), [[-1., 0., 0.]])
        # FLU 上(+Z) → RDF 下(-Y)
        np.testing.assert_allclose(
            link_to_optical(np.array([[0., 0., 1.]])), [[0., -1., 0.]])

    def test_equivalent_to_v9_2_static_tf(self):
        # V9.2 launch 静态 TF: rpy=(-pi/2, 0, -pi/2) → R = Rz(-pi/2) @ Rx(-pi/2)
        rz = np.array([[math.cos(-math.pi / 2), -math.sin(-math.pi / 2), 0.],
                       [math.sin(-math.pi / 2), math.cos(-math.pi / 2), 0.],
                       [0., 0., 1.]])
        rx = np.array([[1., 0., 0.],
                       [0., math.cos(-math.pi / 2), -math.sin(-math.pi / 2)],
                       [0., math.sin(-math.pi / 2), math.cos(-math.pi / 2)]])
        tf_r = rz @ rx  # 静态 TF 发布 link→optical 的旋转
        np.testing.assert_allclose(LINK_TO_OPTICAL_R, tf_r.T, atol=1e-12)

    def test_rotation_orthonormal(self):
        np.testing.assert_allclose(
            LINK_TO_OPTICAL_R.T @ LINK_TO_OPTICAL_R, np.eye(3), atol=1e-12)

    def test_roundtrip(self):
        pts = np.random.default_rng(42).uniform(-10, 10, (100, 3))
        np.testing.assert_allclose(
            optical_to_link(link_to_optical(pts)), pts, atol=1e-12)


class TestProjection(unittest.TestCase):
    """测试 2: CameraInfo 内参 K 投影与相机模型一致性"""

    K = np.array([[1000., 0., 960.],
                  [0., 1000., 540.],
                  [0., 0., 1.]])

    def test_known_point(self):
        # optical 系点 (0.96, 0.54, 2.0): u = fx*x/z + cx = 1440, v = 810
        uv = project_optical_to_pixel(np.array([[0.96, 0.54, 2.0]]), self.K)
        np.testing.assert_allclose(uv, [[1440., 810.]], atol=1e-9)

    def test_optical_axis_point(self):
        # 光轴上的点投影到主点 (cx, cy)
        uv = project_optical_to_pixel(np.array([[0., 0., 5.0]]), self.K)
        np.testing.assert_allclose(uv, [[960., 540.]], atol=1e-9)

    def test_behind_or_degenerate_camera(self):
        pts = np.array([[1., 1., -1.],    # 相机后方
                        [1., 1., 0.],     # Z=0 退化
                        [1., 1., 1e-9]])  # Z≈0
        uv = project_optical_to_pixel(pts, self.K)
        np.testing.assert_array_equal(uv, np.full((3, 2), -1.0))

    def test_unproject_roundtrip(self):
        rng = np.random.default_rng(7)
        z = rng.uniform(0.5, 30.0, (50, 1))
        x = rng.uniform(-3, 3, (50, 1))
        y = rng.uniform(-2, 2, (50, 1))
        pts = np.hstack([x, y, z])
        uv = project_optical_to_pixel(pts, self.K)
        # 反投影还原
        back_x = (uv[:, 0] - self.K[0, 2]) * z[:, 0] / self.K[0, 0]
        back_y = (uv[:, 1] - self.K[1, 2]) * z[:, 0] / self.K[1, 1]
        np.testing.assert_allclose(back_x, x[:, 0], atol=1e-9)
        np.testing.assert_allclose(back_y, y[:, 0], atol=1e-9)

    def test_empty_input(self):
        uv = project_optical_to_pixel(np.zeros((0, 3)), self.K)
        self.assertEqual(uv.shape, (0, 2))


class TestLidarDepth(unittest.TestCase):
    """测试 3: lidar 深度查询（墙面点云 / 空区域 / 离群点稳健）"""

    K = np.array([[500., 0., 320.],
                  [0., 500., 240.],
                  [0., 0., 1.]])

    @staticmethod
    def _wall(K, z_wall, span_x=0.64, span_y=0.48, step=0.032):
        """z=z_wall 处平行墙面的均匀网格点（optical 系）。"""
        xs = np.arange(-span_x, span_x + 1e-9, step)
        ys = np.arange(-span_y, span_y + 1e-9, step)
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel(),
                         np.full(gx.size, z_wall)], axis=1)

    def test_wall_median_depth(self):
        # (0,0,z) 投影到主点 (320,240)；墙面 z=5 → 查询返回中位数 5
        pts = self._wall(self.K, 5.0)
        depth = query_lidar_depth(pts, 320.0, 240.0, self.K, radius=15.0)
        self.assertIsNotNone(depth)
        self.assertAlmostEqual(depth, 5.0, places=9)

    def test_outlier_robustness(self):
        # 墙面 z=5 内混入 z=2 / z=50 离群点 → 中位数+±50% 窗口取最近 → 仍为 5
        pts = self._wall(self.K, 5.0)
        outliers = np.array([[0.05, 0.05, 2.0],
                             [-0.05, -0.05, 50.0],
                             [0.1, 0.0, 0.2]])  # 0.2 < 0.1m 量程下限被过滤
        depth = query_lidar_depth(
            np.vstack([pts, outliers]), 320.0, 240.0, self.K, radius=15.0)
        self.assertAlmostEqual(depth, 5.0, places=9)

    def test_empty_region_returns_none(self):
        # 墙面覆盖主点附近 ±32 像素，查询 (600, 400) 远处 → None
        pts = self._wall(self.K, 5.0)
        self.assertIsNone(
            query_lidar_depth(pts, 600.0, 400.0, self.K, radius=15.0))
        # 空点云 → None
        self.assertIsNone(
            query_lidar_depth(np.zeros((0, 3)), 320.0, 240.0, self.K))

    def test_nan_points_filtered(self):
        pts = self._wall(self.K, 5.0)
        pts = np.vstack([pts, [[np.nan, np.nan, np.nan]]])
        depth = query_lidar_depth(pts, 320.0, 240.0, self.K, radius=15.0)
        self.assertAlmostEqual(depth, 5.0, places=9)


class TestTimeSkew(unittest.TestCase):
    """测试 4: 图像-点云 0.5s 时间差拒绝（PM 冻结边界）"""

    def test_within_boundary(self):
        # 恰好 0.5s → 接受（<= 边界）
        self.assertTrue(stamps_within_skew(
            FakeStamp(10, 0), FakeStamp(10, 500000000), 0.5))

    def test_exceeds_boundary(self):
        # 0.5s + 1ns → 拒绝
        self.assertFalse(stamps_within_skew(
            FakeStamp(10, 0), FakeStamp(10, 500000001), 0.5))
        # 1.0s 差 → 拒绝
        self.assertFalse(stamps_within_skew(
            FakeStamp(10, 0), FakeStamp(11, 0), 0.5))

    def test_negative_direction(self):
        # 点云早于图像 0.4s → |差| 判据接受
        self.assertTrue(stamps_within_skew(
            FakeStamp(20, 0), FakeStamp(19, 600000000), 0.5))

    def test_nanosec_rollover(self):
        # sec 进位不破坏换算: 10.9s vs 11.4s → 差 0.5 → 接受
        self.assertTrue(stamps_within_skew(
            FakeStamp(10, 900000000), FakeStamp(11, 400000000), 0.5))

    def test_zero_stamp_rejected(self):
        # 零时间戳 = 采集时间不可考 → 拒绝
        self.assertFalse(stamps_within_skew(
            FakeStamp(0, 0), FakeStamp(10, 0), 0.5))
        self.assertFalse(stamps_within_skew(
            FakeStamp(10, 0), FakeStamp(0, 0), 0.5))
        # None → 拒绝
        self.assertFalse(stamps_within_skew(None, FakeStamp(10, 0), 0.5))
        self.assertFalse(stamps_within_skew(FakeStamp(10, 0), None, 0.5))


class TestPC2Parse(unittest.TestCase):
    """测试 5: PointCloud2 解析（字段乱序 / gz 布局 / 慢路径 / 防御）"""

    def test_scrambled_field_order(self):
        # 字段顺序与内存布局不一致（z 在最前），仍按 offset 正确解析
        pts = np.array([[1.0, 2.0, 3.0],
                        [-4.0, 5.5, -6.25]])
        buf = bytearray()
        for p in pts:
            row = bytearray(16)
            struct.pack_into('<ffff', row, 0, *p, 42.0)  # 末位 intensity
            buf += row
        pc2 = FakePC2(bytes(buf), 16, [
            FakeField('z', 8), FakeField('intensity', 12),
            FakeField('x', 0), FakeField('y', 4)])
        np.testing.assert_allclose(pc2_to_xyz(pc2), pts)

    def test_gz_12_byte_layout(self):
        # gz 传感器 PointCloudPacked 常见紧凑布局: xyz 各 4 字节，无 intensity
        pts = np.array([[0.5, -0.5, 10.0],
                        [1.25, 2.5, 7.75]])
        buf = bytearray()
        for p in pts:
            buf += struct.pack('<fff', *p)
        pc2 = FakePC2(bytes(buf), 12, [
            FakeField('x', 0), FakeField('y', 4), FakeField('z', 8)])
        np.testing.assert_allclose(pc2_to_xyz(pc2), pts)

    def test_unaligned_slow_path(self):
        # 非 4 字节对齐布局（point_step=13, z 从 offset 9 起）→ 逐点解析
        pts = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        buf = bytearray()
        for p in pts:
            row = bytearray(13)
            # x@0, y@4, 1 字节填充@8, z@9（非 4 字节对齐，强制慢路径）
            struct.pack_into('<ffbf', row, 0, p[0], p[1], 0, p[2])
            buf += row
        pc2 = FakePC2(bytes(buf), 13, [
            FakeField('x', 0), FakeField('y', 4), FakeField('z', 9)])
        np.testing.assert_allclose(pc2_to_xyz(pc2), pts)

    def test_unsupported_datatype_raises(self):
        pc2 = FakePC2(b'\x00' * 32, 16, [
            FakeField('x', 0, datatype=2),  # UINT8 → 不支持
            FakeField('y', 4), FakeField('z', 8)])
        with self.assertRaises(ValueError):
            pc2_to_xyz(pc2)

    def test_missing_fields_raises(self):
        pc2 = FakePC2(b'\x00' * 32, 16, [FakeField('intensity', 0)])
        with self.assertRaises(ValueError):
            pc2_to_xyz(pc2)

    def test_empty_cloud(self):
        pc2 = FakePC2(b'', 16, [FakeField('x', 0), FakeField('y', 4),
                                FakeField('z', 8)])
        out = pc2_to_xyz(pc2)
        self.assertEqual(out.shape, (0, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
