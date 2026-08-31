#!/usr/bin/env python3
"""
Gazebo LiDAR → ROS2 PointCloud2 桥接器（V9.2 帧名适配）
=====================================================
注意：V9.2 正式链由 haiying_v9_2 模型的 gazebo_ros_ray_sensor 插件
直接发布 /drone/lidar/points（frame=mid360_link），**不启动本桥接器**。
本脚本保留用于旧 gz-sensors 链路 / 单线雷达实物适配。

V9.2 适配要点：
  - 输出 frame_id = mid360_link（与 V9.2 雷达帧一致）
  - header.stamp 透传 Gazebo 仿真时间（不再用 now() 覆盖，
    否则 target_localizer 的 0.5s 图像-点云时间差检查失效）
  - 同时订阅 LaserScan 与 PointCloudPacked 两种 gz 消息，
    按仿真时间戳去重（gpu_lidar 同一 tick 会双发）
  - 修复 PointCloudPacked 解析：按 gz 消息自带 point_step 逐点读取
    xyz，重新打包为 xyz+intensity(0) 的 16 字节 ROS PointCloud2

用法:
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    python3 lidar_bridge.py

环境变量:
    GZ_LIDAR_TOPIC:  Gazebo LiDAR话题 (默认 /lidar)
    ROS_LIDAR_TOPIC: ROS2输出话题 (默认 /drone/lidar/points)
    ROS_LIDAR_FRAME: 输出 frame_id (默认 mid360_link)
"""
import os
import time
import struct
import ctypes
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np

from gz.transport13 import Node as GzNode
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked

GZ_LIDAR_TOPIC = os.environ.get('GZ_LIDAR_TOPIC', '/lidar')
ROS_LIDAR_TOPIC = os.environ.get('ROS_LIDAR_TOPIC', '/drone/lidar/points')
ROS_LIDAR_FRAME = os.environ.get('ROS_LIDAR_FRAME', 'mid360_link')


def _gz_stamp_to_ros(msg, node):
    """透传 Gazebo 仿真时间戳；时间戳为零时回退 ROS 时钟。"""
    stamp = msg.header.stamp
    if stamp.sec > 0 or stamp.nsec > 0:
        ros_time = rclpy.time.Time(
            seconds=stamp.sec, nanoseconds=stamp.nsec)
        return ros_time.to_msg()
    return node.get_clock().now().to_msg()


class GzLidarBridge(Node):
    """Gazebo LiDAR → ROS2 PointCloud2"""

    def __init__(self):
        super().__init__('lidar_bridge')
        self.pc_pub = self.create_publisher(PointCloud2, ROS_LIDAR_TOPIC, 10)
        self.gz_node = GzNode()
        self.frame_count = 0
        self.start_time = time.time()
        # 同一仿真 tick 的 LaserScan/PointCloudPacked 双发去重
        self._last_stamp = None

        # 同时订阅两种 gz 消息（gpu_lidar 在 <topic> 发 LaserScan、
        # 在 <topic>/points 发 PointCloudPacked，同一 tick 同时间戳）
        self._subscribe_pointcloud(GZ_LIDAR_TOPIC + '/points')
        self._subscribe_laserscan(GZ_LIDAR_TOPIC)

        self.get_logger().info(
            f'LiDAR桥接器已启动: {GZ_LIDAR_TOPIC} → {ROS_LIDAR_TOPIC} '
            f'(frame={ROS_LIDAR_FRAME}, 双订阅+时间戳去重)')

    def _is_duplicate(self, msg):
        stamp = msg.header.stamp
        key = (stamp.sec, stamp.nsec)
        if self._last_stamp == key:
            return True
        self._last_stamp = key
        return False

    def _make_pc2(self, stamp, points):
        """points: (N,4) ndarray [x,y,z,intensity] → ROS PointCloud2"""
        pc2 = PointCloud2()
        pc2.header.stamp = stamp
        pc2.header.frame_id = ROS_LIDAR_FRAME
        pc2.height = 1
        pc2.width = len(points)
        pc2.is_dense = True
        pc2.is_bigendian = False
        pc2.point_step = 16  # xyz(float32) + intensity(float32)
        pc2.row_step = pc2.point_step * pc2.width
        pc2.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc2.data = np.asarray(points, dtype=np.float32).tobytes()
        return pc2

    def _subscribe_pointcloud(self, topic):
        """订阅 PointCloudPacked（优先，3D 完整）"""
        def on_pointcloud(msg):
            if self._is_duplicate(msg):
                return
            try:
                data = msg.data
                point_step = msg.point_step
                n = len(data) // point_step
                if n == 0:
                    return
                # gz PointCloudPacked 布局: 每点前 12 字节为 x,y,z (float32)
                if point_step < 12:
                    self.get_logger().warn(
                        f'PointCloudPacked point_step={point_step} 异常，跳过')
                    return
                pts = np.zeros((n, 4), dtype=np.float32)
                for i in range(n):
                    base = i * point_step
                    (pts[i, 0], pts[i, 1], pts[i, 2]) = struct.unpack_from(
                        '<fff', data, base)
                pc2 = self._make_pc2(_gz_stamp_to_ros(msg, self), pts)
                self.pc_pub.publish(pc2)
                self._tick(n)
            except Exception as e:
                self.get_logger().error(f'点云处理错误: {e}')

        try:
            self.gz_node.subscribe(PointCloudPacked, topic, on_pointcloud)
            self.get_logger().info(f'订阅Gazebo点云: {topic}')
        except Exception as e:
            self.get_logger().warn(f'PointCloudPacked订阅失败({topic}): {e}')

    def _subscribe_laserscan(self, topic):
        """订阅 LaserScan（降级方案，单线 2D）"""
        def on_laserscan(msg):
            if self._is_duplicate(msg):
                return
            try:
                ranges = np.array(msg.ranges, dtype=np.float32)
                angle_min = msg.angle_min
                angle_step = msg.angle_step
                pts = []
                for i, r in enumerate(ranges):
                    if msg.range_min < r < msg.range_max:
                        angle = angle_min + i * angle_step
                        pts.append([r * np.cos(angle), r * np.sin(angle), 0.0, 0.0])
                if not pts:
                    return
                pc2 = self._make_pc2(
                    _gz_stamp_to_ros(msg, self), np.asarray(pts, dtype=np.float32))
                self.pc_pub.publish(pc2)
                self._tick(len(pts))
            except Exception as e:
                self.get_logger().error(f'LaserScan处理错误: {e}')

        try:
            self.gz_node.subscribe(LaserScan, topic, on_laserscan)
            self.get_logger().info(f'订阅Gazebo激光扫描: {topic}')
        except Exception as e:
            self.get_logger().warn(f'LaserScan订阅失败({topic}): {e}')

    def _tick(self, n_points):
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.get_logger().info(f'LiDAR: {self.frame_count}帧 | {fps:.1f}FPS | {n_points}点')


def main():
    rclpy.init()
    node = GzLidarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
