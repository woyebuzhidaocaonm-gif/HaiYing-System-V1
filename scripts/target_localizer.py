#!/usr/bin/env python3
"""
目标3D坐标转换与发布节点 (Task 27, 72；V9.2 适配)
=================================================
收到 /vision/detection (2D bbox) → LiDAR点云投影定位 → TF变换 → /vision/target_point

数据流 (V9.2 正式链, PM 冻结):
  /vision/detection (DefectDetectionArray, header.stamp = 图像采集时间)
    + /drone/lidar/points (PointCloud2, frame = mid360_link)
    + /drone/camera/camera_info (CameraInfo, 内参 K)
    + TF: world → base_footprint → ar0234_camera_link → ar0234_camera_optical_frame
         (全部由 V9.2 launch / 仿真组提供；本节点不发布任何 TF)
    → /vision/target_point (PointStamped, frame = world, 单位米,
                            header.stamp = 图像采集时间)

接口底线 (PM 冻结):
  - 图像与点云最大时间差 0.5s，超过拒绝本次定位
  - 目标丢失时停止发布，绝不发 NaN（所有无效路径均 return，无 NaN 输出）
  - publish_lidar_tf = false（V9.2 正式配置），复用 URDF 已有 TF，
    禁止重复发布外参

参数 (默认值即 V9.2 正式配置，见 params/v9_2_vision.yaml):
  use_depth_camera:  使用深度相机 (默认False；深度相机已退出正式启动链)
  use_lidar:         使用LiDAR点云 (默认True)
  camera_info_topic: RGB CameraInfo 话题 (默认 /drone/camera/camera_info)
  lidar_topic:       LiDAR点云话题 (默认 /drone/lidar/points)
  pose_topic:        无人机位姿话题 (默认 /mavros/local_position/pose)
  world_frame:       'world'
  base_frame:        'base_footprint'
  camera_link_frame: 'ar0234_camera_link'      (FLU: X前/Y左/Z上)
  camera_frame:      'ar0234_camera_optical_frame' (RDF: Z前/X右/Y下)
  lidar_frame:       'mid360_link'
  max_time_diff:     图像-点云最大时间差 (默认 0.5s)
  publish_lidar_tf:  是否发布相机-雷达外参TF (默认False；V9.2 禁发)
  projection_radius: 点云投影筛选半径 (像素, 默认 15)
  tf_timeout:        TF查询超时 (默认 0.2s)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped, TransformStamped
import numpy as np
import math

from wind_turbine_interfaces.msg import DefectDetectionArray

from lidar_projection import (
    link_to_optical,
    pc2_to_xyz,
    project_optical_to_pixel,
    query_lidar_depth,
    stamps_within_skew,
)


class TargetLocalizer(Node):
    """2D检测→3D世界坐标转换 (仅发布 /vision/target_point; /arm/target_pose 由决策组 mission_fsm_node 发布)"""

    def __init__(self):
        super().__init__('target_localizer')

        # --- 参数（默认值 = V9.2 正式配置） ---
        self.declare_parameter('use_depth_camera', False)
        self.declare_parameter('use_lidar', True)
        self.declare_parameter('depth_topic', '/drone/camera/depth_raw')
        self.declare_parameter('depth_info_topic', '/drone/camera/depth_info')
        self.declare_parameter('camera_info_topic', '/drone/camera/camera_info')
        self.declare_parameter('lidar_topic', '/drone/lidar/points')
        self.declare_parameter('pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('camera_link_frame', 'ar0234_camera_link')
        self.declare_parameter('camera_frame', 'ar0234_camera_optical_frame')
        self.declare_parameter('lidar_frame', 'mid360_link')
        self.declare_parameter('max_time_diff', 0.5)
        self.declare_parameter('publish_lidar_tf', False)
        self.declare_parameter('projection_radius', 15.0)
        self.declare_parameter('tf_timeout', 0.2)

        use_depth = self.get_parameter('use_depth_camera').value
        use_lidar = self.get_parameter('use_lidar').value
        self.max_time_diff = self.get_parameter('max_time_diff').value
        self.projection_radius = self.get_parameter('projection_radius').value

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 数据缓存 ---
        self.latest_lidar = None        # PointCloud2 (含 header.stamp)
        self.latest_depth = None        # float32 深度图 (米, 仅 use_depth_camera)
        self.camera_intrinsics = None   # 3x3 内参矩阵 (来自 RGB camera_info)
        self.latest_pose = None         # 无人机Pose (备用)

        # --- 订阅 ---
        best_effort = QoSProfile(
            depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            DefectDetectionArray, '/vision/detection', self._on_detection, 10)

        # RGB CameraInfo（PM 冻结：内参唯一来源）
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self._on_camera_info, best_effort)

        if use_lidar:
            self.create_subscription(
                PointCloud2, self.get_parameter('lidar_topic').value,
                self._on_lidar, best_effort)

        if use_depth:
            # 深度相机已退出正式启动链，保留代码仅作兼容
            self.create_subscription(
                Image, self.get_parameter('depth_topic').value,
                self._on_depth, best_effort)

        self.create_subscription(
            PoseStamped, self.get_parameter('pose_topic').value,
            self._on_pose, best_effort)

        # --- 发布 ---
        self.target_pub = self.create_publisher(
            PointStamped, '/vision/target_point', 10)

        # --- 相机-雷达外参TF（默认关闭；V9.2 正式配置 publish_lidar_tf=false） ---
        self.tf_broadcaster = None
        if self.get_parameter('publish_lidar_tf').value:
            self.tf_broadcaster = TransformBroadcaster(self)
            self._publish_calibration_tf()
        else:
            self.get_logger().info(
                'publish_lidar_tf=false：不发布相机-雷达外参TF，'
                '复用 V9.2/URDF 已有 TF（PM 冻结要求）')

        self.get_logger().info(
            f'3D定位节点已启动 | 深度相机: {use_depth} | LiDAR: {use_lidar} '
            f'| 最大时间差: {self.max_time_diff}s | '
            f'相机帧: {self.get_parameter("camera_frame").value} | '
            f'雷达帧: {self.get_parameter("lidar_frame").value}')

    def _publish_calibration_tf(self):
        """发布相机-雷达外参静态TF (仅 publish_lidar_tf=true 时；V9.2 不启用)"""
        # 外参以 TF 树为准：优先从 TF 查询 camera_link ← lidar_frame 真实外参
        cam_link = self.get_parameter('camera_link_frame').value
        lidar = self.get_parameter('lidar_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(
                cam_link, lidar, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = cam_link
            t.child_frame_id = lidar
            t.transform.translation.x = tf.transform.translation.x
            t.transform.translation.y = tf.transform.translation.y
            t.transform.translation.z = tf.transform.translation.z
            t.transform.rotation = tf.transform.rotation
            self.tf_broadcaster.sendTransform(t)
            self.get_logger().info(
                f'相机-雷达外参TF: {cam_link} ← {lidar}（来自 TF 树查询）')
        except Exception as e:
            self.get_logger().warn(
                f'TF 查询 {cam_link} ← {lidar} 失败，跳过外参TF发布: {e}')

    # ---- 回调 ----

    def _on_camera_info(self, msg: CameraInfo):
        """RGB CameraInfo 回调：内参唯一来源（V9.2 由 Gazebo 按 hfov/分辨率计算）"""
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if k[0, 0] > 0 and k[1, 1] > 0:
            self.camera_intrinsics = k

    def _on_lidar(self, msg: PointCloud2):
        """LiDAR点云回调（保留 header.stamp = 雷达采集时间用于 0.5s 时间差检查）"""
        self.latest_lidar = msg

    def _on_depth(self, msg: Image):
        """深度图回调 (32FC1 = float32米；仅 use_depth_camera=true 时订阅)"""
        try:
            if msg.encoding == '32FC1':
                self.latest_depth = np.frombuffer(
                    msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding in ('16UC1', 'mono16'):
                mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                self.latest_depth = mm.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().error(f'深度图解析失败: {e}')

    def _on_pose(self, msg: PoseStamped):
        self.latest_pose = msg.pose

    def _on_detection(self, msg: DefectDetectionArray):
        """收到检测结果 → LiDAR点云投影定位 → TF变换 → 发布 /vision/target_point

        PM 冻结底线：任何无效路径（时间差超限/无点云/无内参/TF缺失/NaN）
        一律 return，停止发布，绝不发 NaN。
        """
        # 0.5s 图像-点云时间差检查（PM 冻结：超过拒绝本次定位）
        if self.latest_lidar is None:
            self.get_logger().debug('等待LiDAR点云...', throttle_duration_sec=5.0)
            return
        if not stamps_within_skew(
                msg.header.stamp, self.latest_lidar.header.stamp,
                self.max_time_diff):
            self.get_logger().warn(
                f'图像-点云时间差超限(>{self.max_time_diff}s)，'
                f'拒绝本次定位', throttle_duration_sec=5.0)
            return

        # 内参必须来自 /drone/camera/camera_info（PM 冻结），缺失则不出结果
        if self.camera_intrinsics is None:
            self.get_logger().debug('等待RGB CameraInfo...', throttle_duration_sec=5.0)
            return

        # 雷达点云 → 相机 link 系（FLU）：TF 查 mid360_link ← 世界链
        cam_link = self.get_parameter('camera_link_frame').value
        lidar = self.get_parameter('lidar_frame').value
        lidar_to_cam_link = self._lookup_tf(cam_link, lidar)
        if lidar_to_cam_link is None:
            return

        # 相机 link 系 → optical 系（PM 冻结要求的实际代码，见 lidar_projection.link_to_optical）
        try:
            points_lidar = pc2_to_xyz(self.latest_lidar)
            points_link = (lidar_to_cam_link @ np.c_[points_lidar, np.ones(len(points_lidar))].T).T[:, :3]
            points_optical = link_to_optical(points_link)
        except (ValueError, TypeError) as e:
            self.get_logger().warn(f'点云解析失败，跳过本次定位: {e}', throttle_duration_sec=5.0)
            return

        # camera(optical) → world 变换
        cam_frame = self.get_parameter('camera_frame').value
        camera_to_world = self._lookup_tf(
            self.get_parameter('world_frame').value, cam_frame)
        if camera_to_world is None:
            return

        for det in msg.detections:
            # bbox中心
            cx = (det.bbox_x_min + det.bbox_x_max) / 2.0
            cy = (det.bbox_y_min + det.bbox_y_max) / 2.0

            # 点云投影 + 框中心邻域稳健深度
            depth = query_lidar_depth(
                points_optical, cx, cy,
                self.camera_intrinsics, self.projection_radius)
            if depth is None or not np.isfinite(depth) or depth <= 0:
                continue

            # 像素 → optical 相机坐标（用 CameraInfo 内参投影逆变换）
            fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
            cx_i, cy_i = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
            x_cam = (cx - cx_i) / fx * depth
            y_cam = (cy - cy_i) / fy * depth
            z_cam = depth

            # 相机 → 世界
            point_world = camera_to_world @ np.array([x_cam, y_cam, z_cam, 1.0])
            wx, wy, wz = float(point_world[0]), float(point_world[1]), float(point_world[2])

            # NaN/Inf 防御（PM 底线：绝不发 NaN）
            if not (math.isfinite(wx) and math.isfinite(wy) and math.isfinite(wz)):
                self.get_logger().warn('定位结果含非法值，已丢弃', throttle_duration_sec=5.0)
                continue

            # 发布 /vision/target_point (世界坐标, 时间戳=图像采集时间)
            target = PointStamped()
            target.header.stamp = msg.header.stamp
            target.header.frame_id = self.get_parameter('world_frame').value
            target.point.x = wx
            target.point.y = wy
            target.point.z = wz
            self.target_pub.publish(target)

    # ---- 工具 ----

    def _lookup_tf(self, target_frame: str, source_frame: str):
        """查询TF: source_frame → target_frame (4x4矩阵)"""
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame,
                rclpy.time.Time(seconds=0, nanoseconds=0),
                timeout=rclpy.duration.Duration(
                    seconds=self.get_parameter('tf_timeout').value))
        except Exception:
            self.get_logger().debug(
                f'TF未找到: {source_frame}→{target_frame}', throttle_duration_sec=5.0)
            return None

        t = tf.transform.translation
        r = tf.transform.rotation

        # 四元数→旋转矩阵
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2*r.y*r.y - 2*r.z*r.z
        R[0, 1] = 2*r.x*r.y - 2*r.z*r.w
        R[0, 2] = 2*r.x*r.z + 2*r.y*r.w
        R[1, 0] = 2*r.x*r.y + 2*r.z*r.w
        R[1, 1] = 1 - 2*r.x*r.x - 2*r.z*r.z
        R[1, 2] = 2*r.y*r.z - 2*r.x*r.w
        R[2, 0] = 2*r.x*r.z - 2*r.y*r.w
        R[2, 1] = 2*r.y*r.z + 2*r.x*r.w
        R[2, 2] = 1 - 2*r.x*r.x - 2*r.y*r.y

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T


def main():
    rclpy.init()
    node = TargetLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
