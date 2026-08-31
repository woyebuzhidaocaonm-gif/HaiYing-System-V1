#!/usr/bin/env python3
"""
Gazebo Harmonic → ROS2 相机桥接器（V9.2 帧名适配）
=================================================
直接用 gz-transport13 Python 绑定订阅 Gazebo 相机话题，
转为 ROS2 sensor_msgs/Image 发布，避开 ros_gz_bridge 的库版本不兼容问题。

注意：V9.2 正式链由 haiying_v9_2 模型的 gazebo_ros_camera 插件
直接发布 /drone/camera/image_raw + camera_info（frame=ar0234_camera_optical_frame），
**不启动本桥接器**。本脚本保留用于旧 gz-sensors 链路。

V9.2 适配要点：
  - 输出 frame_id = ar0234_camera_optical_frame（与 V9.2 相机帧一致）
  - header.stamp 透传 Gazebo 仿真时间（不再用 now() 覆盖）
  - CameraInfo 的 K 优先来自 gz /camera_info 话题（projection 矩阵），
    话题缺失时按图像分辨率 + hfov 计算（默认值=AR0234 仿真相机：
    1920x1080, hfov=1.3962634016），不再硬编码 320x240 的 277

用法:
    source /opt/ros/humble/setup.bash
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    python3 gz_camera_bridge.py

环境变量:
    GZ_CAMERA_TOPIC:      Gazebo相机话题名 (默认 /camera)
    GZ_CAMERA_INFO_TOPIC: Gazebo相机信息话题名 (默认 <GZ_CAMERA_TOPIC>_info)
    GZ_CAMERA_HFOV:       Gazebo相机水平FOV(rad) (默认 1.3962634016, AR0234)
    ROS_IMAGE_TOPIC:      ROS2输出话题名 (默认 /drone/camera/image_raw)
    ROS_CAMINFO_TOPIC:    ROS2 CameraInfo话题名 (默认 /drone/camera/camera_info)
    ROS_CAMERA_FRAME:     输出 frame_id (默认 ar0234_camera_optical_frame)
"""

import os
import sys
import time
import math

# 必须在导入 gz.msgs 之前设置，否则 protobuf 版本冲突
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

from gz.transport13 import Node as GzNode

# Gazebo PixelFormatType 枚举值 (from gz-msgs10)
PIXEL_FORMAT = {
    0:  ('UNKNOWN',         0, None),
    1:  ('L_INT8',          1, 'mono8'),
    2:  ('L_INT16',         2, 'mono16'),
    3:  ('RGB_INT8',        3, 'rgb8'),
    4:  ('RGBA_INT8',       4, 'rgba8'),
    5:  ('BGRA_INT8',       4, 'bgra8'),
    6:  ('RGB_INT16',       6, None),
    7:  ('RGB_INT32',       12, None),
    8:  ('BGR_INT8',        3, 'bgr8'),
    9:  ('BGR_INT16',       6, 'bgr16'),
    10: ('BGR_INT32',       12, None),
    11: ('R_FLOAT16',       2, None),
    12: ('RGB_FLOAT16',     6, None),
    13: ('R_FLOAT32',       4, None),
    14: ('RGB_FLOAT32',     12, None),
    15: ('BAYER_RGGB8',     1, 'bayer_rggb8'),
    16: ('BAYER_BGGR8',     1, 'bayer_bggr8'),
    17: ('BAYER_GBRG8',     1, 'bayer_gbrg8'),
    18: ('BAYER_GRBG8',     1, 'bayer_grbg8'),
}

# Bayer → BGR OpenCV转换映射
BAYER_TO_BGR = {
    15: cv2.COLOR_BAYER_RG2BGR,
    16: cv2.COLOR_BAYER_BG2BGR,
    17: cv2.COLOR_BAYER_GB2BGR,
    18: cv2.COLOR_BAYER_GR2BGR,
}


class GzCameraBridge(Node):
    """订阅 Gazebo 相机，桥接到 ROS2"""

    def __init__(self):
        super().__init__('gz_camera_bridge')

        gz_topic = os.environ.get('GZ_CAMERA_TOPIC', '/camera')
        gz_info_topic = os.environ.get(
            'GZ_CAMERA_INFO_TOPIC', gz_topic + '_info')
        ros_topic = os.environ.get('ROS_IMAGE_TOPIC', '/drone/camera/image_raw')
        ros_caminfo_topic = os.environ.get('ROS_CAMINFO_TOPIC', '/drone/camera/camera_info')
        self.ros_frame = os.environ.get(
            'ROS_CAMERA_FRAME', 'ar0234_camera_optical_frame')
        self.gz_hfov = float(
            os.environ.get('GZ_CAMERA_HFOV', '1.3962634016'))

        self.get_logger().info(f'GZ topic: {gz_topic} → ROS2 topic: {ros_topic}')

        # ROS2 发布者
        self.image_pub = self.create_publisher(Image, ros_topic, 10)
        self.caminfo_pub = self.create_publisher(CameraInfo, ros_caminfo_topic, 10)
        self.bridge = CvBridge()

        # Gazebo Transport 订阅
        self.gz_node = GzNode()
        self.frame_count = 0
        self.start_time = time.time()
        self._format_warned = set()
        self._k = None            # 来自 gz CameraInfo 的 K；None=按 hfov 计算
        self._k_warned = False

        # 订阅相机信息话题（K 的动态来源）
        self._subscribe_gz_info(gz_info_topic)

        # 订阅相机话题
        self._subscribe_gz(gz_topic)
        self.get_logger().info(
            f'Gazebo相机桥接器已启动 | frame={self.ros_frame}，等待图像...')

    def _gz_to_cv2(self, msg) -> np.ndarray:
        """将 Gazebo Image 转为 OpenCV BGR numpy 数组"""
        width = msg.width
        height = msg.height
        pixel_format = msg.pixel_format_type
        step = msg.step
        data = msg.data

        fmt_info = PIXEL_FORMAT.get(pixel_format)
        fmt_name = fmt_info[0] if fmt_info else f'UNKNOWN_{pixel_format}'

        if pixel_format in (3, 4, 5, 8):
            # RGB_INT8, RGBA_INT8, BGRA_INT8, BGR_INT8
            channels = fmt_info[1]
            expected = width * height * channels
            if len(data) < expected:
                return None
            raw = np.frombuffer(data, dtype=np.uint8, count=expected)
            raw = raw.reshape((height, width, channels))

            if pixel_format == 3:  # RGB_INT8 → BGR
                return raw[:, :, ::-1].copy()
            elif pixel_format == 8:  # BGR_INT8
                return raw.copy()
            elif pixel_format == 4:  # RGBA_INT8 → BGR
                return cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
            elif pixel_format == 5:  # BGRA_INT8 → BGR
                return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            return raw.copy()

        elif pixel_format == 1:  # L_INT8 (grayscale)
            expected = width * height
            if len(data) < expected:
                return None
            raw = np.frombuffer(data, dtype=np.uint8, count=expected)
            raw = raw.reshape((height, width))
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        elif pixel_format in (15, 16, 17, 18):  # BAYER_*
            expected = width * height
            if len(data) < expected:
                return None
            raw = np.frombuffer(data, dtype=np.uint8, count=expected)
            raw = raw.reshape((height, width))
            bayer_code = BAYER_TO_BGR.get(pixel_format)
            if bayer_code is not None:
                return cv2.cvtColor(raw, bayer_code)
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        else:
            # 其他未处理格式，尝试作为RGB处理
            if pixel_format not in self._format_warned:
                self._format_warned.add(pixel_format)
                self.get_logger().warn(
                    f'未直接支持的像素格式: {fmt_name}, '
                    f'data_len={len(data)}, w={width}, h={height}, step={step}')
            expected = width * height * 3
            if len(data) >= expected:
                raw = np.frombuffer(data, dtype=np.uint8, count=expected)
                return raw.reshape((height, width, 3))[:, :, ::-1].copy()
            elif len(data) >= width * height:
                raw = np.frombuffer(data, dtype=np.uint8, count=width*height)
                return cv2.cvtColor(raw.reshape((height, width)), cv2.COLOR_GRAY2BGR)
            return None

    def _gz_stamp_to_ros(self, msg):
        """透传 Gazebo 仿真时间戳；时间戳为零时回退 ROS 时钟。"""
        stamp = msg.header.stamp
        if stamp.sec > 0 or stamp.nsec > 0:
            return rclpy.time.Time(
                seconds=stamp.sec, nanoseconds=stamp.nsec).to_msg()
        return self.get_clock().now().to_msg()

    def _k_from_hfov(self, width, height):
        """按水平FOV与分辨率计算内参 K（gz CameraInfo 缺失时的回退）"""
        fx = (width / 2.0) / math.tan(self.gz_hfov / 2.0)
        return [fx, 0.0, width / 2.0,
                0.0, fx, height / 2.0,
                0.0, 0.0, 1.0]

    def _subscribe_gz_info(self, topic: str):
        """订阅 gz CameraInfo（projection 矩阵 → K）"""
        def on_info(msg):
            try:
                p = msg.projection.p  # 4x4 row-major
                fx, fy = p[0], p[5]
                cx, cy = p[2], p[6]
                if fx > 0 and fy > 0:
                    self._k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            except Exception as e:
                self.get_logger().warn(f'gz CameraInfo 解析失败: {e}')

        try:
            from gz.msgs10.camera_info_pb2 import CameraInfo as GzCameraInfo
            self.gz_node.subscribe(GzCameraInfo, topic, on_info)
            self.get_logger().info(f'已订阅 Gazebo 相机信息话题: {topic}')
        except Exception as e:
            self.get_logger().warn(
                f'gz CameraInfo 订阅失败({topic})，K 将按 hfov 计算: {e}')

    def _subscribe_gz(self, topic: str):
        """订阅 Gazebo Transport 话题"""

        def on_image(msg):
            """Gazebo 图像回调"""
            try:
                np_img = self._gz_to_cv2(msg)
                if np_img is None or np_img.size == 0:
                    return

                # 转为 ROS2 Image 并发布
                # 使用 passthrough 避开 cv_bridge 与 OpenCV 5.x 的 cvtype 编码表冲突
                ros_img = self.bridge.cv2_to_imgmsg(np_img, encoding='passthrough')
                ros_img.header.stamp = self._gz_stamp_to_ros(msg)
                ros_img.header.frame_id = self.ros_frame
                self.image_pub.publish(ros_img)

                # 发布 CameraInfo（K 优先 gz CameraInfo，回退 hfov 计算）
                caminfo = CameraInfo()
                caminfo.header = ros_img.header
                caminfo.height = np_img.shape[0]
                caminfo.width = np_img.shape[1]
                if self._k is None:
                    self._k = self._k_from_hfov(
                        caminfo.width, caminfo.height)
                    if not self._k_warned:
                        self._k_warned = True
                        self.get_logger().warn(
                            f'gz CameraInfo 未就绪，K 按 hfov={self.gz_hfov} '
                            f'计算: fx={self._k[0]:.2f}')
                caminfo.k = self._k
                self.caminfo_pub.publish(caminfo)

                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.get_logger().info(
                        f'已桥接 {self.frame_count} 帧 | {fps:.1f} FPS | '
                        f'{np_img.shape[1]}x{np_img.shape[0]}')

            except Exception as e:
                import traceback
                self.get_logger().error(
                    f'图像处理错误: {e}\n{traceback.format_exc()}')

        from gz.msgs10.image_pb2 import Image as GzImage
        self.gz_node.subscribe(GzImage, topic, on_image)
        self.get_logger().info(f'已订阅 Gazebo 话题: {topic}')


def main():
    rclpy.init()
    node = GzCameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
