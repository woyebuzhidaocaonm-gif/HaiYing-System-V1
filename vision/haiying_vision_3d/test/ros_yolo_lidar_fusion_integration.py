"""End-to-end ROS check: real YOLO model plus a synthetic LiDAR cluster."""

import sys
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Header, Int32

from haiying_vision_3d.yolo_lidar_fusion_node import YoloLidarFusionNode


MODEL_PATH = "/mnt/c/Users/Jokei/Desktop/挑战杯/yolov5/output/best.onnx"
IMAGE_PATH = (
    "/mnt/c/Users/Jokei/Desktop/挑战杯/"
    "yolo_pycharm/dataset_1000/images/test/10.jpg"
)


class FusionSource(Node):
    def __init__(self):
        super().__init__("fusion_test_source")
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            raise FileNotFoundError(IMAGE_PATH)
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(
            Image, "/camera/image_rect", qos_profile_sensor_data
        )
        self.info_pub = self.create_publisher(
            CameraInfo, "/camera/camera_info_rect", qos_profile_sensor_data
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/livox/lidar", qos_profile_sensor_data
        )
        self.distance = None
        self.point = None
        self.count = None
        self.annotated = None
        self.target_cloud = None
        self.create_subscription(
            Float32, "/vision/target_distance", self._on_distance, 10
        )
        self.create_subscription(PointStamped, "/vision/target_point", self._on_point, 10)
        self.create_subscription(Int32, "/vision/target_point_count", self._on_count, 10)
        self.create_subscription(
            Image, "/vision/yolo_lidar_image", self._on_image, 10
        )
        self.create_subscription(
            PointCloud2, "/vision/target_cloud", self._on_cloud, 10
        )
        self.create_timer(0.25, self._publish)

    def _on_distance(self, message):
        self.distance = float(message.data)

    def _on_point(self, message):
        self.point = message

    def _on_count(self, message):
        self.count = int(message.data)

    def _on_image(self, message):
        self.annotated = message

    def _on_cloud(self, message):
        self.target_cloud = message

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id="camera_test_frame")
        image = self.bridge.cv2_to_imgmsg(self.image, encoding="bgr8")
        image.header = header
        self.image_pub.publish(image)

        info = CameraInfo()
        info.header = header
        info.height, info.width = self.image.shape[:2]
        info.k = [700.0, 0.0, 512.0, 0.0, 700.0, 512.0, 0.0, 0.0, 1.0]
        info.p = [700.0, 0.0, 512.0, 0.0, 0.0, 700.0, 512.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)

        # Project a 3 m-deep cluster into the known class-2 test box near
        # (426, 953). One 8 m point verifies robust depth outlier rejection.
        points = []
        for u, v, z in [
            (420, 949, 2.98),
            (424, 951, 3.00),
            (428, 953, 3.02),
            (432, 955, 3.01),
            (425, 958, 2.99),
        ]:
            points.append(((u - 512.0) * z / 700.0, (v - 512.0) * z / 700.0, z))
        points.append(((426 - 512.0) * 8.0 / 700.0, (953 - 512.0) * 8.0 / 700.0, 8.0))
        cloud_header = Header(stamp=stamp, frame_id="lidar_test_frame")
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(cloud_header, points))


def main():
    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    rclpy.init(
        args=[
            "--ros-args",
            "-p", f"model_path:={MODEL_PATH}",
            "-p", "calibration_ready:=true",
            "-p", f"lidar_to_camera:={identity}",
            "-p", "max_sensor_skew_sec:=0.5",
            "-p", "minimum_inference_interval_sec:=0.0",
        ]
    )
    fusion = YoloLidarFusionNode()
    source = FusionSource()
    executor = SingleThreadedExecutor()
    executor.add_node(fusion)
    executor.add_node(source)
    deadline = time.monotonic() + 12.0
    try:
        while (
            source.distance is None
            or source.point is None
            or source.count is None
            or source.annotated is None
            or source.target_cloud is None
        ) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if source.distance is None:
            print("FAIL: no fused distance published", file=sys.stderr)
            return 1
        expected_point = np.asarray([-0.37, 1.89, 3.0])
        actual_point = np.asarray(
            [source.point.point.x, source.point.point.y, source.point.point.z]
        )
        expected_distance = float(np.linalg.norm(expected_point))
        if not np.allclose(actual_point, expected_point, atol=0.08):
            print(f"FAIL: point {actual_point} != {expected_point}", file=sys.stderr)
            return 1
        if abs(source.distance - expected_distance) > 0.08:
            print(
                f"FAIL: distance {source.distance:.3f} != {expected_distance:.3f}",
                file=sys.stderr,
            )
            return 1
        if source.count < 3 or source.count > 5:
            print(f"FAIL: unexpected inlier count {source.count}", file=sys.stderr)
            return 1
        if source.target_cloud.width != source.count:
            print(
                f"FAIL: cloud width {source.target_cloud.width} != count {source.count}",
                file=sys.stderr,
            )
            return 1
        print(
            "PASS: real YOLO detection + box LiDAR cluster -> "
            f"distance={source.distance:.3f}m, points={source.count}"
        )
        return 0
    finally:
        executor.shutdown()
        fusion.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
