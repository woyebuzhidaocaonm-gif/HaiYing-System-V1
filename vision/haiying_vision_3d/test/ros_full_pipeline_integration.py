"""End-to-end ROS 2 check: real labeled image + deterministic depth."""

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
from sensor_msgs.msg import CameraInfo, Image

from haiying_vision_3d.target_point_node import TargetPointNode
from haiying_vision_3d.yolo_target_pixel_node import YoloTargetPixelNode


MODEL_PATH = "/mnt/c/Users/Jokei/Desktop/挑战杯/yolov5/output/best.onnx"
IMAGE_PATH = (
    "/mnt/c/Users/Jokei/Desktop/挑战杯/"
    "yolo_pycharm/dataset_1000/images/test/10.jpg"
)


class CameraWithDepth(Node):
    def __init__(self):
        super().__init__("camera_with_depth")
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            raise FileNotFoundError(IMAGE_PATH)
        self.depth = np.full(self.image.shape[:2], 2000, dtype=np.uint16)
        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(
            Image, "/camera/image_rect", qos_profile_sensor_data
        )
        self.depth_pub = self.create_publisher(
            Image,
            "/camera/aligned_depth_to_color/image_raw",
            qos_profile_sensor_data,
        )
        self.info_pub = self.create_publisher(
            CameraInfo, "/camera/camera_info_rect", qos_profile_sensor_data
        )
        self.result = None
        self.create_subscription(PointStamped, "/vision/target_point", self._result, 10)
        self.create_timer(0.25, self._publish)

    def _result(self, message):
        self.result = message

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        color = self.bridge.cv2_to_imgmsg(self.image, encoding="bgr8")
        color.header.stamp = stamp
        color.header.frame_id = "camera_color_optical_frame"
        self.color_pub.publish(color)

        depth = self.bridge.cv2_to_imgmsg(self.depth, encoding="16UC1")
        depth.header.stamp = stamp
        depth.header.frame_id = "camera_depth_optical_frame"
        self.depth_pub.publish(depth)

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera_depth_optical_frame"
        info.width = 1024
        info.height = 1024
        info.k = [800.0, 0.0, 512.0, 0.0, 800.0, 512.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)


def main():
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"model_path:={MODEL_PATH}",
            "-p",
            "mode:=depth",
            "-p",
            "max_measurement_age_sec:=2.0",
        ]
    )
    detector = YoloTargetPixelNode()
    target = TargetPointNode()
    camera = CameraWithDepth()
    executor = SingleThreadedExecutor()
    for node in (detector, target, camera):
        executor.add_node(node)
    deadline = time.monotonic() + 10.0
    try:
        while camera.result is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if camera.result is None:
            print("FAIL: no end-to-end target point received", file=sys.stderr)
            return 1
        actual = np.asarray(
            [camera.result.point.x, camera.result.point.y, camera.result.point.z]
        )
        # Best detection center is approximately (425.86, 951.63); use the
        # configured 2 m depth and K=(fx=fy=800,cx=cy=512).
        expected = np.asarray(
            [(425.8634 - 512.0) * 2.0 / 800.0, (951.6339 - 512.0) * 2.0 / 800.0, 2.0]
        )
        if not np.allclose(actual, expected, atol=0.01):
            print(f"FAIL: expected {expected}, received {actual}", file=sys.stderr)
            return 1
        print(
            "PASS: labeled image -> YOLO pixel -> depth -> /vision/target_point = "
            f"({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}) m"
        )
        return 0
    finally:
        executor.shutdown()
        detector.destroy_node()
        target.destroy_node()
        camera.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
