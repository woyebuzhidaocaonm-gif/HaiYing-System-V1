"""ROS 2 YOLO check using a real labeled held-out test image."""

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
from sensor_msgs.msg import Image

from haiying_vision_3d.yolo_target_pixel_node import YoloTargetPixelNode


MODEL_PATH = "/mnt/c/Users/Jokei/Desktop/挑战杯/yolov5/output/best.onnx"
IMAGE_PATH = (
    "/mnt/c/Users/Jokei/Desktop/挑战杯/"
    "yolo_pycharm/dataset_1000/images/test/10.jpg"
)


class LabeledImageSource(Node):
    def __init__(self):
        super().__init__("labeled_image_source")
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            raise FileNotFoundError(IMAGE_PATH)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image, "/camera/image_rect", qos_profile_sensor_data
        )
        self.result = None
        self.create_subscription(PointStamped, "/vision/target_pixel", self._result, 10)
        self.create_timer(0.25, self._publish)

    def _result(self, message):
        self.result = message

    def _publish(self):
        message = self.bridge.cv2_to_imgmsg(self.image, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera_color_optical_frame"
        self.publisher.publish(message)


def main():
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"model_path:={MODEL_PATH}",
            "-p",
            "confidence_threshold:=0.25",
        ]
    )
    detector = YoloTargetPixelNode()
    source = LabeledImageSource()
    executor = SingleThreadedExecutor()
    executor.add_node(detector)
    executor.add_node(source)
    deadline = time.monotonic() + 8.0
    try:
        while source.result is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if source.result is None:
            print("FAIL: no /vision/target_pixel message received", file=sys.stderr)
            return 1
        actual = np.asarray([source.result.point.x, source.result.point.y])
        # Ground-truth class-2 box center from labels/test/10.txt.
        expected = np.asarray([0.416504 * 1024, 0.930664 * 1024])
        if not np.allclose(actual, expected, atol=5.0):
            print(f"FAIL: expected center near {expected}, received {actual}", file=sys.stderr)
            return 1
        if source.result.point.z < 0.9:
            print(
                f"FAIL: expected confidence >= 0.9, received {source.result.point.z}",
                file=sys.stderr,
            )
            return 1
        print(
            "PASS: real labeled image -> /vision/target_pixel = "
            f"({actual[0]:.1f}, {actual[1]:.1f}), "
            f"confidence={source.result.point.z:.3f}"
        )
        return 0
    finally:
        executor.shutdown()
        detector.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
