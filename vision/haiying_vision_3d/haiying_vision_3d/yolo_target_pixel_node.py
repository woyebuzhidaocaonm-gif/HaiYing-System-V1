"""ROS 2 YOLOv5 ONNX detector that publishes the best target pixel."""

from __future__ import annotations

import os
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .detector import decode_yolov5_predictions, letterbox


DEFAULT_CLASSES = [
    "craze",
    "corrosion",
    "surface_injure",
    "thunderstrike",
    "crack",
    "hide_craze",
]


class YoloTargetPixelNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_target_pixel_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_topic", "/camera/image_rect")
        self.declare_parameter("output_topic", "/vision/target_pixel")
        self.declare_parameter("input_size", 640)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("target_class_id", -1)
        self.declare_parameter("minimum_inference_interval_sec", 0.0)
        self.declare_parameter("class_names", DEFAULT_CLASSES)

        model_path = os.path.expanduser(str(self.get_parameter("model_path").value))
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                "model_path must point to the exported YOLOv5 ONNX model"
            )
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.bridge = CvBridge()
        self.last_inference_time = 0.0
        self.class_names = list(self.get_parameter("class_names").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.publisher = self.create_publisher(PointStamped, output_topic, 10)
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Loaded {model_path}; publishing best detection center on {output_topic}"
        )

    def _on_image(self, message: Image) -> None:
        now = time.monotonic()
        interval = float(
            self.get_parameter("minimum_inference_interval_sec").value
        )
        if now - self.last_inference_time < interval:
            return
        self.last_inference_time = now

        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        input_size = int(self.get_parameter("input_size").value)
        prepared, scale, pad_x, pad_y = letterbox(image, input_size)
        blob = cv2.dnn.blobFromImage(
            prepared, scalefactor=1.0 / 255.0, size=(input_size, input_size), swapRB=True
        )
        self.net.setInput(blob)
        prediction = self.net.forward()
        detections = decode_yolov5_predictions(
            prediction,
            image.shape[:2],
            scale,
            pad_x,
            pad_y,
            float(self.get_parameter("confidence_threshold").value),
            float(self.get_parameter("iou_threshold").value),
            int(self.get_parameter("target_class_id").value),
        )
        if not detections:
            return
        best = detections[0]
        pixel = PointStamped()
        pixel.header = message.header
        pixel.point.x = float(best["center"][0])
        pixel.point.y = float(best["center"][1])
        pixel.point.z = float(best["confidence"])
        self.publisher.publish(pixel)
        class_id = int(best["class_id"])
        class_name = (
            self.class_names[class_id]
            if 0 <= class_id < len(self.class_names)
            else str(class_id)
        )
        self.get_logger().info(
            f"target pixel ({pixel.point.x:.1f}, {pixel.point.y:.1f}), "
            f"class={class_name}, confidence={pixel.point.z:.3f}",
            throttle_duration_sec=0.5,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloTargetPixelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
