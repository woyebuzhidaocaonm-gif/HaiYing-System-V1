"""ROS 2 node that publishes a stable pinhole view from an equidistant camera."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .fisheye import (
    create_rectification_maps,
    load_fisheye_calibration,
    make_rectified_camera_matrix,
    rectify_image,
)


class FisheyeRectifierNode(Node):
    def __init__(self) -> None:
        super().__init__("fisheye_rectifier_node")
        default_calibration = os.path.join(
            get_package_share_directory("haiying_vision_3d"),
            "config",
            "ar0234_fisheye_1920x1080.yaml",
        )
        self.declare_parameter("input_image_topic", "/camera/image_raw")
        self.declare_parameter("output_image_topic", "/camera/image_rect")
        self.declare_parameter(
            "output_camera_info_topic", "/camera/camera_info_rect"
        )
        self.declare_parameter("calibration_file", default_calibration)
        self.declare_parameter("output_width", 1920)
        self.declare_parameter("output_height", 1080)
        self.declare_parameter("rectified_focal_length", 700.0)
        self.declare_parameter("frame_id", "ar0234_optical_frame")
        self.declare_parameter("drop_out_of_order_frames", True)

        calibration_path = str(self.get_parameter("calibration_file").value)
        if not calibration_path:
            calibration_path = default_calibration
        self.calibration = load_fisheye_calibration(calibration_path)
        self.output_width = int(self.get_parameter("output_width").value)
        self.output_height = int(self.get_parameter("output_height").value)
        focal_length = float(self.get_parameter("rectified_focal_length").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.drop_out_of_order_frames = bool(
            self.get_parameter("drop_out_of_order_frames").value
        )
        self.rectified_camera_matrix = make_rectified_camera_matrix(
            self.output_width, self.output_height, focal_length
        )
        self.map1, self.map2 = create_rectification_maps(
            self.calibration,
            (self.output_width, self.output_height),
            self.rectified_camera_matrix,
        )

        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(
            Image,
            str(self.get_parameter("output_image_topic").value),
            qos_profile_sensor_data,
        )
        self.info_publisher = self.create_publisher(
            CameraInfo,
            str(self.get_parameter("output_camera_info_topic").value),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("input_image_topic").value),
            self._on_image,
            qos_profile_sensor_data,
        )
        self.frame_count = 0
        self.last_input_stamp_ns = None
        horizontal_fov = np.degrees(
            2.0 * np.arctan(self.output_width / (2.0 * focal_length))
        )
        self.get_logger().info(
            f"Loaded {calibration_path}; expected raw image "
            f"{self.calibration.width}x{self.calibration.height}; rectified output "
            f"{self.output_width}x{self.output_height}, horizontal FOV "
            f"{horizontal_fov:.1f} deg"
        )

    def _camera_info(self, source: Image) -> CameraInfo:
        info = CameraInfo()
        info.header = source.header
        if self.frame_id:
            info.header.frame_id = self.frame_id
        info.width = self.output_width
        info.height = self.output_height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = self.rectified_camera_matrix.reshape(-1).tolist()
        info.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
        info.p = [
            float(self.rectified_camera_matrix[0, 0]),
            0.0,
            float(self.rectified_camera_matrix[0, 2]),
            0.0,
            0.0,
            float(self.rectified_camera_matrix[1, 1]),
            float(self.rectified_camera_matrix[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return info

    def _on_image(self, message: Image) -> None:
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if (
            self.drop_out_of_order_frames
            and stamp_ns > 0
            and self.last_input_stamp_ns is not None
            and stamp_ns <= self.last_input_stamp_ns
        ):
            self.get_logger().warning(
                "Dropping duplicate or out-of-order camera frame",
                throttle_duration_sec=2.0,
            )
            return
        if (
            message.width != self.calibration.width
            or message.height != self.calibration.height
        ):
            self.get_logger().error(
                "Dropping image: calibration is for "
                f"{self.calibration.width}x{self.calibration.height}, but input is "
                f"{message.width}x{message.height}",
                throttle_duration_sec=5.0,
            )
            return
        if stamp_ns > 0:
            self.last_input_stamp_ns = stamp_ns
        try:
            raw = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            rectified = rectify_image(raw, self.map1, self.map2)
            output = self.bridge.cv2_to_imgmsg(rectified, encoding="bgr8")
        except Exception as error:  # cv_bridge/OpenCV boundary: keep node alive.
            self.get_logger().error(
                f"Image rectification failed: {error}", throttle_duration_sec=2.0
            )
            return

        output.header = message.header
        if self.frame_id:
            output.header.frame_id = self.frame_id
        self.image_publisher.publish(output)
        self.info_publisher.publish(self._camera_info(message))
        self.frame_count += 1
        if self.frame_count == 1:
            self.get_logger().info("First rectified image and CameraInfo published")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FisheyeRectifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
