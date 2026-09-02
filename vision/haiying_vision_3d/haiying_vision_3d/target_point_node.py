"""ROS 2 node that publishes measured target XYZ coordinates."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from .geometry import (
    apply_transform,
    deproject_depth_pixel,
    median_depth_in_window,
    quaternion_to_rotation_matrix,
    select_lidar_point_for_pixel,
)
from .pointcloud import pointcloud2_to_xyz


class TargetPointNode(Node):
    """Fuse a target pixel with depth or calibrated LiDAR measurements."""

    def __init__(self) -> None:
        super().__init__("target_point_node")
        self.declare_parameter("mode", "lidar")
        self.declare_parameter("target_pixel_topic", "/vision/target_pixel")
        self.declare_parameter("output_topic", "/vision/target_point")
        self.declare_parameter("camera_info_topic", "/camera/camera_info_rect")
        self.declare_parameter(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("target_frame", "")
        self.declare_parameter("max_measurement_age_sec", 0.5)
        self.declare_parameter("max_sensor_skew_sec", 0.1)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 30.0)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("depth_window_radius", 2)
        self.declare_parameter("search_radius_px", 12.0)
        self.declare_parameter("calibration_ready", False)
        self.declare_parameter("use_tf_extrinsic", False)
        self.declare_parameter("camera_optical_frame", "ar0234_optical_frame")
        self.declare_parameter("lidar_to_camera", np.eye(4).reshape(-1).tolist())

        self.mode = str(self.get_parameter("mode").value).lower()
        if self.mode not in {"depth", "lidar"}:
            raise ValueError("mode must be either 'depth' or 'lidar'")

        transform_values = np.asarray(
            self.get_parameter("lidar_to_camera").value, dtype=np.float64
        )
        if transform_values.size != 16:
            raise ValueError("lidar_to_camera must contain 16 row-major values")
        self.lidar_to_camera = transform_values.reshape(4, 4)
        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.depth_image: Optional[Image] = None
        self.lidar_cloud: Optional[PointCloud2] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        output_topic = str(self.get_parameter("output_topic").value)
        self.publisher = self.create_publisher(PointStamped, output_topic, 10)
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("target_pixel_topic").value),
            self._on_target_pixel,
            10,
        )
        if self.mode == "depth":
            self.create_subscription(
                Image,
                str(self.get_parameter("depth_topic").value),
                self._on_depth,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                PointCloud2,
                str(self.get_parameter("lidar_topic").value),
                self._on_lidar,
                qos_profile_sensor_data,
            )

        self.get_logger().info(
            f"mode={self.mode}; publishing measured XYZ on {output_topic}"
        )
        if self.mode == "lidar" and not self.get_parameter("calibration_ready").value:
            self.get_logger().warn(
                "LiDAR publication is locked until validated calibration is supplied "
                "and calibration_ready is true."
            )

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _on_depth(self, message: Image) -> None:
        self.depth_image = message

    def _on_lidar(self, message: PointCloud2) -> None:
        self.lidar_cloud = message

    @staticmethod
    def _stamp_is_zero(message) -> bool:
        return message.header.stamp.sec == 0 and message.header.stamp.nanosec == 0

    def _is_fresh(self, message) -> bool:
        if self._stamp_is_zero(message):
            return True
        age = self.get_clock().now() - Time.from_msg(message.header.stamp)
        limit = float(self.get_parameter("max_measurement_age_sec").value)
        return -0.05 <= age.nanoseconds / 1e9 <= limit

    def _on_target_pixel(self, pixel: PointStamped) -> None:
        u, v = float(pixel.point.x), float(pixel.point.y)
        if not math.isfinite(u) or not math.isfinite(v):
            self.get_logger().warn("Ignoring a non-finite target pixel")
            return
        if self.camera_info is None:
            self.get_logger().warn("Waiting for CameraInfo", throttle_duration_sec=2.0)
            return
        if self.mode == "depth":
            result = self._solve_from_depth(u, v, pixel.header.stamp)
        else:
            result = self._solve_from_lidar(u, v, pixel.header.stamp)
        if result is None:
            return
        point_xyz, source_frame, source_stamp = result
        published = self._to_output_message(point_xyz, source_frame, source_stamp)
        if published is not None:
            self.publisher.publish(published)
            self.get_logger().info(
                f"target XYZ [{published.header.frame_id}]: "
                f"{published.point.x:.3f}, {published.point.y:.3f}, "
                f"{published.point.z:.3f} m",
                throttle_duration_sec=0.5,
            )

    def _camera_matrix(self) -> np.ndarray:
        return np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)

    def _timestamps_are_synchronized(self, first, second) -> bool:
        if (first.sec == 0 and first.nanosec == 0) or (
            second.sec == 0 and second.nanosec == 0
        ):
            return True
        first_seconds = first.sec + first.nanosec / 1e9
        second_seconds = second.sec + second.nanosec / 1e9
        skew = abs(first_seconds - second_seconds)
        limit = float(self.get_parameter("max_sensor_skew_sec").value)
        if skew > limit:
            self.get_logger().warn(
                f"Rejecting unsynchronized measurements: skew={skew:.3f}s, "
                f"limit={limit:.3f}s",
                throttle_duration_sec=1.0,
            )
            return False
        return True

    def _solve_from_depth(self, u: float, v: float, pixel_stamp):
        message = self.depth_image
        if message is None:
            self.get_logger().warn("Waiting for aligned depth image", throttle_duration_sec=2.0)
            return None
        if not self._is_fresh(message):
            self.get_logger().warn("Depth image is stale", throttle_duration_sec=2.0)
            return None
        if not self._timestamps_are_synchronized(pixel_stamp, message.header.stamp):
            return None
        image = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
        if message.encoding in {"16UC1", "mono16"} or image.dtype == np.uint16:
            image_m = image.astype(np.float32) * float(
                self.get_parameter("depth_scale").value
            )
        else:
            image_m = image.astype(np.float32)
        depth = median_depth_in_window(
            image_m,
            u,
            v,
            int(self.get_parameter("depth_window_radius").value),
            float(self.get_parameter("min_depth_m").value),
            float(self.get_parameter("max_depth_m").value),
        )
        if depth is None:
            self.get_logger().warn("No valid depth near target pixel", throttle_duration_sec=1.0)
            return None
        k = self._camera_matrix()
        xyz = deproject_depth_pixel(u, v, depth, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        frame = message.header.frame_id or self.camera_info.header.frame_id
        return xyz, frame, message.header.stamp

    def _solve_from_lidar(self, u: float, v: float, pixel_stamp):
        if not bool(self.get_parameter("calibration_ready").value):
            self.get_logger().error(
                "Refusing uncalibrated LiDAR coordinates", throttle_duration_sec=2.0
            )
            return None
        message = self.lidar_cloud
        if message is None:
            self.get_logger().warn("Waiting for PointCloud2", throttle_duration_sec=2.0)
            return None
        if not self._is_fresh(message):
            self.get_logger().warn("LiDAR cloud is stale", throttle_duration_sec=2.0)
            return None
        if not self._timestamps_are_synchronized(pixel_stamp, message.header.stamp):
            return None
        points = pointcloud2_to_xyz(message)
        lidar_to_camera = self._resolve_lidar_to_camera(message)
        if lidar_to_camera is None:
            return None
        xyz = select_lidar_point_for_pixel(
            points,
            u,
            v,
            self._camera_matrix(),
            lidar_to_camera,
            float(self.get_parameter("search_radius_px").value),
            float(self.get_parameter("min_depth_m").value),
            float(self.get_parameter("max_depth_m").value),
        )
        if xyz is None:
            self.get_logger().warn(
                "No projected LiDAR point near target pixel", throttle_duration_sec=1.0
            )
            return None
        return xyz, message.header.frame_id, message.header.stamp

    def _resolve_lidar_to_camera(self, message: PointCloud2):
        if not bool(self.get_parameter("use_tf_extrinsic").value):
            return self.lidar_to_camera
        camera_frame = str(self.get_parameter("camera_optical_frame").value)
        lidar_frame = message.header.frame_id
        if not camera_frame or not lidar_frame:
            self.get_logger().error(
                "Camera optical frame and LiDAR frame must be non-empty",
                throttle_duration_sec=2.0,
            )
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                camera_frame,
                lidar_frame,
                Time.from_msg(message.header.stamp)
                if message.header.stamp.sec or message.header.stamp.nanosec
                else Time(),
                timeout=Duration(seconds=0.1),
            ).transform
        except TransformException as error:
            self.get_logger().warn(
                f"No calibrated TF {lidar_frame} -> {camera_frame}: {error}",
                throttle_duration_sec=2.0,
            )
            return None
        quaternion = [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quaternion_to_rotation_matrix(*quaternion)
        matrix[:3, 3] = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
        return matrix

    def _to_output_message(self, xyz, source_frame: str, stamp):
        target_frame = str(self.get_parameter("target_frame").value)
        output = PointStamped()
        output.header.stamp = stamp
        output.header.frame_id = source_frame
        transformed = np.asarray(xyz, dtype=np.float64)
        if target_frame and target_frame != source_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time.from_msg(stamp) if stamp.sec or stamp.nanosec else Time(),
                    timeout=Duration(seconds=0.1),
                ).transform
            except TransformException as error:
                self.get_logger().warn(
                    f"No TF {source_frame} -> {target_frame}: {error}",
                    throttle_duration_sec=2.0,
                )
                return None
            transformed = apply_transform(
                transformed,
                [transform.translation.x, transform.translation.y, transform.translation.z],
                [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ],
            )
            output.header.frame_id = target_frame
        output.point.x, output.point.y, output.point.z = map(float, transformed)
        return output


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetPointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
