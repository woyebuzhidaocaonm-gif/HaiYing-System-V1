"""Publish a validated, loop-free camera/LiDAR static TF tree."""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from .calibration import (
    compute_lidar_to_camera,
    load_calibration_file,
    matrix_to_transform,
)


class CalibrationNode(Node):
    """Publish previously solved calibration; it does not run PnP or ICP."""

    def __init__(self) -> None:
        super().__init__("camera_lidar_calibration")
        self.declare_parameter("calib_file", "")
        self.declare_parameter("simulation_mode", False)
        self.declare_parameter("max_reprojection_error_px", 3.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "ar0234_optical_frame")
        self.declare_parameter("lidar_frame", "livox_frame")
        self.declare_parameter("tree_mode", "base_tree")

        calib_file = str(self.get_parameter("calib_file").value)
        simulation_mode = bool(self.get_parameter("simulation_mode").value)
        data = load_calibration_file(
            calib_file,
            simulation_mode=simulation_mode,
            max_reprojection_error_px=float(
                self.get_parameter("max_reprojection_error_px").value
            ),
        )
        self.camera_in_base = data["camera_in_base"]
        self.lidar_in_base = data["lidar_in_base"]
        self.lidar_to_camera_matrix = compute_lidar_to_camera(
            self.camera_in_base, self.lidar_in_base
        )

        base = str(self.get_parameter("base_frame").value)
        camera = str(self.get_parameter("camera_frame").value)
        lidar = str(self.get_parameter("lidar_frame").value)
        if len({base, camera, lidar}) != 3 or not all((base, camera, lidar)):
            raise ValueError("base, camera, and LiDAR frames must be distinct and non-empty")

        tree_mode = str(self.get_parameter("tree_mode").value)
        now = self.get_clock().now().to_msg()
        if tree_mode == "base_tree":
            transforms = [
                self._message(base, camera, self.camera_in_base, now),
                self._message(base, lidar, self.lidar_in_base, now),
            ]
        elif tree_mode == "direct_extrinsic":
            transforms = [
                self._message(
                    camera,
                    lidar,
                    matrix_to_transform(self.lidar_to_camera_matrix),
                    now,
                )
            ]
        else:
            raise ValueError("tree_mode must be base_tree or direct_extrinsic")

        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(transforms)
        flattened = self.lidar_to_camera_matrix.reshape(-1)
        self.get_logger().info(
            f"Published validated loop-free calibration ({tree_mode}); "
            f"T_camera_lidar={np.array2string(flattened, precision=6, separator=',')}"
        )

    @staticmethod
    def _message(parent: str, child: str, transform: dict, stamp):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        translation = transform["translation"]
        rotation = transform["rotation"]
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        message.transform.rotation.x = float(rotation[0])
        message.transform.rotation.y = float(rotation[1])
        message.transform.rotation.z = float(rotation[2])
        message.transform.rotation.w = float(rotation[3])
        return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
