"""ROS 2 LiDAR projection check using deterministic synthetic point data."""

import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from haiying_vision_3d.target_point_node import TargetPointNode


class SyntheticLidarAndCamera(Node):
    def __init__(self):
        super().__init__("synthetic_lidar_and_camera")
        self.info_pub = self.create_publisher(
            CameraInfo, "/camera/camera_info_rect", qos_profile_sensor_data
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/livox/lidar", qos_profile_sensor_data
        )
        self.pixel_pub = self.create_publisher(PointStamped, "/vision/target_pixel", 10)
        self.result = None
        self.create_subscription(PointStamped, "/vision/target_point", self._result, 10)
        self.create_timer(0.1, self._publish)

    def _result(self, message):
        self.result = message

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "ar0234_optical_frame"
        info.width = 640
        info.height = 480
        info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)

        header = Header(stamp=stamp, frame_id="livox_frame")
        cloud = point_cloud2.create_cloud_xyz32(
            header,
            [[0.0, 0.0, 2.0], [0.4, 0.2, 2.0], [-0.4, -0.2, 2.0]],
        )
        self.cloud_pub.publish(cloud)

        pixel = PointStamped()
        pixel.header.stamp = stamp
        pixel.header.frame_id = "ar0234_optical_frame"
        pixel.point.x = 420.0
        pixel.point.y = 290.0
        pixel.point.z = 0.95
        self.pixel_pub.publish(pixel)


def main():
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "mode:=lidar",
            "-p",
            "calibration_ready:=true",
            "-p",
            "max_measurement_age_sec:=2.0",
        ]
    )
    target = TargetPointNode()
    source = SyntheticLidarAndCamera()
    executor = SingleThreadedExecutor()
    executor.add_node(target)
    executor.add_node(source)
    deadline = time.monotonic() + 8.0
    try:
        while source.result is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if source.result is None:
            print("FAIL: no /vision/target_point message received", file=sys.stderr)
            return 1
        actual = np.asarray(
            [source.result.point.x, source.result.point.y, source.result.point.z]
        )
        expected = np.asarray([0.4, 0.2, 2.0])
        if not np.allclose(actual, expected, atol=1e-6):
            print(f"FAIL: expected {expected}, received {actual}", file=sys.stderr)
            return 1
        if source.result.header.frame_id != "livox_frame":
            print(
                f"FAIL: unexpected frame {source.result.header.frame_id}", file=sys.stderr
            )
            return 1
        print(
            "PASS: projected /livox/lidar point -> /vision/target_point = "
            f"({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}) m "
            f"in {source.result.header.frame_id}"
        )
        return 0
    finally:
        executor.shutdown()
        target.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
