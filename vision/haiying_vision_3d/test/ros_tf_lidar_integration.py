"""ROS 2 integration check for calibration TF -> LiDAR projection."""

import tempfile
import time

from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import yaml

from haiying_vision_3d.calibration_node import CalibrationNode
from haiying_vision_3d.target_point_node import TargetPointNode


class SyntheticTfSensor(Node):
    def __init__(self):
        super().__init__("synthetic_tf_sensor")
        self.info_pub = self.create_publisher(
            CameraInfo, "/camera/camera_info_rect", qos_profile_sensor_data
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/livox/lidar", qos_profile_sensor_data
        )
        self.pixel_pub = self.create_publisher(PointStamped, "/vision/target_pixel", 10)
        self.result = None
        self.create_subscription(PointStamped, "/vision/target_point", self._result, 10)
        self.create_timer(0.2, self._publish)

    def _result(self, message):
        self.result = message

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "ar0234_optical_frame"
        info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)
        self.cloud_pub.publish(
            point_cloud2.create_cloud_xyz32(
                Header(stamp=stamp, frame_id="livox_frame"),
                [[0.0, 0.0, 2.0], [0.4, 0.2, 2.0]],
            )
        )
        pixel = PointStamped()
        pixel.header.stamp = stamp
        pixel.header.frame_id = "ar0234_optical_frame"
        pixel.point.x = 420.0
        pixel.point.y = 290.0
        self.pixel_pub.publish(pixel)


def main():
    data = {
        "calibration": {"status": "simulation"},
        "camera_in_base": {
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
        },
        "lidar_in_base": {
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
        },
        "validation": {"reprojection_error_px": 0.0},
    }
    temporary = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(data, temporary)
    temporary.close()
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"calib_file:={temporary.name}",
            "-p",
            "simulation_mode:=true",
            "-p",
            "mode:=lidar",
            "-p",
            "calibration_ready:=true",
            "-p",
            "use_tf_extrinsic:=true",
            "-p",
            "max_measurement_age_sec:=2.0",
            "-p",
            "max_sensor_skew_sec:=0.3",
        ]
    )
    calibration = CalibrationNode()
    target = TargetPointNode()
    source = SyntheticTfSensor()
    executor = SingleThreadedExecutor()
    for node in (calibration, target, source):
        executor.add_node(node)
    deadline = time.monotonic() + 8.0
    try:
        while source.result is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if source.result is None:
            print("FAIL: no TF-calibrated target point received")
            return 1
        actual = [
            source.result.point.x,
            source.result.point.y,
            source.result.point.z,
        ]
        if any(abs(a - b) > 1e-6 for a, b in zip(actual, [0.4, 0.2, 2.0])):
            print(f"FAIL: unexpected target point {actual}")
            return 1
        print(
            "PASS: loop-free calibration TF -> LiDAR projection -> "
            f"/vision/target_point = {actual}"
        )
        return 0
    finally:
        executor.shutdown()
        calibration.destroy_node()
        target.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
