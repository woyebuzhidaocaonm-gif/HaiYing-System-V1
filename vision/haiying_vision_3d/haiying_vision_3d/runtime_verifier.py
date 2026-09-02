"""Collect machine-readable evidence from the live radar/3D ROS 2 pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class RuntimeVerifier(Node):
    def __init__(self) -> None:
        super().__init__("haiying_runtime_verifier")
        self.declare_parameter("duration_sec", 15.0)
        self.declare_parameter("minimum_lidar_hz", 5.0)
        self.declare_parameter("minimum_target_distance_m", 0.05)
        self.declare_parameter("maximum_target_distance_m", 50.0)
        self.declare_parameter("evidence_source", "unconfirmed")
        self.declare_parameter("report_path", "")
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("pixel_topic", "/vision/target_pixel")
        self.declare_parameter("target_topic", "/vision/target_point")

        self.started = time.monotonic()
        self.done = False
        self.lidar_receipts: list[float] = []
        self.lidar_points: list[int] = []
        self.pixel_receipts: list[float] = []
        self.target_receipts: list[float] = []
        self.target_samples: list[dict] = []
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("lidar_topic").value),
            self._on_lidar,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("pixel_topic").value),
            self._on_pixel,
            10,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("target_topic").value),
            self._on_target,
            10,
        )
        self.create_timer(0.1, self._check_deadline)

    def _on_lidar(self, message: PointCloud2) -> None:
        self.lidar_receipts.append(time.monotonic())
        self.lidar_points.append(int(message.width * message.height))

    def _on_target(self, message: PointStamped) -> None:
        self.target_receipts.append(time.monotonic())
        self.target_samples.append(
            {
                "frame_id": message.header.frame_id,
                "x": float(message.point.x),
                "y": float(message.point.y),
                "z": float(message.point.z),
            }
        )

    def _on_pixel(self, message: PointStamped) -> None:
        self.pixel_receipts.append(time.monotonic())

    def _check_deadline(self) -> None:
        duration = float(self.get_parameter("duration_sec").value)
        if time.monotonic() - self.started >= duration:
            self.done = True

    @staticmethod
    def _rate(receipts: list[float]) -> float:
        if len(receipts) < 2 or receipts[-1] <= receipts[0]:
            return 0.0
        return (len(receipts) - 1) / (receipts[-1] - receipts[0])

    def report(self) -> dict:
        lidar_hz = self._rate(self.lidar_receipts)
        pixel_hz = self._rate(self.pixel_receipts)
        target_hz = self._rate(self.target_receipts)
        finite_targets = bool(self.target_samples) and all(
            sample["frame_id"]
            and all(math.isfinite(sample[key]) for key in ("x", "y", "z"))
            for sample in self.target_samples
        )
        minimum_distance = float(
            self.get_parameter("minimum_target_distance_m").value
        )
        maximum_distance = float(
            self.get_parameter("maximum_target_distance_m").value
        )
        plausible_targets = finite_targets and all(
            minimum_distance
            <= math.sqrt(sample["x"] ** 2 + sample["y"] ** 2 + sample["z"] ** 2)
            <= maximum_distance
            for sample in self.target_samples
        )
        reasons = []
        evidence_source = str(self.get_parameter("evidence_source").value)
        if evidence_source != "live_hardware":
            reasons.append(
                "evidence_source is not explicitly declared as live_hardware"
            )
        minimum_lidar_hz = float(self.get_parameter("minimum_lidar_hz").value)
        if len(self.lidar_receipts) < 2:
            reasons.append("fewer than two PointCloud2 messages received")
        elif lidar_hz < minimum_lidar_hz:
            reasons.append(
                f"LiDAR rate {lidar_hz:.2f} Hz is below {minimum_lidar_hz:.2f} Hz"
            )
        if len(self.target_receipts) < 2:
            reasons.append("fewer than two /vision/target_point messages received")
        if len(self.pixel_receipts) < 2:
            reasons.append("fewer than two /vision/target_pixel messages received")
        if self.target_samples and not finite_targets:
            reasons.append("target coordinates are non-finite or frame_id is empty")
        elif self.target_samples and not plausible_targets:
            reasons.append(
                "target distance is outside the configured physically plausible range"
            )

        return {
            "status": "passed" if not reasons else "incomplete",
            "evidence_source": evidence_source,
            "synthetic": evidence_source != "live_hardware",
            "duration_sec": round(time.monotonic() - self.started, 3),
            "lidar": {
                "topic": str(self.get_parameter("lidar_topic").value),
                "message_count": len(self.lidar_receipts),
                "measured_hz": round(lidar_hz, 3),
                "minimum_required_hz": minimum_lidar_hz,
                "points_per_frame_min": min(self.lidar_points, default=0),
                "points_per_frame_max": max(self.lidar_points, default=0),
            },
            "target_point": {
                "topic": str(self.get_parameter("target_topic").value),
                "message_count": len(self.target_receipts),
                "measured_hz": round(target_hz, 3),
                "finite_and_framed": finite_targets,
                "physically_plausible": plausible_targets,
                "distance_range_m": [minimum_distance, maximum_distance],
                "last_sample": self.target_samples[-1] if self.target_samples else None,
            },
            "target_pixel": {
                "topic": str(self.get_parameter("pixel_topic").value),
                "message_count": len(self.pixel_receipts),
                "measured_hz": round(pixel_hz, 3),
            },
            "failure_reasons": reasons,
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RuntimeVerifier()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        report = node.report()
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        report_path = str(node.get_parameter("report_path").value)
        if report_path:
            path = Path(report_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
            print(f"Report written to {path}")
        if report["status"] != "passed":
            raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
