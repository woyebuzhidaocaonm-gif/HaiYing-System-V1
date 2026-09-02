"""ROS 2 YOLO detector with calibrated MID-360 box-range fusion."""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Int32

from .detector import decode_yolov5_predictions, letterbox
from .fusion import estimate_range_in_box, project_lidar_to_image
from .pointcloud import pointcloud2_to_xyz


DEFAULT_CLASSES = [
    "craze",
    "corrosion",
    "surface_injure",
    "thunderstrike",
    "crack",
    "hide_craze",
]


class YoloLidarFusionNode(Node):
    """Detect objects and calculate robust range from points inside each box."""

    def __init__(self) -> None:
        super().__init__("yolo_lidar_fusion_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_topic", "/camera/image_rect")
        self.declare_parameter("camera_info_topic", "/camera/camera_info_rect")
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("annotated_image_topic", "/vision/yolo_lidar_image")
        self.declare_parameter("target_point_topic", "/vision/target_point")
        self.declare_parameter("target_distance_topic", "/vision/target_distance")
        self.declare_parameter("target_point_count_topic", "/vision/target_point_count")
        self.declare_parameter("target_cloud_topic", "/vision/target_cloud")
        self.declare_parameter("input_size", 640)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("target_class_id", -1)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("minimum_inference_interval_sec", 0.05)
        self.declare_parameter("max_sensor_skew_sec", 0.12)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 30.0)
        self.declare_parameter("minimum_points", 3)
        self.declare_parameter("box_inner_margin_ratio", 0.08)
        self.declare_parameter("box_fallback_expand_px", 8.0)
        self.declare_parameter("mad_scale", 3.0)
        self.declare_parameter("minimum_depth_gate_m", 0.12)
        self.declare_parameter("point_radius_px", 2)
        self.declare_parameter("snapshot_path", "")
        self.declare_parameter("class_names", DEFAULT_CLASSES)
        self.declare_parameter("calibration_ready", False)
        self.declare_parameter("lidar_to_camera", np.eye(4).reshape(-1).tolist())

        if not bool(self.get_parameter("calibration_ready").value):
            raise RuntimeError(
                "Fusion is locked: set calibration_ready=true only after supplying "
                "the intended camera intrinsics and LiDAR-to-camera transform"
            )
        values = np.asarray(
            self.get_parameter("lidar_to_camera").value, dtype=np.float64
        )
        if values.size != 16:
            raise ValueError("lidar_to_camera must contain 16 row-major values")
        self.lidar_to_camera = values.reshape(4, 4)
        self.class_names = list(self.get_parameter("class_names").value)
        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.lidar_cloud: Optional[PointCloud2] = None
        self.last_inference_time = 0.0
        self.snapshot_saved = False
        self.backend = ""
        self.model = None
        self.net = None
        self._load_model()

        self.image_publisher = self.create_publisher(
            Image, str(self.get_parameter("annotated_image_topic").value), 10
        )
        self.point_publisher = self.create_publisher(
            PointStamped, str(self.get_parameter("target_point_topic").value), 10
        )
        self.distance_publisher = self.create_publisher(
            Float32, str(self.get_parameter("target_distance_topic").value), 10
        )
        self.count_publisher = self.create_publisher(
            Int32, str(self.get_parameter("target_point_count_topic").value), 10
        )
        self.cloud_publisher = self.create_publisher(
            PointCloud2, str(self.get_parameter("target_cloud_topic").value), 10
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("lidar_topic").value),
            self._on_lidar,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"YOLO backend={self.backend}; publishing annotated fusion on "
            f"{self.get_parameter('annotated_image_topic').value}"
        )

    def _load_model(self) -> None:
        path = os.path.abspath(
            os.path.expanduser(str(self.get_parameter("model_path").value))
        )
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("model_path must point to best.pt or best.onnx")
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".onnx":
            self.net = cv2.dnn.readNetFromONNX(path)
            self.backend = "opencv-onnx"
            return
        if suffix != ".pt":
            raise ValueError("only .pt and .onnx YOLO models are supported")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Loading best.pt requires ultralytics and torch. Install them in "
                "the Ubuntu ROS environment, or use the exported best.onnx."
            ) from error
        self.model = YOLO(path)
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.class_names = [str(names[index]) for index in sorted(names)]
        elif names:
            self.class_names = list(names)
        self.backend = "ultralytics-pt"

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _on_lidar(self, message: PointCloud2) -> None:
        self.lidar_cloud = message

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) / 1e9

    def _timestamps_are_close(self, image: Image, cloud: PointCloud2) -> bool:
        first = self._stamp_seconds(image.header.stamp)
        second = self._stamp_seconds(cloud.header.stamp)
        if first == 0.0 or second == 0.0:
            return True
        skew = abs(first - second)
        limit = float(self.get_parameter("max_sensor_skew_sec").value)
        if skew > limit:
            self.get_logger().warn(
                f"Skipping unsynchronized image/cloud pair: {skew:.3f}s > {limit:.3f}s",
                throttle_duration_sec=1.0,
            )
            return False
        return True

    def _infer(self, image: np.ndarray) -> list[dict]:
        confidence = float(self.get_parameter("confidence_threshold").value)
        iou = float(self.get_parameter("iou_threshold").value)
        target_class = int(self.get_parameter("target_class_id").value)
        input_size = int(self.get_parameter("input_size").value)
        if self.backend == "ultralytics-pt":
            result = self.model.predict(
                source=image,
                imgsz=input_size,
                conf=confidence,
                iou=iou,
                device=str(self.get_parameter("device").value),
                verbose=False,
            )[0]
            detections = []
            if result.boxes is None:
                return detections
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, scores, classes):
                if target_class >= 0 and class_id != target_class:
                    continue
                detections.append(
                    {
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "box_xyxy": np.asarray(box, dtype=np.float64),
                    }
                )
            return sorted(
                detections, key=lambda item: item["confidence"], reverse=True
            )

        prepared, scale, pad_x, pad_y = letterbox(image, input_size)
        blob = cv2.dnn.blobFromImage(
            prepared,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            swapRB=True,
        )
        self.net.setInput(blob)
        decoded = decode_yolov5_predictions(
            self.net.forward(),
            image.shape[:2],
            scale,
            pad_x,
            pad_y,
            confidence,
            iou,
            target_class,
        )
        return [
            {
                "class_id": item["class_id"],
                "confidence": item["confidence"],
                "box_xyxy": np.asarray(
                    [
                        item["box"][0],
                        item["box"][1],
                        item["box"][0] + item["box"][2],
                        item["box"][1] + item["box"][3],
                    ],
                    dtype=np.float64,
                ),
            }
            for item in decoded
        ]

    def _on_image(self, message: Image) -> None:
        now = time.monotonic()
        interval = float(
            self.get_parameter("minimum_inference_interval_sec").value
        )
        if now - self.last_inference_time < interval:
            return
        self.last_inference_time = now
        if self.camera_info is None or self.lidar_cloud is None:
            self.get_logger().warn(
                "Waiting for CameraInfo and MID-360 PointCloud2",
                throttle_duration_sec=2.0,
            )
            return
        cloud = self.lidar_cloud
        if not self._timestamps_are_close(message, cloud):
            return

        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        detections = self._infer(image)
        annotated = image.copy()
        if not detections:
            self._publish_image(annotated, message)
            return

        k = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        points = pointcloud2_to_xyz(cloud)
        lidar_points, camera_points, pixels = project_lidar_to_image(
            points,
            k,
            self.lidar_to_camera,
            image.shape[:2],
            float(self.get_parameter("min_depth_m").value),
            float(self.get_parameter("max_depth_m").value),
        )
        valid_results = []
        for detection in detections:
            estimate = estimate_range_in_box(
                lidar_points,
                camera_points,
                pixels,
                detection["box_xyxy"],
                int(self.get_parameter("minimum_points").value),
                float(self.get_parameter("box_inner_margin_ratio").value),
                float(self.get_parameter("box_fallback_expand_px").value),
                float(self.get_parameter("mad_scale").value),
                float(self.get_parameter("minimum_depth_gate_m").value),
            )
            self._draw_detection(annotated, detection, estimate)
            if estimate is not None:
                valid_results.append((detection, estimate))

        if valid_results:
            best_detection, best_estimate = max(
                valid_results, key=lambda item: item[0]["confidence"]
            )
            self._publish_best(best_estimate, cloud)
            self.get_logger().info(
                f"target class={best_detection['class_id']} "
                f"distance={best_estimate.distance_m:.3f}m "
                f"points={best_estimate.point_count}",
                throttle_duration_sec=0.5,
            )
        self._publish_image(annotated, message)
        self._save_snapshot(annotated, valid_results)

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return str(class_id)

    def _draw_detection(self, image, detection, estimate) -> None:
        x1, y1, x2, y2 = np.rint(detection["box_xyxy"]).astype(int)
        color = (0, 220, 0) if estimate is not None else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{self._class_name(detection['class_id'])} "
            f"{detection['confidence']:.2f}"
        )
        if estimate is None:
            label += " | no lidar points"
        else:
            label += f" | {estimate.distance_m:.2f} m ({estimate.point_count} pts)"
            if estimate.depths_m.size:
                d_min = float(estimate.depths_m.min())
                d_max = float(estimate.depths_m.max())
                span = max(d_max - d_min, 1e-6)
                radius = int(self.get_parameter("point_radius_px").value)
                for pixel, depth in zip(estimate.pixels, estimate.depths_m):
                    ratio = (float(depth) - d_min) / span
                    dot_color = (int(255 * ratio), 0, int(255 * (1.0 - ratio)))
                    cv2.circle(
                        image,
                        tuple(np.rint(pixel).astype(int)),
                        radius,
                        dot_color,
                        -1,
                        lineType=cv2.LINE_AA,
                    )
        text_y = max(24, y1 - 8)
        cv2.putText(
            image,
            label,
            (max(0, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    def _publish_best(self, estimate, cloud: PointCloud2) -> None:
        point = PointStamped()
        point.header = cloud.header
        point.point.x, point.point.y, point.point.z = map(
            float, estimate.point_lidar
        )
        self.point_publisher.publish(point)
        distance = Float32()
        distance.data = float(estimate.distance_m)
        self.distance_publisher.publish(distance)
        count = Int32()
        count.data = int(estimate.point_count)
        self.count_publisher.publish(count)
        selected_cloud = point_cloud2.create_cloud_xyz32(
            cloud.header, estimate.points_lidar.tolist()
        )
        self.cloud_publisher.publish(selected_cloud)

    def _publish_image(self, image: np.ndarray, source: Image) -> None:
        output = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        output.header = source.header
        self.image_publisher.publish(output)

    def _save_snapshot(self, image: np.ndarray, valid_results) -> None:
        raw_path = str(self.get_parameter("snapshot_path").value).strip()
        if self.snapshot_saved or not raw_path or not valid_results:
            return
        path = os.path.abspath(os.path.expanduser(raw_path))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not cv2.imwrite(path, image):
            self.get_logger().error(f"Failed to save snapshot: {path}")
            return
        self.snapshot_saved = True
        self.get_logger().info(f"Saved fusion snapshot: {path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloLidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
