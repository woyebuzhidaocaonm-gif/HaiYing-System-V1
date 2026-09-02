"""Launch YOLO + MID-360 fusion with calibrated intrinsics and ideal extrinsic."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("haiying_vision_3d")
    parameters = os.path.join(
        package_share, "config", "yolo_lidar_fusion_ideal.yaml"
    )
    model_path = LaunchConfiguration("model_path")
    device = LaunchConfiguration("device")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value=(
                    "/mnt/c/Users/Jokei/Desktop/挑战杯/yolov5/output/best.onnx"
                ),
                description="Absolute Ubuntu path to trained best.pt or best.onnx",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="Ultralytics device for .pt models: cpu, 0, 0,1, ...",
            ),
            Node(
                package="haiying_vision_3d",
                executable="yolo_lidar_fusion_node",
                name="yolo_lidar_fusion_node",
                output="screen",
                parameters=[
                    parameters,
                    {"model_path": model_path, "device": device},
                ],
            ),
        ]
    )
