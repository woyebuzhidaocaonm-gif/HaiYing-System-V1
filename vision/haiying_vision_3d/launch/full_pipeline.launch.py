import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("haiying_vision_3d")
    detector_parameters = os.path.join(
        package_share, "config", "yolo_target_pixel.yaml"
    )
    target_parameters = os.path.join(package_share, "config", "target_point.yaml")
    model_path = LaunchConfiguration("model_path")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value=EnvironmentVariable("HAIYING_YOLO_ONNX", default_value=""),
                description="Absolute WSL path to the exported YOLOv5 ONNX model",
            ),
            Node(
                package="haiying_vision_3d",
                executable="yolo_target_pixel_node",
                name="yolo_target_pixel_node",
                output="screen",
                parameters=[detector_parameters, {"model_path": model_path}],
            ),
            Node(
                package="haiying_vision_3d",
                executable="target_point_node",
                name="target_point_node",
                output="screen",
                parameters=[target_parameters],
            ),
        ]
    )
