import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("haiying_vision_3d")
    calibration_parameters = os.path.join(
        package_share, "config", "calibration_publisher.yaml"
    )
    detector_parameters = os.path.join(
        package_share, "config", "yolo_target_pixel.yaml"
    )
    target_parameters = os.path.join(package_share, "config", "target_point.yaml")

    model_path = LaunchConfiguration("model_path")
    calib_file = LaunchConfiguration("calib_file")
    simulation_mode = LaunchConfiguration("simulation_mode")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value=EnvironmentVariable("HAIYING_YOLO_ONNX", default_value=""),
                description="Absolute WSL path to the YOLOv5 ONNX model",
            ),
            DeclareLaunchArgument(
                "calib_file",
                default_value=EnvironmentVariable("HAIYING_CALIBRATION", default_value=""),
                description="Validated camera/LiDAR JSON or YAML calibration file",
            ),
            DeclareLaunchArgument("simulation_mode", default_value="false"),
            Node(
                package="haiying_vision_3d",
                executable="calibration_node",
                name="camera_lidar_calibration",
                output="screen",
                parameters=[
                    calibration_parameters,
                    {
                        "calib_file": calib_file,
                        "simulation_mode": ParameterValue(
                            simulation_mode, value_type=bool
                        ),
                    },
                ],
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
                parameters=[
                    target_parameters,
                    {
                        "mode": "lidar",
                        "calibration_ready": True,
                        "use_tf_extrinsic": True,
                    },
                ],
            ),
        ]
    )
