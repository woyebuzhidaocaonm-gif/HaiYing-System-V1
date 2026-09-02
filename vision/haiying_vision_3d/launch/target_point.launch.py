from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    package_share = get_package_share_directory("haiying_vision_3d")
    parameters = os.path.join(package_share, "config", "target_point.yaml")
    return LaunchDescription(
        [
            Node(
                package="haiying_vision_3d",
                executable="target_point_node",
                name="target_point_node",
                output="screen",
                parameters=[parameters],
            )
        ]
    )
