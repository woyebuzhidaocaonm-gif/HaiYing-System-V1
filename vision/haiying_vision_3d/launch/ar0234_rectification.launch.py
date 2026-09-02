"""Start the AR0234 USB camera and fisheye rectification pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("haiying_vision_3d")
    rectifier_parameters = os.path.join(
        package_share, "config", "ar0234_rectifier.yaml"
    )
    calibration_file = os.path.join(
        package_share, "config", "ar0234_fisheye_1920x1080.yaml"
    )
    start_camera = LaunchConfiguration("start_camera")
    video_device = LaunchConfiguration("video_device")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_camera",
                default_value="true",
                description="Start usb_cam; set false when another driver publishes /camera/image_raw",
            ),
            DeclareLaunchArgument(
                "video_device",
                default_value="/dev/video0",
                description="V4L2 device exposed by the AR0234 camera",
            ),
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                namespace="camera",
                name="ar0234_camera",
                output="screen",
                condition=IfCondition(start_camera),
                parameters=[
                    {
                        "video_device": video_device,
                        "framerate": 30.0,
                        "io_method": "mmap",
                        "frame_id": "ar0234_optical_frame",
                        "pixel_format": "mjpeg2rgb",
                        "image_width": 1920,
                        "image_height": 1080,
                        "camera_name": "ar0234_fisheye",
                    }
                ],
            ),
            Node(
                package="haiying_vision_3d",
                executable="fisheye_rectifier_node",
                name="fisheye_rectifier_node",
                output="screen",
                parameters=[
                    rectifier_parameters,
                    {"calibration_file": calibration_file},
                ],
            ),
        ]
    )
