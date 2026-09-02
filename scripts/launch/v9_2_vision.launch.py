#!/usr/bin/env python3
"""V9.2 视觉链正式启动文件（PM 冻结：RGB + 雷达点云定位，无深度相机）

启动内容：
  - yolo_detector     （订阅 /drone/camera/image_raw → /vision/detection）
  - target_localizer  （/vision/detection + /drone/lidar/points
                        + /drone/camera/camera_info → /vision/target_point）

前置：V9.2 联合仿真已启动（ros2 launch haiying_v9_2 v9_2_simulation.launch.py），
      由 V9.2/仿真组提供 world→base_footprint 动态 TF 与全部传感器静态 TF。
本启动文件不启动任何桥接器、不发布任何 TF（publish_lidar_tf=false）。

用法：
    ros2 launch v9_2_vision.launch.py   # 或见 scripts/run_all.sh 一键脚本
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMS_FILE = str(SCRIPT_DIR.parent / 'params' / 'v9_2_vision.yaml')
DEFAULT_MODEL_PATH = os.environ.get('HAIYING_YOLO_WEIGHTS', '')
DEFAULT_DEVICE = os.environ.get('HAIYING_YOLO_DEVICE', 'cpu')


def generate_launch_description():
    model_path = LaunchConfiguration('model_path')
    device = LaunchConfiguration('device')
    conf_threshold = LaunchConfiguration('conf_threshold')
    iou_threshold = LaunchConfiguration('iou_threshold')
    img_size = LaunchConfiguration('img_size')
    publish_annotated = LaunchConfiguration('publish_annotated')

    yolo = ExecuteProcess(
        cmd=[
            'python3', str(SCRIPT_DIR.parent / 'yolo_detector.py'),
            '--ros-args',
            '-p', 'image_topic:=/drone/camera/image_raw',
            '-p', ['model_path:=', model_path],
            '-p', ['device:=', device],
            '-p', ['conf_threshold:=', conf_threshold],
            '-p', ['iou_threshold:=', iou_threshold],
            '-p', ['img_size:=', img_size],
            '-p', ['publish_annotated:=', publish_annotated],
            '-p', 'camera_frame:=ar0234_camera_optical_frame',
        ],
        output='screen',
        name='yolo_detector',
    )

    localizer = ExecuteProcess(
        cmd=[
            'python3', str(SCRIPT_DIR.parent / 'target_localizer.py'),
            '--ros-args',
            '--params-file', PARAMS_FILE,
        ],
        output='screen',
        name='target_localizer',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=DEFAULT_MODEL_PATH,
            description=(
                'YOLOv5权重路径；可通过HAIYING_YOLO_WEIGHTS设置')),
        DeclareLaunchArgument(
            'device',
            default_value=DEFAULT_DEVICE,
            description='推理设备，默认cpu'),
        DeclareLaunchArgument('conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('iou_threshold', default_value='0.45'),
        DeclareLaunchArgument('img_size', default_value='[640, 640]'),
        DeclareLaunchArgument('publish_annotated', default_value='false'),
        yolo,
        localizer,
    ])
