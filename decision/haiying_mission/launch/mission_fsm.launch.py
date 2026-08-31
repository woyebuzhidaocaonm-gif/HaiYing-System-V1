#!/usr/bin/env python3
"""唯一任务状态机启动文件（决策层单独启动，独立于冻结链）。

用法:
    ros2 launch haiying_mission mission_fsm.launch.py
    ros2 launch haiying_mission mission_fsm.launch.py params_file:=/path/to.yaml

正式联仿拓扑（与 docs/FROZEN_CONTROL_CHAIN.md 一致）:
    ros2 launch attitude_cmd freeze_chain.launch.py   # 链路+转换节点
    ros2 launch haiying_mission mission_fsm.launch.py # 决策层（本文件）
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('haiying_mission'),
        'config', 'mission_fsm.yaml')

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='状态机参数文件路径'),
        Node(
            package='haiying_mission',
            executable='mission_fsm_node',
            name='mission_fsm_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
