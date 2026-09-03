"""HaiYing V9.2 Gazebo model, ros2_control and sensor-TF entry."""

import os
import re
import subprocess

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def static_tf(name, parent, child, xyz, rpy, use_sim_time):
    """Create one static-transform publisher."""
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            "--x", str(xyz[0]),
            "--y", str(xyz[1]),
            "--z", str(xyz[2]),
            "--roll", str(rpy[0]),
            "--pitch", str(rpy[1]),
            "--yaw", str(rpy[2]),
            "--frame-id", parent,
            "--child-frame-id", child,
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )


def generate_launch_description():
    package_share = get_package_share_directory("haiying_v9_2")
    arm_uav_share = get_package_share_directory("arm_uav_joint")
    so101_prefix = get_package_prefix("so-101_description")
    gazebo_share = get_package_share_directory("gazebo_ros")

    gui = LaunchConfiguration("gui")
    verbose = LaunchConfiguration("verbose")
    use_sim_time = LaunchConfiguration("use_sim_time")
    pause = LaunchConfiguration("pause")
    px4_dir = LaunchConfiguration("px4_autopilot_dir")

    wrapper = os.path.join(
        package_share,
        "urdf",
        "so101_arm_uav_gazebo_v9_2.urdf.xacro",
    )
    world = os.path.join(
        package_share,
        "worlds",
        "offshore_wind_turbine_takeoff_stand_v2_wind12.world",
    )

    xacro_result = subprocess.run(
        ["xacro", wrapper],
        check=True,
        capture_output=True,
        text=True,
    )
    clean_robot_description = re.sub(
        r"<!--.*?-->",
        "",
        xacro_result.stdout,
        flags=re.DOTALL,
    ).strip()
    robot_description = ParameterValue(
        clean_robot_description,
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world,
            "gui": gui,
            "verbose": verbose,
            "pause": pause,
        }.items(),
    )

    model_publisher = Node(
        package="haiying_v9_2",
        executable="publish_v9_2_model.py",
        # Model publication must use wall time so it also works while
        # Gazebo is intentionally paused before PX4 is ready.
        output="screen",
    )

    spawn_model = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "custom_quad_333_v9_2",
            "-topic", "/haiying_v9_2/model_description",
            "-x", LaunchConfiguration("model_x"),
            "-y", LaunchConfiguration("model_y"),
            "-z", LaunchConfiguration("model_z"),
        ],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    world_base_tf = Node(
        package="haiying_v9_2",
        executable="publish_world_base_tf.py",
        name="v9_2_world_base_tf",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"link_name": "custom_quad_333_v9_2::base_footprint"},
            {"world_frame": "world"},
            {"child_frame": "base_footprint"},
            {"link_states_topic": "/gazebo/link_states"},
        ],
        output="screen",
    )

    controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "arm_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
            # Gazebo starts paused. Activation completes after physics resumes.
            "--switch-timeout", "300",
        ],
        output="screen",
    )

    px4_plugin_dir = PathJoinSubstitution([
        px4_dir,
        "build",
        "px4_sitl_default",
        "build_gazebo-classic",
    ])
    px4_model_dir = PathJoinSubstitution([
        px4_dir,
        "Tools",
        "simulation",
        "gazebo-classic",
        "sitl_gazebo-classic",
        "models",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("verbose", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "pause",
            default_value="true",
            description="Keep Gazebo paused until PX4 is ready",
        ),
        DeclareLaunchArgument("model_x", default_value="0.0"),
        DeclareLaunchArgument("model_y", default_value="-1.0"),
        DeclareLaunchArgument("model_z", default_value="0.3"),
        DeclareLaunchArgument(
            "px4_autopilot_dir",
            default_value=EnvironmentVariable(
                "PX4_AUTOPILOT_DIR",
                default_value="",
            ),
            description="Absolute PX4-Autopilot root directory",
        ),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        AppendEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            os.path.join(package_share, "models"),
        ),
        AppendEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            os.path.join(arm_uav_share, "models"),
        ),
        AppendEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            os.path.join(so101_prefix, "share"),
        ),
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH", px4_model_dir),
        AppendEnvironmentVariable("GAZEBO_PLUGIN_PATH", px4_plugin_dir),
        AppendEnvironmentVariable("LD_LIBRARY_PATH", px4_plugin_dir),
        gazebo,
        robot_state_publisher,
        world_base_tf,
        model_publisher,
        spawn_model,
        static_tf(
            "v9_2_camera_link_tf",
            "base_footprint",
            "ar0234_camera_link",
            (0.0, -0.149887, 0.055),
            (3.1415926536, 0.0, -1.5707963268),
            use_sim_time,
        ),
        static_tf(
            "v9_2_camera_optical_tf",
            "ar0234_camera_link",
            "ar0234_camera_optical_frame",
            (0.0, 0.0, 0.0),
            (-1.5707963268, 0.0, -1.5707963268),
            use_sim_time,
        ),
        static_tf(
            "v9_2_mid360_tf",
            "base_footprint",
            "mid360_link",
            (0.0, 0.000113, -0.145),
            (3.1415926536, 0.0, -1.5707963268),
            use_sim_time,
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_model,
                on_exit=[
                    TimerAction(
                        period=1.0,
                        actions=[controller_spawner],
                    )
                ],
            )
        ),
    ])
