#!/usr/bin/env python3
"""Publish Gazebo's real base_footprint pose as world -> base_footprint TF."""

import math

import rclpy
from gazebo_msgs.msg import LinkStates
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class WorldBaseTfPublisher(Node):
    def __init__(self):
        super().__init__("v9_2_world_base_tf")

        self.declare_parameter(
            "link_name",
            "custom_quad_333_v9_2::base_footprint",
        )
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("child_frame", "base_footprint")
        self.declare_parameter("link_states_topic", "/gazebo/link_states")

        self.link_name = self.get_parameter("link_name").value
        self.world_frame = self.get_parameter("world_frame").value
        self.child_frame = self.get_parameter("child_frame").value
        topic = self.get_parameter("link_states_topic").value

        self._broadcaster = TransformBroadcaster(self)
        self._missing_reported = False

        self.create_subscription(
            LinkStates,
            topic,
            self._on_link_states,
            10,
        )

        self.get_logger().info(
            f"WORLD_BASE_TF_READY topic={topic} "
            f"link={self.link_name} "
            f"tf={self.world_frame}->{self.child_frame}"
        )

    def _on_link_states(self, msg):
        try:
            index = msg.name.index(self.link_name)
        except ValueError:
            if not self._missing_reported:
                self.get_logger().warning(
                    f"LinkStates中尚未找到 {self.link_name}"
                )
                self._missing_reported = True
            return

        self._missing_reported = False
        pose = msg.pose[index]
        q = pose.orientation

        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            q.x,
            q.y,
            q.z,
            q.w,
        )

        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("拒绝发布包含NaN或Inf的Gazebo位姿")
            return

        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1.0e-9:
            self.get_logger().error("拒绝发布零范数四元数")
            return

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.child_frame

        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z

        transform.transform.rotation.x = q.x / norm
        transform.transform.rotation.y = q.y / norm
        transform.transform.rotation.z = q.z / norm
        transform.transform.rotation.w = q.w / norm

        self._broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = WorldBaseTfPublisher()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
