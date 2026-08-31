"""契约测试：话题名/类型/服务与 NED 转换（契约 V2.1，5 项）。"""
import pytest
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def test_publisher_topics_and_types(helpers):
    """三发布器话题名与消息类型符合契约 V2.1。"""
    fsm = helpers.fsm
    found = {}
    for topic in ('/system/current_state', '/uav/cmd_vel', '/arm/target_pose'):
        found[topic] = sorted(
            {i.topic_type for i in fsm.get_publishers_info_by_topic(topic)})
    assert found['/system/current_state'] == ['std_msgs/msg/String']
    assert found['/uav/cmd_vel'] == ['geometry_msgs/msg/Twist']
    assert found['/arm/target_pose'] == ['geometry_msgs/msg/PoseStamped']


def test_subscription_topics(helpers):
    """三订阅话题与契约一致。"""
    fsm = helpers.fsm
    for topic in ('/vision/target_point', '/mavros/local_position/pose',
                  '/uav/flight_fault'):
        infos = fsm.get_subscriptions_info_by_topic(topic)
        assert len(infos) >= 1, f'未订阅 {topic}'


def test_reset_service_registered(helpers):
    """/fsm/reset 服务已注册（std_srvs/Trigger）。"""
    host, executor = helpers.host, helpers.executor
    def _types():
        # Humble: 返回 (name, [types]) 元组列表
        return dict(host.get_service_names_and_types_by_node(
            'mission_fsm_node', '/'))
    assert helpers.pump_until(executor, lambda: '/fsm/reset' in _types())
    assert 'std_srvs/srv/Trigger' in _types()['/fsm/reset']


def test_cmd_vel_ned_conversion(helpers):
    """ENU → NED 转换: v_ned=(v_enu.y, v_enu.x, -v_enu.z); wz_ned=-wz_enu。"""
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    helpers.fsm._publish_cmd_vel_enu(1.0, 2.0, 3.0, 0.5)
    assert helpers.wait_msgs(msgs, 1)
    m = msgs[-1]
    assert m.linear.x == pytest.approx(2.0)
    assert m.linear.y == pytest.approx(1.0)
    assert m.linear.z == pytest.approx(-3.0)
    assert m.angular.z == pytest.approx(-0.5)


def test_state_published_every_tick(helpers):
    """/system/current_state 每拍持续发布（防下游丢消息）。"""
    msgs = helpers.collector('/system/current_state', String)
    for _ in range(3):
        helpers.fsm._tick()
    assert helpers.wait_msgs(msgs, 3)
    assert all(m.data == 'SEARCHING' for m in msgs)
