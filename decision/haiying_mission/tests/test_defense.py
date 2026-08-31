"""防御性测试：脏数据/位姿缺失/TF 缺失/零距离（8 项）。"""
import math

import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist


def test_nan_target_dropped_clock_not_refreshed(helpers):
    """NaN 样本丢弃且不刷新看门狗计时（3s 后按原有效样本计时 ERROR）。"""
    fsm, clock = helpers.fsm, helpers.clock
    helpers.send_target(10.0, 0.0, 0.0)
    assert helpers.pump_until(helpers.executor,
                              lambda: fsm._latest_target is not None)
    clock.advance(1.0)
    helpers.send_target(math.nan, 0.0, 0.0)
    helpers.pump(n=5)
    assert fsm._latest_target.point.x == 10.0   # 未被脏数据覆盖
    assert fsm._last_target_time == 100.0       # 计时未刷新
    clock.advance(2.0 + 0.01)                   # 距有效目标 3.01s
    fsm._tick()
    assert fsm._state == 'ERROR'                # 若刷新过计时则不会 ERROR


def test_inf_target_dropped(helpers):
    fsm = helpers.fsm
    helpers.send_target(math.inf, 0.0, 0.0)
    helpers.pump(n=5)
    assert fsm._latest_target is None
    assert fsm._armed is False


def test_out_of_bounds_target_dropped(helpers):
    fsm = helpers.fsm
    helpers.send_target(1000.0, 0.0, 0.0)   # max_target_pos=50
    helpers.pump(n=5)
    assert fsm._latest_target is None
    assert fsm._armed is False


def test_pose_missing_holds_zero_vel(helpers):
    """位姿缺失 → APPROACHING 悬停（不盲飞）。"""
    helpers.goto('APPROACHING')
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    helpers.fsm._tick()
    assert helpers.fsm._state == 'APPROACHING'
    assert helpers.wait_msgs(msgs, 1)
    m = msgs[-1]
    assert (m.linear.x, m.linear.y, m.linear.z) == (0.0, 0.0, 0.0)


def test_stale_pose_zero_vel(helpers):
    """位姿超时（pose_timeout）→ 悬停，即使目标新鲜。"""
    helpers.send_pose(0.0, 0.0, 0.0)
    helpers.send_target(10.0, 0.0, 0.0)
    fsm, clock = helpers.fsm, helpers.clock
    assert helpers.pump_until(helpers.executor,
                              lambda: fsm._latest_target is not None
                              and fsm._latest_pose is not None)
    fsm._tick()
    fsm._tick()   # → APPROACHING
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    clock.advance(1.0 + 0.01)
    helpers.send_target(10.0, 0.0, 0.0)   # 目标保持新鲜
    helpers.pump(n=3)
    fsm._tick()
    assert fsm._state == 'APPROACHING'
    assert helpers.wait_msgs(msgs, 1)
    m = msgs[-1]
    assert (m.linear.x, m.linear.y, m.linear.z) == (0.0, 0.0, 0.0)


def test_tf_missing_arm_pose_not_published(helpers):
    """TF 缺失 → 不下发 /arm/target_pose（hold，不发布脏坐标）。"""
    helpers.goto('BRUSHING')
    msgs = helpers.collector('/arm/target_pose', PoseStamped)
    helpers.fsm._tick()
    helpers.pump(n=5)
    assert len(msgs) == 0


def test_zero_distance_brushing_no_divzero(helpers):
    """目标与位姿重合 → 直接 BRUSHING，无除零崩溃。"""
    helpers.send_target(0.0, 0.0, 0.0)
    helpers.send_pose(0.0, 0.0, 0.0)
    fsm = helpers.fsm
    assert helpers.pump_until(helpers.executor,
                              lambda: fsm._latest_target is not None
                              and fsm._latest_pose is not None)
    fsm._tick()
    fsm._tick()
    fsm._tick()
    assert fsm._state == 'BRUSHING'


def test_arm_pose_published_with_tf(helpers):
    """TF 可用 → /arm/target_pose 以 base_footprint 发布目标位姿。"""
    fsm = helpers.fsm

    def fake_tf(target, source):
        t = TransformStamped()
        t.header.frame_id = source
        t.child_frame_id = target
        t.transform.rotation.w = 1.0
        return t

    fsm._tf_lookup = fake_tf
    helpers.goto('BRUSHING')
    msgs = helpers.collector('/arm/target_pose', PoseStamped)
    fsm._tick()
    assert helpers.wait_msgs(msgs, 1)
    m = msgs[-1]
    assert m.header.frame_id == 'base_footprint'
    assert m.pose.position.x == pytest.approx(0.3)
    assert m.pose.position.y == pytest.approx(0.0)
    assert m.pose.position.z == pytest.approx(0.0)
    assert m.pose.orientation.w == pytest.approx(1.0)
