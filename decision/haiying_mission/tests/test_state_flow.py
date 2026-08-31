"""状态流转测试：SEARCHING→BRUSHING 全链 + 丢失映射 + reset（9 项）。"""
import pytest
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger


def test_searching_to_target_found(helpers):
    helpers.send_target(10.0, 0.0, 0.0)
    assert helpers.pump_until(helpers.executor,
                              lambda: helpers.fsm._latest_target is not None)
    helpers.fsm._tick()
    assert helpers.fsm._state == 'TARGET_FOUND'
    assert helpers.fsm._armed is True   # 首个目标 → 看门狗武装


def test_target_found_to_approaching_one_tick(helpers):
    helpers.goto('TARGET_FOUND')
    helpers.fsm._tick()   # 1 拍
    assert helpers.fsm._state == 'APPROACHING'


def test_approaching_far_full_speed(helpers):
    helpers.send_pose(0.0, 0.0, 0.0)
    helpers.send_target(10.0, 0.0, 0.0)
    assert helpers.pump_until(helpers.executor,
                              lambda: helpers.fsm._latest_target is not None
                              and helpers.fsm._latest_pose is not None)
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    helpers.fsm._tick()   # → TARGET_FOUND
    helpers.fsm._tick()   # → APPROACHING
    helpers.fsm._tick()   # dist=10 ≥ 2 → 满速 1.0 m/s 朝目标
    assert helpers.fsm._state == 'APPROACHING'
    assert helpers.wait_msgs(msgs, 3)
    m = msgs[-1]
    # ENU v=(1,0,0) → NED linear=(0,1,0)
    assert m.linear.x == pytest.approx(0.0)
    assert m.linear.y == pytest.approx(1.0)
    assert m.linear.z == pytest.approx(0.0)


def test_approaching_mid_deceleration(helpers):
    helpers.send_pose(0.0, 0.0, 0.0)
    helpers.send_target(1.0, 0.0, 0.0)
    assert helpers.pump_until(helpers.executor,
                              lambda: helpers.fsm._latest_target is not None
                              and helpers.fsm._latest_pose is not None)
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    helpers.fsm._tick()
    helpers.fsm._tick()
    helpers.fsm._tick()
    assert helpers.wait_msgs(msgs, 3)
    m = msgs[-1]
    # ratio=(1.0-0.5)/(2.0-0.5)=1/3, speed=1.0*(1/3)*0.5=1/6 → ENU v=(1/6,0,0)
    assert m.linear.y == pytest.approx(1.0 / 6.0)
    assert m.linear.x == pytest.approx(0.0)


def test_brushing_near_target_zero_vel(helpers):
    helpers.goto('BRUSHING')
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    helpers.fsm._tick()
    assert helpers.wait_msgs(msgs, 1)
    m = msgs[-1]
    assert (m.linear.x, m.linear.y, m.linear.z, m.angular.z) == \
        (0.0, 0.0, 0.0, 0.0)


def test_brushing_arm_timeout_to_returning(helpers):
    """BRUSHING 兜底超时（目标持续新鲜时）→ RETURNING。"""
    helpers.goto('BRUSHING')
    fsm, clock = helpers.fsm, helpers.clock
    # 模拟视觉 1Hz 持续喂目标，保持新鲜；30s 作业兜底 → RETURNING
    for i in range(31):
        clock.advance(1.0)
        helpers.send_target(0.3, 0.0, 0.0)
        helpers.pump(n=2)
        fsm._tick()
        if i < 29:
            assert fsm._state == 'BRUSHING'
    assert fsm._state == 'RETURNING'


def test_returning_timeout_back_to_searching_disarmed(helpers):
    helpers.goto('RETURNING')
    fsm, clock = helpers.fsm, helpers.clock
    clock.advance(5.0 + 0.01)
    fsm._tick()
    assert fsm._state == 'SEARCHING'
    assert fsm._armed is False   # 已解除武装
    # 解除武装后长时间无目标不会被看门狗拉入 ERROR
    clock.advance(10.0)
    fsm._tick()
    fsm._tick()
    assert fsm._state == 'SEARCHING'


def test_loss_in_brushing_goes_returning(helpers):
    """BRUSHING 期间目标丢失 → RETURNING（冻结语义）。"""
    helpers.goto('BRUSHING')
    fsm, clock = helpers.fsm, helpers.clock
    clock.advance(1.0 + 0.01)   # 越过短丢失窗口（<3s 看门狗）
    fsm._tick()
    assert fsm._state == 'RETURNING'


def test_loss_in_approaching_goes_searching_hover(helpers):
    """非作业态目标丢失 → SEARCHING 悬停再找（<3s 不触发 ERROR）。"""
    helpers.goto('APPROACHING')
    fsm, clock = helpers.fsm, helpers.clock
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    clock.advance(1.0 + 0.01)
    fsm._tick()
    assert fsm._state == 'SEARCHING'
    assert helpers.wait_msgs(msgs, 1)
    m = msgs[-1]
    assert (m.linear.x, m.linear.y, m.linear.z) == (0.0, 0.0, 0.0)


def test_reset_from_error_to_searching(helpers):
    """/fsm/reset 服务: 任意态（含 ERROR）→ SEARCHING 且解除武装。"""
    helpers.goto('ERROR')
    host, executor, fsm = helpers.host, helpers.executor, helpers.fsm
    client = host.create_client(Trigger, '/fsm/reset')
    assert helpers.pump_until(executor,
                              lambda: client.wait_for_service(0.05))
    fut = client.call_async(Trigger.Request())
    assert helpers.pump_until(executor, lambda: fut.done())
    assert fut.result().success is True
    assert fsm._state == 'SEARCHING'
    assert fsm._armed is False
    # 复位后不再被拉回 ERROR
    helpers.clock.advance(10.0)
    fsm._tick()
    assert fsm._state == 'SEARCHING'
