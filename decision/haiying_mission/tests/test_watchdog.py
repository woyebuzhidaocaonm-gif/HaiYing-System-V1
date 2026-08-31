"""3 秒目标看门狗测试（组长指令核心，5 项）。"""
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def test_watchdog_errors_after_3s(helpers):
    """目标出现后 3s 无有效数据 → ERROR + 零速。"""
    helpers.goto('TARGET_FOUND')
    fsm, clock = helpers.fsm, helpers.clock
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    clock.advance(3.0 + 0.01)
    fsm._tick()
    assert fsm._state == 'ERROR'
    fsm._tick()
    assert helpers.wait_msgs(msgs, 1)
    m = msgs[-1]
    assert (m.linear.x, m.linear.y, m.linear.z) == (0.0, 0.0, 0.0)


def test_watchdog_not_armed_no_error(helpers):
    """从未出现目标 → 看门狗未武装，永不 ERROR。"""
    fsm, clock = helpers.fsm, helpers.clock
    clock.advance(100.0)
    for _ in range(3):
        fsm._tick()
    assert fsm._state == 'SEARCHING'


def test_watchdog_suppressed_in_returning(helpers):
    """RETURNING 期间看门狗抑制（丢失已按作业语义处理）。"""
    helpers.goto('RETURNING')
    fsm, clock = helpers.fsm, helpers.clock
    clock.advance(4.0)   # > 3s 看门狗阈值、< 5s RETURNING 阈值
    fsm._tick()
    assert fsm._state == 'RETURNING'


def test_error_zero_velocity_persists(helpers):
    """ERROR 下 10Hz 持续零速永不停止（保 offboard 心跳）。"""
    helpers.goto('ERROR')
    fsm, clock = helpers.fsm, helpers.clock
    msgs = helpers.collector('/uav/cmd_vel', Twist)
    for _ in range(5):
        clock.advance(0.1)
        fsm._tick()
    assert helpers.wait_msgs(msgs, 5)
    for m in msgs[-5:]:
        assert (m.linear.x, m.linear.y, m.linear.z, m.angular.z) == \
            (0.0, 0.0, 0.0, 0.0)
    assert fsm._state == 'ERROR'


def test_flight_fault_forces_error(helpers):
    """飞控层 /uav/flight_fault=ERROR → 任务级 ERROR。"""
    fsm = helpers.fsm
    msg = String()
    msg.data = 'ERROR'
    helpers.mock_fault.publish(msg)
    assert helpers.pump_until(helpers.executor,
                              lambda: fsm._flight_fault_event)
    fsm._tick()
    assert fsm._state == 'ERROR'
