"""haiying_mission 单测共享装置（pytest）。

环境: ROS_LOCALHOST_ONLY=1 + 随机 ROS_DOMAIN_ID（套件间互不干扰）。
驱动方式: 手动时钟 ManualClock 注入 fsm._time_fn；rate=0 禁用内部定时器，
测试直接调用 fsm._tick() 步进状态机；订阅/发布经 SingleThreadedExecutor
spin_once 泵送。TF 经 fsm._tf_lookup monkeypatch 注入替身。
"""
import os
import random
import time
import uuid
from types import SimpleNamespace

os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
os.environ['ROS_DOMAIN_ID'] = str(random.randint(10, 200))

import pytest  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from geometry_msgs.msg import PointStamped, PoseStamped  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from haiying_mission.mission_fsm_node import MissionFsmNode  # noqa: E402


class ManualClock:
    """可步进的手动时钟（注入 fsm._time_fn）。"""

    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


# ---------------------------------------------------------------------------
# 基础装置
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def executor(ros):
    ex = SingleThreadedExecutor()
    yield ex
    ex.shutdown()


@pytest.fixture
def host(ros, executor):
    node = rclpy.create_node('host_' + uuid.uuid4().hex[:8])
    executor.add_node(node)
    yield node
    executor.remove_node(node)
    node.destroy_node()


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def fsm(ros, executor, clock):
    node = MissionFsmNode(parameter_overrides=[
        rclpy.Parameter('rate', value=0.0),   # 禁用内部定时器，测试手动步进
    ])
    node._time_fn = clock
    node._tf_lookup = lambda target, source: None   # 默认无 TF（用例内按需替换）
    executor.add_node(node)
    yield node
    executor.remove_node(node)
    node.destroy_node()


@pytest.fixture
def mock_vision(host):
    return host.create_publisher(PointStamped, '/vision/target_point', 10)


@pytest.fixture
def mock_pose(host):
    return host.create_publisher(PoseStamped, '/mavros/local_position/pose', 10)


@pytest.fixture
def mock_fault(host):
    return host.create_publisher(String, '/uav/flight_fault', 10)


# ---------------------------------------------------------------------------
# 泵送 / 发送 / 状态驱动助手
# ---------------------------------------------------------------------------
def pump_until(executor, cond, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if cond():
            return True
    return False


def pump(executor, n=3, timeout_sec=0.02):
    for _ in range(n):
        executor.spin_once(timeout_sec=timeout_sec)


def send_target(pub, x, y, z, frame='world'):
    msg = PointStamped()
    msg.header.frame_id = frame
    msg.point.x, msg.point.y, msg.point.z = x, y, z
    pub.publish(msg)
    return msg


def send_pose(pub, x, y, z, frame='world'):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
    msg.pose.orientation.w = 1.0
    pub.publish(msg)
    return msg


def make_collector(host, topic, msg_type):
    msgs = []
    host.create_subscription(msg_type, topic, msgs.append, 100)
    return msgs


def goto(fsm, executor, clock, target_pub, pose_pub, state):
    """从初始态把状态机驱动到指定状态（每个测试独立使用）。

    - TARGET_FOUND / APPROACHING: 目标 (10,0,0)，无需位姿
    - BRUSHING / RETURNING: 目标 (0.3,0,0) + 位姿 (0,0,0)
    - ERROR: 目标出现后时钟越过 3s 看门狗
    """
    if state == 'TARGET_FOUND':
        send_target(target_pub, 10.0, 0.0, 0.0)
        assert pump_until(executor,
                          lambda: fsm._latest_target is not None)
        fsm._tick()
        assert fsm._state == 'TARGET_FOUND'
    elif state == 'APPROACHING':
        goto(fsm, executor, clock, target_pub, pose_pub, 'TARGET_FOUND')
        fsm._tick()
        assert fsm._state == 'APPROACHING'
    elif state == 'BRUSHING':
        send_target(target_pub, 0.3, 0.0, 0.0)
        send_pose(pose_pub, 0.0, 0.0, 0.0)
        assert pump_until(executor, lambda: fsm._latest_target is not None
                          and fsm._latest_pose is not None)
        fsm._tick()   # → TARGET_FOUND
        fsm._tick()   # → APPROACHING
        fsm._tick()   # → BRUSHING（0.3m ≤ 作业圈 0.5m）
        assert fsm._state == 'BRUSHING'
    elif state == 'RETURNING':
        goto(fsm, executor, clock, target_pub, pose_pub, 'BRUSHING')
        clock.advance(1.0 + 0.01)   # 越过短丢失窗口 → BRUSHING 丢目标映射
        fsm._tick()
        assert fsm._state == 'RETURNING'
    elif state == 'ERROR':
        goto(fsm, executor, clock, target_pub, pose_pub, 'TARGET_FOUND')
        clock.advance(3.0 + 0.01)
        fsm._tick()
        assert fsm._state == 'ERROR'
    else:
        raise ValueError(f'未知目标状态: {state}')


@pytest.fixture
def helpers(executor, host, fsm, clock, mock_vision, mock_pose, mock_fault):
    h = SimpleNamespace()
    h.executor = executor
    h.host = host
    h.fsm = fsm
    h.clock = clock
    h.mock_vision = mock_vision
    h.mock_pose = mock_pose
    h.mock_fault = mock_fault
    h.pump_until = pump_until
    h.pump = lambda n=3: pump(executor, n)
    h.send_target = lambda x, y, z, frame='world': \
        send_target(mock_vision, x, y, z, frame)
    h.send_pose = lambda x, y, z, frame='world': \
        send_pose(mock_pose, x, y, z, frame)
    h.collector = lambda topic, msg_type: make_collector(host, topic, msg_type)
    h.wait_msgs = lambda msgs, n, timeout=2.0: \
        pump_until(executor, lambda: len(msgs) >= n, timeout)
    h.goto = lambda state: goto(fsm, executor, clock,
                                mock_vision, mock_pose, state)
    return h
