#!/usr/bin/env python3
"""
唯一任务状态机 mission_fsm_node（决策层，海鹰智巡）
==================================================

六态（冻结，HOVERING 已删除，与 docs/FROZEN_CONTROL_CHAIN.md 一致）:
  SEARCHING / TARGET_FOUND / APPROACHING / BRUSHING / RETURNING / ERROR

3 秒目标看门狗（组长指令 + PM 冻结）:
  目标首次有效出现后武装；>3s 未收到 /vision/target_point 有效数据
  → 立即切入 ERROR（10Hz 持续零速悬停，永不停止）。
  RETURNING 期间抑制（丢失已按作业语义处理）。
  接口底线: 视觉侧"丢失停止发布、绝不发 NaN"；本节点仍独立校验，
  非法样本（NaN/Inf/越界）丢弃且不刷新看门狗计时。

目标丢失映射（冻结 5529caf6 语义）:
  短丢失（超过 target_loss_timeout）→ 立即悬停:
    BRUSHING 丢失 → RETURNING；其余状态 → SEARCHING（悬停再找）。
  若持续无数据满 target_timeout → 看门狗升级 ERROR。

发布（决策层唯一发布者）:
  /system/current_state  String      10Hz 持续（六态；链路层消费:
                                     ERROR→SAFETY_HOLD, 其余解除 HOLD）
  /uav/cmd_vel           Twist       10Hz（正式契约: NED 系, ENU 计算后转换）
  /arm/target_pose       PoseStamped BRUSHING 期间按周期（frame=base_footprint,
                                     world→base_footprint TF 缺失则暂不下发）

订阅:
  /vision/target_point        PointStamped  可靠（视觉默认 QoS）
  /mavros/local_position/pose PoseStamped   BEST_EFFORT（或 /drone/pose_gt）
  /uav/flight_fault           String        飞控层故障上报 → ERROR

服务:
  /fsm/reset  std_srvs/Trigger  任意态（含 ERROR）→ SEARCHING，看门狗解除武装
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from std_msgs.msg import String
from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformListener

from .fsm_state import (APPROACHING, BRUSHING, ERROR, RETURNING, SEARCHING,
                        TARGET_FOUND, target_is_valid)
from .tf_util import (lookup_base_fallback, lookup_world_to_frame,
                      point_to_frame)


class MissionFsmNode(Node):
    """唯一任务状态机节点。回调只存数据，全部状态变更在 _tick 收敛。"""

    def __init__(self, node_name='mission_fsm_node', **kwargs):
        super().__init__(node_name, **kwargs)

        # ---- 参数（默认值 = 联合仿真正式配置，见 config/mission_fsm.yaml）----
        self.declare_parameter('rate', 10.0)                    # Hz; 0=禁用内部定时器（仅测试）
        self.declare_parameter('target_timeout', 3.0)           # 看门狗: >3s 无有效目标 → ERROR
        self.declare_parameter('target_loss_timeout', 1.0)      # 短丢失判定窗口（悬停再找）
        self.declare_parameter('approach_distance', 2.0)        # 满速接近阈值(m)
        self.declare_parameter('brush_distance', 0.5)           # 作业距离(m)
        self.declare_parameter('max_speed', 1.0)                # 最大接近速度(m/s)
        self.declare_parameter('slow_factor', 0.5)              # 近距减速系数
        self.declare_parameter('pose_source', 'mavros')         # mavros | gt
        self.declare_parameter('pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('pose_gt_topic', '/drone/pose_gt')
        self.declare_parameter('pose_timeout', 1.0)             # 位姿过期阈值(s)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('arm_frame', 'base_footprint')
        self.declare_parameter('arm_publish_period', 2.0)       # 机械臂位姿下发周期(s)
        self.declare_parameter('arm_exec_enable', False)        # 预留: 机械臂反馈（本次不订阅）
        self.declare_parameter('arm_exec_topic', '/arm/execution_status')
        self.declare_parameter('arm_exec_timeout', 30.0)        # BRUSHING 兜底超时 → RETURNING
        self.declare_parameter('returning_timeout', 5.0)        # RETURNING 保持时长 → SEARCHING
        self.declare_parameter('max_target_pos', 50.0)          # 目标单轴上限(m)
        self.declare_parameter('tf_timeout', 0.2)               # TF 查询超时(s)
        self.declare_parameter('tf_failure_policy', 'hold')     # hold | identity
        # 注: use_sim_time 由 rclpy 自动声明（Humble），无需显式 declare
        self.declare_parameter('target_topic', '/vision/target_point')
        self.declare_parameter('state_topic', '/system/current_state')
        self.declare_parameter('cmd_vel_topic', '/uav/cmd_vel')
        self.declare_parameter('arm_topic', '/arm/target_pose')
        self.declare_parameter('reset_service', '/fsm/reset')
        self.declare_parameter('flight_fault_topic', '/uav/flight_fault')

        p = self.get_parameter
        self._rate = float(p('rate').value)
        self._target_timeout = float(p('target_timeout').value)
        self._target_loss_timeout = float(p('target_loss_timeout').value)
        self._approach_dist = float(p('approach_distance').value)
        self._brush_dist = float(p('brush_distance').value)
        self._max_speed = float(p('max_speed').value)
        self._slow_factor = float(p('slow_factor').value)
        self._pose_timeout = float(p('pose_timeout').value)
        self._world_frame = str(p('world_frame').value)
        self._arm_frame = str(p('arm_frame').value)
        self._arm_publish_period = float(p('arm_publish_period').value)
        self._arm_exec_enable = bool(p('arm_exec_enable').value)
        self._arm_exec_topic = str(p('arm_exec_topic').value)
        self._arm_exec_timeout = float(p('arm_exec_timeout').value)
        self._returning_timeout = float(p('returning_timeout').value)
        self._max_target_pos = float(p('max_target_pos').value)
        self._tf_timeout = float(p('tf_timeout').value)
        self._tf_failure_policy = str(p('tf_failure_policy').value)

        # ---- 状态（回调只存数据，_tick 收敛状态变更）----
        self._state = SEARCHING
        self._armed = False                # 看门狗武装（首个有效目标出现后）
        self._last_target_time = 0.0
        self._latest_target = None         # PointStamped
        self._latest_pose = None           # PoseStamped
        self._last_pose_time = 0.0
        self._flight_fault_event = False   # 飞控层 ERROR 事件（粘滞，reset 清除）
        self._brush_started = 0.0
        self._returning_started = 0.0
        self._arm_publish_last = -1e9

        # 可注入时钟（单测注入手动时钟；运行时 = 单调时钟）
        self._time_fn = time.monotonic

        # ---- 订阅 ----
        self._target_sub = self.create_subscription(
            PointStamped, str(p('target_topic').value), self._on_target, 10)

        pose_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        pose_topic = str(p('pose_gt_topic').value) \
            if str(p('pose_source').value) == 'gt' \
            else str(p('pose_topic').value)
        self._pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self._on_pose, pose_qos)
        self.get_logger().info(f'位姿源: {pose_topic}')

        self._fault_sub = self.create_subscription(
            String, str(p('flight_fault_topic').value),
            self._on_flight_fault, 10)

        # ---- 发布 ----
        self._state_pub = self.create_publisher(
            String, str(p('state_topic').value), 10)
        self._cmd_pub = self.create_publisher(
            Twist, str(p('cmd_vel_topic').value), 10)
        self._arm_pub = self.create_publisher(
            PoseStamped, str(p('arm_topic').value), 10)

        # ---- 服务 ----
        self._reset_srv = self.create_service(
            Trigger, str(p('reset_service').value), self._on_reset)

        # ---- TF ----
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- 定时器 ----
        if self._rate > 0.0:
            self.create_timer(1.0 / self._rate, self._tick)

        self.get_logger().info(
            f'唯一任务状态机启动 | 六态 | 看门狗 {self._target_timeout}s | '
            f'接近 {self._approach_dist}m | 作业 {self._brush_dist}m | '
            f'TF 策略 {self._tf_failure_policy}')

    # ------------------------------------------------------------------
    # 回调（只存数据）
    # ------------------------------------------------------------------
    def _on_target(self, msg):
        """目标样本：非法丢弃且不刷新计时（接口底线防御）。"""
        if not target_is_valid(msg.point, self._max_target_pos):
            self.get_logger().warn(
                '丢弃无效目标样本（NaN/Inf/越界），不刷新看门狗计时',
                throttle_duration_sec=2.0)
            return
        self._latest_target = msg
        self._last_target_time = self._time_fn()
        if not self._armed:
            self._armed = True
            self.get_logger().info('首个有效目标出现，3s 看门狗武装')

    def _on_pose(self, msg):
        p = msg.pose.position
        if not (math.isfinite(p.x) and math.isfinite(p.y)
                and math.isfinite(p.z)):
            self.get_logger().warn('丢弃非法位姿样本',
                                   throttle_duration_sec=2.0)
            return
        self._latest_pose = msg
        self._last_pose_time = self._time_fn()

    def _on_flight_fault(self, msg):
        if msg.data == 'ERROR':
            self._flight_fault_event = True
            self.get_logger().warn('飞控层上报 /uav/flight_fault=ERROR')

    def _on_reset(self, request, response):
        self._disarm()
        self._latest_target = None
        self._flight_fault_event = False
        self._set_state(SEARCHING)
        self.get_logger().info('/fsm/reset: 任意态 → SEARCHING')
        response.success = True
        response.message = 'FSM reset to SEARCHING'
        return response

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _tick(self):
        now = self._time_fn()

        # 1) 飞控层故障 → ERROR（安全优先，覆盖一切任务状态）
        if self._flight_fault_event and self._state != ERROR:
            self.get_logger().warn('飞控层故障 → ERROR')
            self._set_state(ERROR)

        # 2) 3s 目标看门狗（RETURNING 抑制；丢失协议见下）
        if (self._armed and self._state not in (RETURNING, ERROR)
                and now - self._last_target_time > self._target_timeout):
            self.get_logger().warn(
                f'目标看门狗超时（>{self._target_timeout}s 无有效目标）→ ERROR')
            self._set_state(ERROR)

        # 3) 短丢失映射（冻结语义: 作业后丢→RETURNING，其余→SEARCHING 悬停再找）
        has_fresh = (self._armed
                     and now - self._last_target_time
                     <= self._target_loss_timeout)
        if not has_fresh and self._state not in (SEARCHING, RETURNING, ERROR):
            if self._state == BRUSHING:
                self.get_logger().warn('BRUSHING 期间目标丢失 → RETURNING')
                self._set_state(RETURNING)
                self._returning_started = now
            else:
                self._set_state(SEARCHING)
            self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)
            self._publish_state()
            return

        # 4) 状态动作
        if self._state == SEARCHING:
            if has_fresh:
                self._set_state(TARGET_FOUND)
            self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)
        elif self._state == TARGET_FOUND:
            # 1 拍后进入接近
            self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)
            self._set_state(APPROACHING)
        elif self._state == APPROACHING:
            self._act_approaching(now)
        elif self._state == BRUSHING:
            self._act_brushing(now)
        elif self._state == RETURNING:
            self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)
            if now - self._returning_started > self._returning_timeout:
                self.get_logger().info('RETURNING 超时 → SEARCHING（解除武装）')
                self._disarm()
                self._set_state(SEARCHING)
        elif self._state == ERROR:
            # 10Hz 持续零速永不停止（保 offboard 心跳；链路层 SAFETY_HOLD 兜底）
            self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)

        self._publish_state()

    def _act_approaching(self, now):
        pos = self._drone_pos_world(now)
        if pos is None:
            # 位姿缺失/过期 → 悬停（hold），不盲飞
            self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)
            return
        t = self._latest_target.point
        dx, dy, dz = t.x - pos[0], t.y - pos[1], t.z - pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= self._brush_dist or dist <= 1e-9:
            self._set_state(BRUSHING)
            self._brush_started = now
            self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)
            return
        if dist >= self._approach_dist:
            speed = self._max_speed
        else:
            ratio = (dist - self._brush_dist) \
                / (self._approach_dist - self._brush_dist)
            speed = self._max_speed * ratio * self._slow_factor
        self._publish_cmd_vel_enu(dx / dist * speed, dy / dist * speed,
                                  dz / dist * speed, 0.0)

    def _act_brushing(self, now):
        self._publish_cmd_vel_enu(0.0, 0.0, 0.0, 0.0)
        # 机械臂目标位姿（TF 缺失 → hold 跳过）
        if now - self._arm_publish_last >= self._arm_publish_period:
            self._publish_arm_pose()
            self._arm_publish_last = now
        # 机械臂反馈只留参数不实现（arm_exec_enable）——
        # 本次 BRUSHING→RETURNING 用超时兜底：
        if now - self._brush_started > self._arm_exec_timeout:
            self.get_logger().info(
                f'机械臂作业兜底超时（{self._arm_exec_timeout}s）→ RETURNING')
            self._set_state(RETURNING)
            self._returning_started = now

    # ------------------------------------------------------------------
    # 输出辅助
    # ------------------------------------------------------------------
    def _publish_state(self):
        """持续发布当前状态（10Hz），防下游丢消息。"""
        msg = String()
        msg.data = self._state
        self._state_pub.publish(msg)

    def _publish_cmd_vel_enu(self, vx, vy, vz, yaw_rate=0.0):
        """发布 /uav/cmd_vel（正式契约: NED 系）。

        输入（/vision/target_point 与位姿）为 world/ENU，
        发布前转换: v_ned = (v_enu.y, v_enu.x, -v_enu.z); wz_ned = -wz_enu。
        """
        msg = Twist()
        msg.linear.x = vy
        msg.linear.y = vx
        msg.linear.z = -vz
        msg.angular.z = -yaw_rate
        self._cmd_pub.publish(msg)

    def _publish_arm_pose(self):
        """BRUSHING 期间下发 /arm/target_pose（frame=base_footprint）。

        world→base_footprint 回退链: 直达 → base_link → drone_base_link。
        TF 全部缺失 → 暂不下发（hold），不发布脏坐标。
        """
        if self._latest_target is None:
            return
        tf = lookup_base_fallback(self._tf_lookup, self._world_frame,
                                  self._arm_frame)
        if tf is None:
            self.get_logger().warn(
                f'TF {self._world_frame}→{self._arm_frame} 缺失，'
                '暂不下发机械臂位姿（hold）', throttle_duration_sec=2.0)
            return
        x, y, z = point_to_frame(self._latest_target, tf)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._arm_frame
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self._arm_pub.publish(msg)

    # ------------------------------------------------------------------
    # TF / 位姿辅助
    # ------------------------------------------------------------------
    def _tf_lookup(self, target_frame, source_frame):
        """TF 查询封装（单测可 monkeypatch 注入替身）。"""
        try:
            return self._tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(),
                rclpy.time.Duration(seconds=self._tf_timeout))
        except Exception:
            return None

    def _drone_pos_world(self, now):
        """无人机当前位置（world 系）或 None（缺失/过期/换算失败→hold）。"""
        pose = self._latest_pose
        if pose is None or now - self._last_pose_time > self._pose_timeout:
            return None
        pos = pose.pose.position
        if pose.header.frame_id in ('', self._world_frame):
            return (pos.x, pos.y, pos.z)
        tf = lookup_world_to_frame(self._tf_lookup, self._world_frame,
                                   pose.header.frame_id)
        if tf is not None:
            stamped = PointStamped()
            stamped.header = pose.header
            stamped.point = pose.pose.position
            return point_to_frame(stamped, tf)
        if pose.header.frame_id == 'map':
            # 链路节点位姿别名 frame=map（ENU 世界对齐）→ 恒等近似
            return (pos.x, pos.y, pos.z)
        if self._tf_failure_policy == 'identity':
            return (pos.x, pos.y, pos.z)
        return None

    # ------------------------------------------------------------------
    # 状态工具
    # ------------------------------------------------------------------
    def _set_state(self, new_state):
        if self._state != new_state:
            self.get_logger().info(f'状态切换: {self._state} → {new_state}')
            self._state = new_state

    def _disarm(self):
        self._armed = False
        self._last_target_time = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = MissionFsmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
