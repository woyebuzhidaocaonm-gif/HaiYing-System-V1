# 决策与状态机组代码

## haiying_mission — 唯一任务状态机（正式实现）

六态任务状态机 `mission_fsm_node`：SEARCHING / TARGET_FOUND / APPROACHING /
BRUSHING / RETURNING / ERROR（冻结，HOVERING 已删除）。

- **3 秒目标看门狗**（组长指令）：首个有效目标出现后武装，>3s 无有效
  `/vision/target_point` → ERROR（10Hz 持续零速悬停）；RETURNING 抑制；
  NaN/Inf/越界样本丢弃且不刷新计时。
- **唯一发布** `/system/current_state`、`/uav/cmd_vel`（NED）、
  `/arm/target_pose`（BRUSHING，base_footprint 系）。
- **机械臂反馈只留参数**：`arm_exec_enable`/`arm_exec_topic` 预留，
  BRUSHING→RETURNING 暂用 30s 超时兜底。
- 设计文档：[docs/状态机设计.md](haiying_mission/docs/状态机设计.md)

构建 / 测试 / 启动：

```bash
colcon build --packages-select haiying_mission
source install/setup.bash
python3 -m pytest haiying_mission/tests -v        # 28 项单测（需 ROS 2 Humble 环境）
ros2 launch haiying_mission mission_fsm.launch.py # 决策层单独启动
```

> 正式联仿拓扑见 docs/FROZEN_CONTROL_CHAIN.md：冻结链
> （`attitude_cmd freeze_chain.launch.py`）与决策层分别启动。

## 历史遗留

- `*.tar` / `*.tar.gz`：早期 FSM/机械臂代码归档（仅参考，不参与联仿）；
- `scripts/approach_controller.py`（上层 scripts/ 目录）：legacy 单体调试
  控制器，已被本包取代，联合仿真不启动。
