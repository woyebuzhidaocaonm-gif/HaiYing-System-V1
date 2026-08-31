#!/usr/bin/env bash
# ============================================================
# 海鹰智巡 V9.2 联合仿真 + 视觉链一键启动（PM 冻结方案）
#   普通 RGB 相机 + YOLO + 雷达点云定位（无深度相机）
# ============================================================
# 启动顺序:
#   1. PX4 SITL（飞控仿真）
#   2. V9.2 联合仿真（Gazebo: 四旋翼 + SO-101 + AR0234 + MID-360）
#   3. 视觉链（yolo_detector + target_localizer）
#
# 用法:
#   export PX4_AUTOPILOT_DIR=/绝对路径/PX4-Autopilot
#   bash run_all.sh
#
# 要求: 已按 simulation/haiying_v9_2/README.md 完成 colcon build。
# ============================================================
set -euo pipefail

source /opt/ros/humble/setup.bash

PX4_AUTOPILOT_DIR="${PX4_AUTOPILOT_DIR:-}"
if [ -z "$PX4_AUTOPILOT_DIR" ]; then
    echo "[run_all] 错误: 请先 export PX4_AUTOPILOT_DIR=/绝对路径/PX4-Autopilot"
    exit 1
fi

PX4_BUILD_DIR="$PX4_AUTOPILOT_DIR/build/px4_sitl_default"
if [ ! -f "$PX4_BUILD_DIR/bin/px4" ]; then
    echo "[run_all] 错误: 未找到 $PX4_BUILD_DIR/bin/px4，请先编译 PX4 SITL"
    exit 1
fi

# --- 1. PX4 SITL（后台） ---
echo "[run_all] 1/3 启动 PX4 SITL..."
(
    cd "$PX4_BUILD_DIR"
    PX4_SYS_AUTOSTART=4002 ./bin/px4 -d &
    echo $! > /tmp/run_all_px4.pid
)
sleep 5

# --- 2. V9.2 联合仿真（后台） ---
echo "[run_all] 2/3 启动 V9.2 联合仿真（pause:=false，PX4 就绪后物理恢复）..."
ros2 launch haiying_v9_2 v9_2_simulation.launch.py pause:=false &
echo $! > /tmp/run_all_gazebo.pid

# --- 3. 视觉链 ---
echo "[run_all] 3/3 启动视觉链（yolo_detector + target_localizer）..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ros2 launch "$SCRIPT_DIR/launch/v9_2_vision.launch.py"

# ros2 launch 退出后收尾
trap 'kill $(cat /tmp/run_all_px4.pid 2>/dev/null) $(cat /tmp/run_all_gazebo.pid 2>/dev/null) 2>/dev/null || true' EXIT
