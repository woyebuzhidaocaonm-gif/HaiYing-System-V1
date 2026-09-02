# 视觉组代码 (Vision Module)

## V9.2视觉运行资产

YOLOv5源码和训练权重不提交到本仓库。运行前设置：

    export YOLOV5_ROOT=/absolute/path/to/yolov5
    export HAIYING_YOLO_WEIGHTS=/absolute/path/to/best.pt
    export HAIYING_YOLO_DEVICE=cpu
    export PYTHONPATH="$YOLOV5_ROOT${PYTHONPATH:+:$PYTHONPATH}"

当前验收基线：

- YOLOv5 commit：`915bbf294bb74c859f0b41f1c23bc395014ea679`
- `best.pt` SHA256：`68717b5dc6cc2bf2b53eb53f63d79b18d1731170fa74bdb3ba6ce197ad5d77d8`
- 输入尺寸：`640×640`
- 类别顺序：`craze, corrosion, surface_injure, thunderstrike, crack, hide_craze`
- ROS 2 Humble CPU验证：PyTorch `2.0.1+cpu`、torchvision `0.15.2+cpu`


海鹰智巡 — 视觉感知模块 V2

## 文件说明

| 文件 | 任务 | 功能 |
|------|------|------|
| `yolo_detector.py` | Task 26, 71 | YOLOv5 GPU推理 → `/vision/detection`（header 透传图像采集时间，frame=ar0234_camera_optical_frame） |
| `target_localizer.py` | Task 27, 72, V9.2 | 检测框 + LiDAR点云投影定位 → `/vision/target_point`（world, 米） |
| `lidar_projection.py` | V9.2 新增 | 点云→图像投影核心：mid360 点云变换、**ar0234_camera_link→optical 坐标转换实际代码**、K 投影、稳健深度查询（纯 numpy，可离线单测） |
| `calibration.py` | Task 29 | 相机-LiDAR外参标定TF【V9.2 正式链不启动，外参由 URDF TF 提供】 |
| `approach_controller.py` | 动作3 | 【LEGACY】接近状态机，已被决策组 `mission_fsm_node` 取代，联合仿真不启动 |
| `gz_camera_bridge.py` | 辅助 | Gazebo RGB相机 → ROS2【V9.2 正式链不启动：V9.2 相机插件原生发布】 |
| `gz_depth_bridge.py` | 辅助 | Gazebo 深度相机 → ROS2 (32FC1)【已退出正式启动链】 |
| `tf_publisher.py` | 辅助 | 旧帧名 TF 发布【V9.2 不启动：由 V9.2 launch/仿真组发布 TF】 |
| `lidar_bridge.py` | 辅助 | Gazebo LiDAR → ROS2 PointCloud2（frame=mid360_link，仿真时间透传）【V9.2 正式链不启动：V9.2 雷达插件原生发布】 |
| `live_turbine_5shots.py` | 验收 | Gazebo实时检测，保存5张不同缺陷/视角截图 |
| `live_turbine_detect.py` | 验收 | 实时检测 + 相机多视角扫描，检出即存截图 |
| `test_turbine_model.py` | 验收 | 模型自检：训练集/实时画面跑检测 |
| `collect_turbine_data.py` | 数据 | Gazebo渲染图采集 + 缺陷坐标投影自动标注YOLO标签 |
| `pose_gt_bridge.py` | 辅助 | Gazebo真值位姿 → `/drone/pose_gt`（MAVROS EKF不可靠时的备选位姿源） |
| `launch/v9_2_vision.launch.py` | V9.2 新增 | 视觉链正式启动文件（yolo_detector + target_localizer） |
| `params/v9_2_vision.yaml` | V9.2 新增 | V9.2 正式参数（含 publish_lidar_tf=false、max_time_diff=0.5） |
| `run_all.sh` | V9.2 新增 | PX4 SITL + V9.2 联合仿真 + 视觉链一键启动 |
| `tests/` | V9.2 新增 | 5 项离线测试（无 ROS 环境可跑，见 tests/README.md） |
| `wind_turbine_interfaces/` | 接口包 | 自定义消息/服务 (DefectDetection, DefectDetectionArray, StartInspection)，`yolo_detector.py`/`target_localizer.py` 依赖，运行前需 colcon build |

## 运行方式（V9.2 正式链，PM 冻结：RGB + 雷达点云定位，无深度相机）

```bash
# 0. 构建（首次）
colcon build --base-paths simulation \
  --packages-select so-101_description arm_uav_joint haiying_v9_2 --symlink-install
source install/setup.bash

# 1. 一键启动：PX4 SITL + V9.2 联合仿真 + 视觉链
export PX4_AUTOPILOT_DIR=/绝对路径/PX4-Autopilot
bash run_all.sh

# 或手动分步：
# 1) PX4 SITL
#    cd $PX4_AUTOPILOT_DIR/build/px4_sitl_default && PX4_SYS_AUTOSTART=4002 ./bin/px4 -d
# 2) V9.2 联合仿真（Gazebo + 四旋翼 + SO-101 + AR0234 + MID-360；含全部传感器静态 TF）
#    ros2 launch haiying_v9_2 v9_2_simulation.launch.py pause:=false
# 3) 视觉链
#    ros2 launch <本目录>/launch/v9_2_vision.launch.py
```

- **话题复用（V9.2 直接发布，不启动桥接器）**：`/drone/camera/image_raw`、`/drone/camera/camera_info`、`/drone/lidar/points`
- **TF 复用（不发布任何外参）**：`world → base_footprint`（仿真组动态 TF）、`base_footprint → ar0234_camera_link → ar0234_camera_optical_frame`、`base_footprint → mid360_link`（V9.2 launch 静态 TF）；正式配置 `publish_lidar_tf=false`
- 旧 gz-sensors 链路（调试用，非正式）才需桥接器：`gz_camera_bridge.py`、`lidar_bridge.py`、`tf_publisher.py`；深度相机 `gz_depth_bridge.py` 已退出

## 话题接口

| 话题 | 类型 | 方向 |
|------|------|------|
| `/vision/detection` | DefectDetectionArray | 发布：YOLO检测结果 |
| `/vision/target_point` | PointStamped | 发布：目标绝对XYZ坐标 (world) |

> 决策组话题（视觉组不再发布）：`/arm/target_pose` (PoseStamped)、`/uav/cmd_vel` (Twist)、
> `/system/current_state` (String) 由决策组 `mission_fsm_node` 唯一发布，详见下文「最终节点职责确认」。

### `/vision/target_point` 接口约定（V9.2 PM 冻结）

- **frame**: `world`（`world_frame` 参数可配），**单位**: 米（绝对 XYZ）
- **频率**: 事件驱动，随检测逐帧发布（仿真链路 ≈ 相机帧率 3.3 FPS），无固定频率/心跳
- **时间戳**: 与图像采集时间一致（`header` 从 `/drone/camera/image_raw` → `/vision/detection` 透传，目标点发布时 `stamp = 检测消息 header.stamp`，不被本节点 `now()` 覆盖）
- **图像与点云时间同步**: 点云 `header.stamp` 与检测 `header.stamp` 最大差 0.5 秒（`max_time_diff` 参数，PM 冻结），**超过则拒绝本次定位**（该帧不发布）；零时间戳视为不可考，同样拒绝
- **目标丢失协议**: **停止发布**（不发布 NaN、无有效位标志）。消费端以 3 秒超时兜底（决策组 `mission_fsm_node` 已实现；`approach_controller.py` 为 legacy，不参与联合仿真）

### ar0234_camera_link → optical 坐标转换（实际代码）

按 CameraInfo 内参投影前，必须先把雷达点云从 link 系（FLU：X前/Y左/Z上）转到 optical 系（RDF：Z前/X右/Y下）。
实际代码在 `lidar_projection.py`：

```python
# REP-103: optical = R_link2optical @ p_link，其中
R_link2optical = np.array([[0., -1., 0.],
                           [0.,  0., -1.],
                           [1.,  0., 0.]])   # 与 V9.2 静态 TF rpy(-pi/2, 0, -pi/2) 等价

def link_to_optical(points_link):          # (N,3) link系 → optical系
    return points_link @ R_link2optical.T  # 批量旋转：列向量版 p_opt = R @ p_link

def optical_to_link(points_optical):       # 逆变换（R 正交，转置即逆）
    return points_optical @ R_link2optical
```

之后才用 CameraInfo 的 K 投影（`project_optical_to_pixel`），深度反查时按像素邻域在 optical 系点云中稳健取深度（`query_lidar_depth`）。
离线测试 `tests/` 第 1 项对该矩阵与 TF 旋转 `Rz(-pi/2) @ Rx(-pi/2)` 的等价性做了断言。

## 最终节点职责确认（2026-08-28）

视觉组已与组长确认最终节点职责，结论如下：

1. `target_localizer.py` 只发布 `/vision/target_point`，不再发布 `/arm/target_pose`（发布器、`arm_frame` 参数与发布代码已删除）。
2. 最终联合仿真不启动 `approach_controller.py`；`/system/current_state`、`/uav/cmd_vel`、`/arm/target_pose` 由决策组 `mission_fsm_node` 唯一发布。`approach_controller.py` 保留于仓库并标记 legacy。
3. 决策组 FSM 待改事项（转达）：
   - `/arm/target_pose` 的 `frame_id` 需为 `base_footprint`（当前为 `world`）；
   - BRUSHING 后需等待 `/arm/execution_status`：`EXEC_DONE` → RETURNING，`EXEC_FAIL` → ERROR（当前不订阅该话题、立即切 RETURNING）；
   - `/arm/execution_status` 目前全仓库无人发布，需机械臂桥接（`control/haiying_zhixun_bridge/ros_node.py`，当前纯订阅）新增 EXEC_DONE/EXEC_FAIL 发布。
4. 系统状态集删除 HOVERING，最终六态：SEARCHING / TARGET_FOUND / APPROACHING / BRUSHING / RETURNING / ERROR（决策组 FSM 现状即为六态）。
5. ERROR 状态下 `/uav/cmd_vel` 持续发布零速度 Twist（与 FSM 及 `approach_controller.py` 现状一致），不采用「停止发布 + 底盘超时」方案。

## 依赖

- ROS2 Humble
- YOLOv5 + PyTorch (CUDA)
- Gazebo Transport 13 Python绑定
- cv_bridge, OpenCV
- tf2_ros, MAVROS

先编译接口包再运行脚本：

```bash
cd <ros2_ws>
colcon build --packages-select wind_turbine_interfaces
source install/setup.bash
```
