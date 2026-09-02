# 视觉目标三维解算节点

## AR0234 鱼眼相机校正

本包已内置本机 27 张标定图求得的 `1920x1080` equidistant 内参。校正节点订阅
`/camera/image_raw`，发布 `/camera/image_rect` 与
`/camera/camera_info_rect`。默认新焦距为 700 px，约对应 108° 水平视场；映射表
只在启动时生成一次，输出图像与 `CameraInfo` 使用完全相同的原始时间戳。
默认还会丢弃时间戳重复或倒退的 USB 帧，防止下游同步器收到乱序图像。

在连接相机的原生 Ubuntu 22.04/ROS 2 Humble 机器上安装驱动、构建并启动：

```bash
sudo apt update
sudo apt install ros-humble-usb-cam ros-humble-cv-bridge ros-humble-rqt-image-view \
  python3-opencv python3-yaml v4l-utils
mkdir -p ~/ros2_ws/src
cp -r /path/to/HaiYing-System-V1/vision/haiying_vision_3d ~/ros2_ws/src/
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select haiying_vision_3d --symlink-install
source install/setup.bash
ros2 launch haiying_vision_3d ar0234_rectification.launch.py video_device:=/dev/video0
```

若已有其他相机驱动发布 `/camera/image_raw`：

```bash
ros2 launch haiying_vision_3d ar0234_rectification.launch.py start_camera:=false
```

实时验收：

```bash
ros2 topic hz /camera/image_rect
ros2 topic echo /camera/camera_info_rect --once
ros2 run rqt_image_view rqt_image_view
```

在 `rqt_image_view` 左上角选择 `/camera/image_rect`。

## 临时理想相机—MID-360 外参

在实物联合标定完成前，可用 V9.2 仿真安装位姿联调投影链。该配置明确标记为
`ideal`，不会覆盖默认的真实标定安全配置：

```bash
ros2 launch haiying_vision_3d ideal_lidar_projection.launch.py
```

它假设 MID-360 使用 FLU 坐标（X 前、Y 左、Z 上），并使用
`T_ar0234_optical_mid360 = [R|t]`：

```text
[ 0 -1  0   0.00 ]
[ 0  0 -1  -0.20 ]
[ 1  0  0  -0.15 ]
[ 0  0  0   1.00 ]
```

这只用于当前连通性和投影方向验证，不能作为实物测量结果或比赛精度数据。真实外参
标定后应恢复 `target_point.yaml` 的安全门并使用实测 TF。

## YOLO 检测框内 MID-360 点云测距

`yolo_lidar_fusion_node` 在同一个节点内完成以下处理：

1. 从 `/camera/image_rect` 读取使用标定内参校正后的图像；
2. 使用训练得到的 `best.pt` 或对应的 `best.onnx` 产生检测框；
3. 使用 `/camera/camera_info_rect` 中的内参和配置的 4×4 外参，将
   `/livox/lidar` 投影到图像；
4. 提取每个检测框内部的点簇，以深度中位数和 MAD 门限剔除离群点；
5. 输出框选点云、三维坐标、欧氏距离以及用于计算的有效点数。

本机已有的 `yolov5/output/best.onnx` 是训练权重的部署版本。Ubuntu CPU 部署推荐
使用该文件，因为只依赖 OpenCV；如果直接加载 `.pt`，需要额外安装 PyTorch 和
Ultralytics。

确认相机校正节点和 MID-360 驱动已经运行后，启动理想外参融合：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch haiying_vision_3d yolo_lidar_fusion_ideal.launch.py \
  model_path:=/absolute/path/to/best.onnx
```

直接使用 Ultralytics 格式的 `best.pt`：

```bash
python3 -m pip install --user ultralytics
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch haiying_vision_3d yolo_lidar_fusion_ideal.launch.py \
  model_path:=/absolute/path/to/best.pt device:=cpu
```

有 NVIDIA CUDA 环境时可把 `device:=cpu` 改为 `device:=0`。输出话题为：

- `/vision/yolo_lidar_image`：检测框、框内雷达点和距离标注图；
- `/vision/target_point`：最高置信度且有有效点簇的目标 XYZ，坐标系为
  `livox_frame`；
- `/vision/target_distance`：目标到雷达原点的欧氏距离，单位米；
- `/vision/target_point_count`：距离计算实际使用的内点数量。
- `/vision/target_cloud`：从最佳检测框中提取并剔除离群点后的目标点簇，可直接在
  RViz 中显示。

查看标注图和数值：

```bash
ros2 run rqt_image_view rqt_image_view /vision/yolo_lidar_image
ros2 topic echo /vision/target_distance
ros2 topic echo /vision/target_point
ros2 topic echo /vision/target_point_count
```

若检测框显示 `no lidar points`，依次检查相机和点云时间差、两传感器视场是否重合、
外参方向，以及 `minimum_points`。理想外参只能用于系统联调；其数值不是现场实测值，
不能据此声明测距精度。

必须确认输出稳定接近 30 Hz、分辨率为 `1920x1080`、直线不再呈鱼眼弯曲，并且
`CameraInfo.k` 为 `[700, 0, 960, 0, 700, 540, 0, 0, 1]`。若输入不是标定时的
`1920x1080`，节点会明确报错并丢帧，不能通过缩放图像绕过标定分辨率。

该 ROS 2 包把检测框中心像素与真实传感器测量融合，并严格按照全局接口规范发布：

- 输入目标像素：`/vision/target_pixel`，`geometry_msgs/msg/PointStamped`
  - `point.x`：图像横坐标 `u`
  - `point.y`：图像纵坐标 `v`
  - `point.z`：可存放置信度，本节点不参与三维计算
- 输出目标坐标：`/vision/target_point`，`geometry_msgs/msg/PointStamped`

## 两种测量模式

1. `depth`：读取 D435i 对齐到彩色图的深度图，使用相机内参反投影。
2. `lidar`：读取 `/livox/lidar`，使用真实的雷达到相机外参把点云投影到图像，选择目标像素附近的三维点。

`yolo_target_pixel_node` 可通过 OpenCV DNN 加载本项目导出的 YOLOv5 ONNX
权重，从实时彩色图像选择最高置信度缺陷并发布检测框中心像素，形成
`图像 → YOLO → /vision/target_pixel → 深度/点云 → /vision/target_point` 闭环。

LiDAR 模式默认由 `calibration_ready: false` 锁定。没有真实外参时节点拒绝发布，避免把单位矩阵产生的伪坐标作为实验结果。
节点还会拒绝时间差超过 `max_sensor_skew_sec` 的检测与测量数据，默认门限为
0.1 秒，避免无人机或目标运动时发生跨帧错配。

## 相机—雷达标定 TF

`calibration_node` 负责加载并发布**已经求解和验证**的外参。它不会把安装尺寸
估计值伪装成实测结果，也不在运行时自动执行 PnP/ICP。实物模式要求：

- 标定文件状态为 `calibrated`；
- 同时提供 `camera_in_base` 和 `lidar_in_base`；
- 相机帧必须使用光学坐标约定（+Z 前、+X 右、+Y 下）；
- 记录有限且不超过默认 3 px 阈值的重投影误差。

TF 拓扑只能选择一种：`base_tree` 发布 `base→camera` 与 `base→lidar`，或
`direct_extrinsic` 只发布 `camera→lidar`。节点不会同时发布三条边形成闭环。

## 构建

```bash
mkdir -p ~/ros2_ws/src
cp -r /mnt/c/Users/Jokei/Desktop/挑战杯/HaiYing-System-V1/vision/haiying_vision_3d ~/ros2_ws/src/
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select haiying_vision_3d --symlink-install
source install/setup.bash
```

## 运行与验收

先在 `config/target_point.yaml` 中选择模式。LiDAR 模式必须填入标定得到的 `lidar_to_camera` 4×4 外参矩阵并将 `calibration_ready` 改为 `true`。

```bash
ros2 launch haiying_vision_3d target_point.launch.py
```

包含 YOLO 检测的完整管线：

```bash
export HAIYING_YOLO_ONNX=/mnt/c/Users/Jokei/Desktop/挑战杯/yolov5/output/best.onnx
ros2 launch haiying_vision_3d full_pipeline.launch.py
```

使用真实标定 TF 的 LiDAR 完整管线：

```bash
cp /mnt/c/Users/Jokei/Desktop/挑战杯/雷达/交付材料/radar_3d/calibration.template.yaml \
  /mnt/c/Users/Jokei/Desktop/挑战杯/雷达/交付材料/radar_3d/calibration.yaml
# 填入实测值、验证误差，并把 status 改为 calibrated
export HAIYING_CALIBRATION=/mnt/c/Users/Jokei/Desktop/挑战杯/雷达/交付材料/radar_3d/calibration.yaml
export HAIYING_YOLO_ONNX=/mnt/c/Users/Jokei/Desktop/挑战杯/yolov5/output/best.onnx
ros2 launch haiying_vision_3d calibrated_lidar_pipeline.launch.py
```

另开终端检查接口：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic info /vision/target_point
ros2 topic echo /vision/target_point
```

只有同时满足以下条件才可作为真实验收结果：

- `/vision/target_pixel` 来自实际图像检测结果；
- 深度图或 MID-360 点云处于实时发布状态；
- LiDAR 模式使用现场标定外参；
- 输出坐标有限、量纲为米，且 `frame_id` 与实际坐标系一致；
- 需要绝对坐标时配置 `target_frame: map`，并保证对应 TF 链存在。
