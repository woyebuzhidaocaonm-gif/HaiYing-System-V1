# 视觉感知模块

本目录集中存放海鹰智巡项目的YOLO环境检查、数据处理、训练、评估和推理脚本。
模型权重、数据集、ROS bag和训练输出均不进入Git仓库。

## 目录

```text
scripts/
├── environment/            # Python、PyTorch、CUDA与Ultralytics环境检查
├── data_tools/             # 标注转换、数据审计、清洗、合并和划分
├── training/
│   ├── yolo11_search/      # YOLO11参数搜索、正式训练和推理
│   ├── yolo11_baseline/    # 合并六类数据集YOLO11基线
│   ├── yolov5_baseline/    # 合并六类数据集YOLOv5基线
│   └── yolo26n/            # WTBs2025九类YOLO26n实验
└── reports/                # 小体积实验报告，不含模型权重
```

## 环境

建议使用Python虚拟环境。安装基础依赖：

```bash
python -m pip install -r scripts/requirements.txt
python scripts/environment/check_gpu.py
```

PyTorch及CUDA版本应根据开发机和显卡单独安装。

所有脚本通过`HAIYING_DATA_ROOT`定位本地数据目录。例如PowerShell：

```powershell
$env:HAIYING_DATA_ROOT = "D:\HaiYingData"
```

该目录可以包含：

```text
HaiYingData/
├── WT blade defect dataset/
├── WTBs2025/
├── Blade30_yolo_staging/
└── WT_blade_merged_dataset/
```

训练输出默认写入当前工作目录的`runs/`，也可以通过`HAIYING_RUN_ROOT`修改。

## ROS 2接口约束

当前导入内容是离线数据处理、训练和推理脚本，尚未包含ROS 2在线发布节点。
后续视觉节点发布目标坐标时必须遵守`docs/ROS2_Interface_V1.md`：

- Topic：`/vision/target_point`
- 类型：`geometry_msgs/msg/PointStamped`

严禁在视觉脚本中私自更换Topic或消息类型。

## 禁止提交

- `*.pt`、`*.onnx`、`*.weights`等模型文件；
- 数据集、图片批次和压缩包；
- `runs/`、`weights/`、`output/`等训练产物；
- ROS bag、Python缓存和第三方YOLO源码。

YOLOv5基线要求另行克隆官方YOLOv5仓库，并通过`YOLOV5_ROOT`指向其路径。
