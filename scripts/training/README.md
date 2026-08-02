# 模型训练与推理

推荐主流程为`yolo11_search/`：

```bash
python scripts/training/yolo11_search/prepare_dataset.py
python scripts/training/yolo11_search/train.py
python scripts/training/yolo11_search/predict.py /path/to/image.jpg
```

其他实验：

```bash
python scripts/training/yolo11_baseline/train.py
python scripts/training/yolov5_baseline/train.py
python scripts/training/yolo26n/prepare_dataset.py
python scripts/training/yolo26n/train.py
```

运行前设置：

- `HAIYING_DATA_ROOT`：本地数据所在目录；
- `HAIYING_RUN_ROOT`：可选，训练输出目录；
- `YOLOV5_ROOT`：仅YOLOv5基线需要；
- `HAIYING_YOLO11_WEIGHTS`、`HAIYING_YOLO26_WEIGHTS`：可选预训练权重名称或本地路径。

权重和训练输出受`.gitignore`保护，不得使用强制添加上传。
