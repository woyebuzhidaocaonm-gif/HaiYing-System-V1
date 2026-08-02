# 数据处理工具

这些脚本只读取本地数据，并把派生数据写到`HAIYING_DATA_ROOT`或命令行指定目录。

```bash
# VOC XML转YOLO标签
python scripts/data_tools/convert_voc_to_yolo.py "/path/to/WT blade defect dataset"

# 生成两套数据源的标注抽查图
python scripts/data_tools/audit_preview.py --workspace "/path/to/HaiYingData"

# 将Blade30 LabelMe数据转换成待审查YOLO数据
python scripts/data_tools/prepare_blade30.py --source "/path/to/Blade30/source" --output "/path/to/Blade30_yolo_staging"

# 合并三套数据并执行去重、划分和完整性检查
python scripts/data_tools/merge_datasets.py --workspace "/path/to/HaiYingData"
```

`merge_datasets.py`不会覆盖已经存在的`WT_blade_merged_dataset`。需要重新生成时，请先人工检查并移动旧输出。
