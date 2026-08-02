from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


ROOT = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
BASE_ROOT = ROOT / "WT blade defect dataset"
WTBS_ROOT = ROOT / "WTBs2025"
BLADE30_ROOT = ROOT / "Blade30_yolo_staging"
OUTPUT_ROOT = ROOT / "WT_blade_merged_dataset"

BASE_CLASSES = [
    "craze",
    "corrosion",
    "surface_injure",
    "thunderstrike",
    "crack",
    "hide_craze",
]
BASE_CLASS_TO_ID = {name: index for index, name in enumerate(BASE_CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = "wt-blade-merged-v1"

# None means that the source category has no reliable equivalent in the base taxonomy.
WTBS_MAPPING = {
    0: None,  # oil leakage
    1: 0,     # paint cracks -> craze
    2: 2,     # localized damage -> surface_injure
    3: 3,     # lightning strikes -> thunderstrike
    4: None,  # surface stains
    5: 1,     # erosion -> corrosion (same policy as prepare_blade30.py)
    6: 2,     # coating detachment -> surface_injure
    7: 2,     # protective film damage -> surface_injure
    8: 2,     # pinholes -> surface_injure
}
WTBS_CLASSES = [
    "oil leakage",
    "paint cracks",
    "localized damage",
    "lightning strikes",
    "surface stains",
    "erosion",
    "coating detachment",
    "protective film damage",
    "pinholes",
]
WTBS_RATIONALES = {
    0: "Excluded: oil leakage is not one of the six authoritative defect classes.",
    1: "Mapped to craze: paint-layer cracking/crazing.",
    2: "Mapped to surface_injure: local surface damage.",
    3: "Mapped to thunderstrike: direct semantic equivalent.",
    4: "Excluded: stains/contamination are not physical surface injury in the base taxonomy.",
    5: "Mapped to corrosion: follows the existing project policy for leading-edge erosion.",
    6: "Mapped to surface_injure: coating loss is visible surface damage.",
    7: "Mapped to surface_injure: protective-film damage is surface-layer damage.",
    8: "Mapped to surface_injure: pinholes are localized surface defects.",
}
SOURCE_PRIORITY = {"base": 0, "blade30": 1, "wtbs2025": 2}


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float


@dataclass
class Sample:
    image_path: Path
    boxes: list[Box]
    source: str
    source_bucket: str
    group_key: str
    original_relative_path: str
    sha256: str = ""
    width: int = 0
    height: int = 0
    all_sources: set[str] = field(default_factory=set)
    all_paths: list[str] = field(default_factory=list)
    all_group_keys: set[str] = field(default_factory=set)


class Audit:
    def __init__(self):
        self.warnings: list[dict[str, str]] = []
        self.source_images_seen = Counter()
        self.source_images_with_mapped_objects = Counter()
        self.source_objects_seen = Counter()
        self.source_objects_included = Counter()
        self.source_objects_excluded = Counter()
        self.source_class_images_seen = Counter()
        self.source_class_images_included = Counter()
        self.source_class_objects_seen = Counter()
        self.source_class_objects_included = Counter()
        self.invalid_boxes = Counter()
        self.missing_labels = Counter()
        self.unreadable_images = Counter()

    def warn(self, source: str, path: Path, issue: str):
        self.warnings.append(
            {
                "source": source,
                "path": relative_to_root(path),
                "issue": issue,
            }
        )


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def find_image(folder: Path, stem: str) -> Path | None:
    for candidate in folder.glob(f"{stem}.*"):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def image_size(path: Path, audit: Audit, source: str) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid dimensions {width}x{height}")
            return width, height
    except Exception as exc:
        audit.unreadable_images[source] += 1
        audit.warn(source, path, f"Unreadable image: {exc}")
        return None


def normalize_xyxy(
    class_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: float,
    height: float,
) -> Box | None:
    if width <= 0 or height <= 0:
        return None
    xmin = max(0.0, min(float(width), xmin))
    xmax = max(0.0, min(float(width), xmax))
    ymin = max(0.0, min(float(height), ymin))
    ymax = max(0.0, min(float(height), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return Box(
        class_id,
        ((xmin + xmax) / 2.0) / width,
        ((ymin + ymax) / 2.0) / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def normalize_yolo(class_id: int, x: float, y: float, width: float, height: float) -> Box | None:
    return normalize_xyxy(
        class_id,
        x - width / 2.0,
        y - height / 2.0,
        x + width / 2.0,
        y + height / 2.0,
        1.0,
        1.0,
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base(audit: Audit) -> list[Sample]:
    samples = []
    annotations = BASE_ROOT / "Annotations"
    image_folder = BASE_ROOT / "JPEGImages"
    for xml_path in sorted(annotations.glob("*.xml")):
        audit.source_images_seen["base"] += 1
        image_path = find_image(image_folder, xml_path.stem)
        if image_path is None:
            audit.warn("base", xml_path, "No matching image in JPEGImages.")
            continue
        actual_size = image_size(image_path, audit, "base")
        if actual_size is None:
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:
            audit.warn("base", xml_path, f"Invalid XML: {exc}")
            continue
        xml_width = float(root.findtext("size/width") or actual_size[0])
        xml_height = float(root.findtext("size/height") or actual_size[1])
        if (round(xml_width), round(xml_height)) != actual_size:
            audit.warn(
                "base",
                xml_path,
                f"XML size {xml_width:g}x{xml_height:g} differs from image {actual_size[0]}x{actual_size[1]}; XML coordinates retained.",
            )
        boxes = []
        seen_classes = set()
        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            audit.source_objects_seen["base"] += 1
            audit.source_class_objects_seen[("base", class_name)] += 1
            seen_classes.add(class_name)
            if class_name not in BASE_CLASS_TO_ID:
                audit.source_objects_excluded["base"] += 1
                audit.warn("base", xml_path, f"Unknown base class {class_name!r}; object skipped.")
                continue
            bnd = obj.find("bndbox")
            try:
                coords = [float(bnd.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
            except Exception:
                audit.invalid_boxes["base"] += 1
                audit.warn("base", xml_path, f"Invalid bounding box for {class_name!r}.")
                continue
            box = normalize_xyxy(
                BASE_CLASS_TO_ID[class_name],
                *coords,
                xml_width,
                xml_height,
            )
            if box is None:
                audit.invalid_boxes["base"] += 1
                audit.warn("base", xml_path, f"Degenerate bounding box for {class_name!r}.")
                continue
            boxes.append(box)
            audit.source_objects_included["base"] += 1
            audit.source_class_objects_included[("base", class_name)] += 1
        for class_name in seen_classes:
            audit.source_class_images_seen[("base", class_name)] += 1
        if not boxes:
            continue
        for class_id in {box.class_id for box in boxes}:
            audit.source_class_images_included[("base", BASE_CLASSES[class_id])] += 1
        audit.source_images_with_mapped_objects["base"] += 1
        samples.append(
            Sample(
                image_path=image_path,
                boxes=boxes,
                source="base",
                source_bucket="base",
                group_key=f"base:{xml_path.stem.lower()}",
                original_relative_path=relative_to_root(image_path),
                width=actual_size[0],
                height=actual_size[1],
            )
        )
    return samples


def normalized_wtbs_stem(stem: str) -> str:
    stem = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    return stem.lower()


def parse_yolo_file(
    label_path: Path,
    source: str,
    mapping: dict[int, int | None],
    source_class_names: list[str],
    audit: Audit,
) -> tuple[list[Box], set[str], set[str]]:
    boxes = []
    source_classes_seen = set()
    mapped_classes_seen = set()
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        audit.warn(source, label_path, f"Could not read label: {exc}")
        return boxes, source_classes_seen, mapped_classes_seen
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            source_id = int(fields[0])
            x, y, width, height = map(float, fields[1:5])
        except Exception:
            audit.invalid_boxes[source] += 1
            audit.warn(source, label_path, f"Malformed YOLO line {line_number}: {line!r}")
            continue
        source_name = (
            source_class_names[source_id]
            if 0 <= source_id < len(source_class_names)
            else f"unknown_id_{source_id}"
        )
        source_classes_seen.add(source_name)
        audit.source_objects_seen[source] += 1
        audit.source_class_objects_seen[(source, source_name)] += 1
        target_id = mapping.get(source_id)
        if target_id is None:
            audit.source_objects_excluded[source] += 1
            continue
        box = normalize_yolo(target_id, x, y, width, height)
        if box is None:
            audit.invalid_boxes[source] += 1
            audit.warn(source, label_path, f"Invalid or degenerate YOLO box on line {line_number}.")
            continue
        boxes.append(box)
        mapped_classes_seen.add(BASE_CLASSES[target_id])
        audit.source_objects_included[source] += 1
        audit.source_class_objects_included[(source, source_name)] += 1
    return boxes, source_classes_seen, mapped_classes_seen


def load_wtbs(audit: Audit) -> list[Sample]:
    samples = []
    for category_dir in sorted(path for path in WTBS_ROOT.iterdir() if path.is_dir()):
        image_dir = category_dir / "images"
        label_dir = category_dir / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            audit.source_images_seen["wtbs2025"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                audit.missing_labels["wtbs2025"] += 1
                audit.warn("wtbs2025", image_path, "Missing YOLO label file.")
                continue
            actual_size = image_size(image_path, audit, "wtbs2025")
            if actual_size is None:
                continue
            boxes, source_classes, mapped_classes = parse_yolo_file(
                label_path,
                "wtbs2025",
                WTBS_MAPPING,
                WTBS_CLASSES,
                audit,
            )
            for source_class in source_classes:
                audit.source_class_images_seen[("wtbs2025", source_class)] += 1
            if not boxes:
                continue
            for target_class in mapped_classes:
                # Report included source categories separately below from object counts.
                audit.source_class_images_included[("wtbs2025_target", target_class)] += 1
            for source_class in source_classes:
                if any(
                    WTBS_MAPPING.get(source_id) is not None
                    and WTBS_CLASSES[source_id] == source_class
                    for source_id in range(len(WTBS_CLASSES))
                ):
                    audit.source_class_images_included[("wtbs2025", source_class)] += 1
            audit.source_images_with_mapped_objects["wtbs2025"] += 1
            samples.append(
                Sample(
                    image_path=image_path,
                    boxes=boxes,
                    source="wtbs2025",
                    source_bucket=category_dir.name,
                    group_key=f"wtbs2025:{normalized_wtbs_stem(image_path.stem)}",
                    original_relative_path=relative_to_root(image_path),
                    width=actual_size[0],
                    height=actual_size[1],
                )
            )
    return samples


def load_blade30(audit: Audit) -> list[Sample]:
    samples = []
    image_dir = BLADE30_ROOT / "images"
    label_dir = BLADE30_ROOT / "labels"
    identity_mapping = {index: index for index in range(len(BASE_CLASSES))}
    for image_path in sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ):
        audit.source_images_seen["blade30"] += 1
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            audit.missing_labels["blade30"] += 1
            audit.warn("blade30", image_path, "Missing YOLO label file.")
            continue
        actual_size = image_size(image_path, audit, "blade30")
        if actual_size is None:
            continue
        boxes, source_classes, mapped_classes = parse_yolo_file(
            label_path,
            "blade30",
            identity_mapping,
            BASE_CLASSES,
            audit,
        )
        for source_class in source_classes:
            audit.source_class_images_seen[("blade30", source_class)] += 1
            audit.source_class_images_included[("blade30", source_class)] += 1
        if not boxes:
            continue
        audit.source_images_with_mapped_objects["blade30"] += 1
        samples.append(
            Sample(
                image_path=image_path,
                boxes=boxes,
                source="blade30",
                source_bucket="blade30",
                group_key=f"blade30:{image_path.stem.lower()}",
                original_relative_path=relative_to_root(image_path),
                width=actual_size[0],
                height=actual_size[1],
            )
        )
    return samples


def box_iou(first: Box, second: Box) -> float:
    def corners(box: Box):
        return (
            box.x - box.width / 2.0,
            box.y - box.height / 2.0,
            box.x + box.width / 2.0,
            box.y + box.height / 2.0,
        )

    ax1, ay1, ax2, ay2 = corners(first)
    bx1, by1, bx2, by2 = corners(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_boxes(boxes: list[Box], threshold: float = 0.9) -> tuple[list[Box], int]:
    kept = []
    removed = 0
    for box in sorted(boxes, key=lambda item: (item.class_id, item.x, item.y, item.width, item.height)):
        if any(box.class_id == other.class_id and box_iou(box, other) >= threshold for other in kept):
            removed += 1
        else:
            kept.append(box)
    return kept, removed


def deduplicate_samples(samples: list[Sample]) -> tuple[list[Sample], int, int]:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        sample.sha256 = hash_file(sample.image_path)
        sample.all_sources = {sample.source}
        sample.all_paths = [sample.original_relative_path]
        sample.all_group_keys = {sample.group_key}
        by_hash[sample.sha256].append(sample)

    deduplicated = []
    duplicate_images_removed = 0
    duplicate_boxes_removed = 0
    for digest, group in by_hash.items():
        group.sort(
            key=lambda sample: (
                SOURCE_PRIORITY[sample.source],
                -(sample.width * sample.height),
                sample.original_relative_path,
            )
        )
        representative = group[0]
        combined_boxes = []
        for sample in group:
            combined_boxes.extend(sample.boxes)
            representative.all_sources.add(sample.source)
            representative.all_paths.extend(sample.all_paths)
            representative.all_group_keys.add(sample.group_key)
        representative.boxes, removed = deduplicate_boxes(combined_boxes)
        duplicate_boxes_removed += removed
        duplicate_images_removed += len(group) - 1
        representative.sha256 = digest
        representative.all_paths = sorted(set(representative.all_paths))
        deduplicated.append(representative)
    deduplicated.sort(key=lambda sample: sample.sha256)
    return deduplicated, duplicate_images_removed, duplicate_boxes_removed


def dominant_class(sample: Sample) -> int:
    counts = Counter(box.class_id for box in sample.boxes)
    return min(
        (class_id for class_id, count in counts.items() if count == max(counts.values())),
        default=0,
    )


def split_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
    # Keep Roboflow derivatives from the same original stem in one split.
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_key].append(sample)

    strata: dict[tuple[str, int], list[tuple[str, list[Sample]]]] = defaultdict(list)
    for group_key, members in groups.items():
        source = min((member.source for member in members), key=SOURCE_PRIORITY.get)
        class_counts = Counter(box.class_id for member in members for box in member.boxes)
        max_count = max(class_counts.values())
        dominant = min(class_id for class_id, count in class_counts.items() if count == max_count)
        strata[(source, dominant)].append((group_key, members))

    output = {split: [] for split in SPLITS}
    for stratum_key, stratum_groups in sorted(strata.items()):
        ordered = sorted(
            stratum_groups,
            key=lambda item: hashlib.sha256(
                f"{SPLIT_SEED}|{stratum_key}|{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        total_samples = sum(len(members) for _, members in ordered)
        if total_samples < 5:
            for _, members in ordered:
                output["train"].extend(members)
            continue

        targets = {split: total_samples * ratio for split, ratio in SPLIT_RATIOS.items()}
        current = Counter()
        for _, members in ordered:
            group_size = len(members)
            # Fill each target proportionally. A group is indivisible to prevent leakage.
            chosen_split = min(
                SPLITS,
                key=lambda split: (
                    current[split] / max(targets[split], 1e-9),
                    SPLITS.index(split),
                ),
            )
            output[chosen_split].extend(members)
            current[chosen_split] += group_size

    for split in SPLITS:
        output[split].sort(key=lambda sample: sample.sha256)
    return output


def output_name(sample: Sample) -> str:
    suffix = sample.image_path.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    return f"img_{sample.sha256[:20]}{suffix}"


def class_object_counts(samples: list[Sample]) -> Counter:
    return Counter(box.class_id for sample in samples for box in sample.boxes)


def class_image_counts(samples: list[Sample]) -> Counter:
    counts = Counter()
    for sample in samples:
        for class_id in {box.class_id for box in sample.boxes}:
            counts[class_id] += 1
    return counts


def write_mapping_report(path: Path, audit: Audit):
    rows = []
    for class_id, class_name in enumerate(BASE_CLASSES):
        rows.append(
            {
                "source_dataset": "WT blade defect dataset/Annotations",
                "source_class_id": class_id,
                "source_class_name": class_name,
                "action": "keep",
                "target_class_id": class_id,
                "target_class_name": class_name,
                "source_objects_seen": audit.source_class_objects_seen[("base", class_name)],
                "objects_included": audit.source_class_objects_included[("base", class_name)],
                "source_images_seen": audit.source_class_images_seen[("base", class_name)],
                "images_with_included_objects": audit.source_class_images_included[("base", class_name)],
                "rationale": "Authoritative base taxonomy and primary annotation folder.",
            }
        )
    for source_id, source_name in enumerate(WTBS_CLASSES):
        target_id = WTBS_MAPPING[source_id]
        rows.append(
            {
                "source_dataset": "WTBs2025",
                "source_class_id": source_id,
                "source_class_name": source_name,
                "action": "exclude" if target_id is None else "map",
                "target_class_id": "" if target_id is None else target_id,
                "target_class_name": "" if target_id is None else BASE_CLASSES[target_id],
                "source_objects_seen": audit.source_class_objects_seen[("wtbs2025", source_name)],
                "objects_included": audit.source_class_objects_included[("wtbs2025", source_name)],
                "source_images_seen": audit.source_class_images_seen[("wtbs2025", source_name)],
                "images_with_included_objects": audit.source_class_images_included[("wtbs2025", source_name)],
                "rationale": WTBS_RATIONALES[source_id],
            }
        )
    for class_id, class_name in enumerate(BASE_CLASSES):
        rows.append(
            {
                "source_dataset": "Blade30_yolo_staging",
                "source_class_id": class_id,
                "source_class_name": class_name,
                "action": "keep",
                "target_class_id": class_id,
                "target_class_name": class_name,
                "source_objects_seen": audit.source_class_objects_seen[("blade30", class_name)],
                "objects_included": audit.source_class_objects_included[("blade30", class_name)],
                "source_images_seen": audit.source_class_images_seen[("blade30", class_name)],
                "images_with_included_objects": audit.source_class_images_included[("blade30", class_name)],
                "rationale": "Already converted to the authoritative six-class order and manually staged.",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path):
    path.write_text(
        """# 风机叶片缺陷合并数据集

本数据集以 `WT blade defect dataset` 的六类定义为唯一类别标准：

| ID | 类别 |
|---:|---|
| 0 | craze |
| 1 | corrosion |
| 2 | surface_injure |
| 3 | thunderstrike |
| 4 | crack |
| 5 | hide_craze |

## 来源与处理

- 基准集采用 `Annotations`（主标注）+ `JPEGImages`。`annotation_second_person` 是同一批图片的第二人标注，没有作为第二份图片重复导入。
- `WTBs2025` 按 `mapping_report.csv` 映射。`oil leakage` 和 `surface stains` 没有可靠的基准类别，相关对象被剔除；若同一图片还含可映射对象，则保留图片及可映射对象。
- `Blade30_yolo_staging` 沿用项目中已人工限定的映射：前缘侵蚀→`corrosion`，浅表尾缘裂纹→`crack`。
- `yolo_pycharm/dataset_1000` 的 1000 张图片均与基准集同名图片逐字节相同，因此判定为训练用派生子集，没有重复导入。
- `yolov5`、`yolo_pycharm/output` 和 `runs` 是代码、权重及训练产物，不是额外原始数据源。
- 完全相同的图片按 SHA-256 去重，同图标注会合并；同类别且 IoU≥0.9 的重复框只保留一个。
- Roboflow 文件名中的 `.rf.<hash>` 增强版本按原始文件名前缀分组，同组不会跨越 train/val/test，降低数据泄漏风险。

## 目录

```text
images/{train,val,test}
labels/{train,val,test}
data.yaml
classes.txt
manifest.csv
mapping_report.csv
audit_warnings.csv
summary.json
```

YOLO 标注格式为 `class_id x_center y_center width height`，坐标均已归一化。原始数据集未被修改。
""",
        encoding="utf-8",
    )


def write_dataset(
    build_root: Path,
    split_map: dict[str, list[Sample]],
    audit: Audit,
    duplicate_images_removed: int,
    duplicate_boxes_removed: int,
):
    for split in SPLITS:
        (build_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (build_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    all_exported = []
    for split in SPLITS:
        for sample in split_map[split]:
            name = output_name(sample)
            destination = build_root / "images" / split / name
            shutil.copy2(sample.image_path, destination)
            label_path = build_root / "labels" / split / f"{Path(name).stem}.txt"
            label_path.write_text(
                "".join(
                    f"{box.class_id} {box.x:.6f} {box.y:.6f} {box.width:.6f} {box.height:.6f}\n"
                    for box in sample.boxes
                ),
                encoding="utf-8",
            )
            counts = Counter(box.class_id for box in sample.boxes)
            manifest_rows.append(
                {
                    "split": split,
                    "output_image": f"images/{split}/{name}",
                    "output_label": f"labels/{split}/{Path(name).stem}.txt",
                    "sha256": sample.sha256,
                    "width": sample.width,
                    "height": sample.height,
                    "selected_source": sample.source,
                    "all_sources": "|".join(sorted(sample.all_sources)),
                    "source_bucket": sample.source_bucket,
                    "group_key": sample.group_key,
                    "original_paths": "|".join(sample.all_paths),
                    "object_count": len(sample.boxes),
                    "class_counts": "|".join(
                        f"{BASE_CLASSES[class_id]}:{counts[class_id]}" for class_id in sorted(counts)
                    ),
                }
            )
            all_exported.append(sample)

    with (build_root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (build_root / "audit_warnings.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["source", "path", "issue"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit.warnings)

    write_mapping_report(build_root / "mapping_report.csv", audit)
    write_readme(build_root / "README.md")
    (build_root / "classes.txt").write_text("\n".join(BASE_CLASSES) + "\n", encoding="utf-8")
    absolute_root = OUTPUT_ROOT.as_posix()
    # Keep the YAML itself ASCII-only because this workspace's legacy YOLOv5
    # opens YAML with the Windows locale encoding and errors="ignore".
    yaml_root = json.dumps(absolute_root, ensure_ascii=True)
    (build_root / "data.yaml").write_text(
        f"""path: {yaml_root}
train: images/train
val: images/val
test: images/test

nc: {len(BASE_CLASSES)}
names:
"""
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(BASE_CLASSES)),
        encoding="utf-8",
    )

    split_summary = {}
    for split in SPLITS:
        object_counts = class_object_counts(split_map[split])
        image_counts = class_image_counts(split_map[split])
        split_summary[split] = {
            "images": len(split_map[split]),
            "objects": sum(object_counts.values()),
            "objects_by_class": {
                BASE_CLASSES[class_id]: object_counts[class_id]
                for class_id in range(len(BASE_CLASSES))
            },
            "images_by_class": {
                BASE_CLASSES[class_id]: image_counts[class_id]
                for class_id in range(len(BASE_CLASSES))
            },
        }
    total_object_counts = class_object_counts(all_exported)
    total_image_counts = class_image_counts(all_exported)
    selected_source_counts = Counter(sample.source for sample in all_exported)
    contributed_source_counts = Counter(
        source for sample in all_exported for source in sample.all_sources
    )
    summary = {
        "authoritative_classes": {
            index: name for index, name in enumerate(BASE_CLASSES)
        },
        "output_root": absolute_root,
        "source_images_seen": dict(audit.source_images_seen),
        "source_images_with_mapped_objects_before_dedup": dict(
            audit.source_images_with_mapped_objects
        ),
        "source_objects_seen": dict(audit.source_objects_seen),
        "source_objects_included_before_dedup": dict(audit.source_objects_included),
        "source_objects_excluded_by_mapping": dict(audit.source_objects_excluded),
        "deduplication": {
            "exact_duplicate_images_removed": duplicate_images_removed,
            "overlapping_duplicate_boxes_removed": duplicate_boxes_removed,
        },
        "validation": {
            "invalid_boxes_skipped": dict(audit.invalid_boxes),
            "missing_label_files": dict(audit.missing_labels),
            "unreadable_images_skipped": dict(audit.unreadable_images),
            "warnings": len(audit.warnings),
        },
        "final": {
            "images": len(all_exported),
            "objects": sum(total_object_counts.values()),
            "selected_images_by_source": dict(selected_source_counts),
            "images_contributed_by_source": dict(contributed_source_counts),
            "objects_by_class": {
                BASE_CLASSES[class_id]: total_object_counts[class_id]
                for class_id in range(len(BASE_CLASSES))
            },
            "images_by_class": {
                BASE_CLASSES[class_id]: total_image_counts[class_id]
                for class_id in range(len(BASE_CLASSES))
            },
            "splits": split_summary,
        },
    }
    (build_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_output(build_root: Path):
    problems = []
    seen_hashes = set()
    split_group_keys = defaultdict(set)
    manifest_path = build_root / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_path = build_root / row["output_image"]
        label_path = build_root / row["output_label"]
        if not image_path.exists():
            problems.append(f"Missing output image: {image_path}")
        if not label_path.exists():
            problems.append(f"Missing output label: {label_path}")
        if row["sha256"] in seen_hashes:
            problems.append(f"Duplicate final SHA-256: {row['sha256']}")
        seen_hashes.add(row["sha256"])
        split_group_keys[row["group_key"]].add(row["split"])
        if label_path.exists():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    problems.append(f"Bad label field count: {label_path}:{line_number}")
                    continue
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
                if class_id not in range(len(BASE_CLASSES)):
                    problems.append(f"Bad class id: {label_path}:{line_number}")
                if any(value < 0 or value > 1 for value in coords):
                    problems.append(f"Coordinate outside [0,1]: {label_path}:{line_number}")
                if coords[2] <= 0 or coords[3] <= 0:
                    problems.append(f"Non-positive box size: {label_path}:{line_number}")
    leaking_groups = {
        group_key: sorted(splits)
        for group_key, splits in split_group_keys.items()
        if len(splits) > 1
    }
    if leaking_groups:
        problems.append(f"Group leakage across splits: {leaking_groups}")
    expected_images = len(rows)
    actual_images = sum(
        1
        for split in SPLITS
        for path in (build_root / "images" / split).iterdir()
        if path.is_file()
    )
    actual_labels = sum(
        1
        for split in SPLITS
        for path in (build_root / "labels" / split).glob("*.txt")
        if path.is_file()
    )
    if actual_images != expected_images or actual_labels != expected_images:
        problems.append(
            f"Pair count mismatch: manifest={expected_images}, images={actual_images}, labels={actual_labels}"
        )
    if problems:
        raise RuntimeError("Output validation failed:\n" + "\n".join(problems[:50]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the base, WTBs2025 and Blade30 datasets into one audited YOLO dataset."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT,
        help="Directory containing all three source dataset directories.",
    )
    parser.add_argument("--output", type=Path, help="Final merged dataset directory.")
    return parser.parse_args()


def main():
    global ROOT, BASE_ROOT, WTBS_ROOT, BLADE30_ROOT, OUTPUT_ROOT
    args = parse_args()
    ROOT = args.workspace.expanduser().resolve()
    BASE_ROOT = ROOT / "WT blade defect dataset"
    WTBS_ROOT = ROOT / "WTBs2025"
    BLADE30_ROOT = ROOT / "Blade30_yolo_staging"
    OUTPUT_ROOT = (
        args.output.expanduser().resolve()
        if args.output
        else ROOT / "WT_blade_merged_dataset"
    )

    required = (BASE_ROOT, WTBS_ROOT, BLADE30_ROOT)
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required dataset directories: " + ", ".join(missing))

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_ROOT}. "
            "Move or rename it before rerunning."
        )
    audit = Audit()
    print("Loading authoritative base dataset...")
    samples = load_base(audit)
    print(f"  base samples: {len(samples)}")
    print("Loading WTBs2025 and applying class mapping...")
    wtbs_samples = load_wtbs(audit)
    samples.extend(wtbs_samples)
    print(f"  mapped WTBs2025 samples: {len(wtbs_samples)}")
    print("Loading Blade30 staging set...")
    blade30_samples = load_blade30(audit)
    samples.extend(blade30_samples)
    print(f"  Blade30 samples: {len(blade30_samples)}")
    print(f"Hashing and deduplicating {len(samples)} candidate images...")
    samples, duplicate_images_removed, duplicate_boxes_removed = deduplicate_samples(samples)
    print(
        f"  final unique candidates: {len(samples)} "
        f"(removed {duplicate_images_removed} exact duplicate images and "
        f"{duplicate_boxes_removed} overlapping boxes)"
    )
    split_map = split_samples(samples)
    print(
        "Split sizes: "
        + ", ".join(f"{split}={len(split_map[split])}" for split in SPLITS)
    )
    build_root = Path(tempfile.mkdtemp(prefix="WT_blade_merged_build_", dir=ROOT))
    try:
        print(f"Writing dataset to temporary build directory: {build_root.name}")
        write_dataset(
            build_root,
            split_map,
            audit,
            duplicate_images_removed,
            duplicate_boxes_removed,
        )
        print("Validating image/label pairs, class IDs, coordinates, hashes, and split groups...")
        validate_output(build_root)
        build_root.rename(OUTPUT_ROOT)
    except Exception:
        print(f"Build failed; partial files retained for inspection at: {build_root}")
        raise
    print(f"Done: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
