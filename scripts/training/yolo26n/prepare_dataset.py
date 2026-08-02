from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


WORKSPACE = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
SOURCE = WORKSPACE / "WTBs2025"
OUTPUT = Path(
    os.environ.get(
        "HAIYING_YOLO26_DATASET",
        Path.cwd() / "runs" / "yolo26n" / "dataset",
    )
)
SEED = 2026
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
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


def group_key(path: Path) -> str:
    """Keep Roboflow variants of the same source image in one split."""
    return path.stem.split("_jpg.rf.", 1)[0]


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def clean_label(path: Path) -> tuple[list[str], int, Counter]:
    lines: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0
    class_counts: Counter = Counter()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        normalized = " ".join(raw.split())
        if not normalized:
            continue
        parts = normalized.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(parts)}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class id {class_id}")
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized box {values}")
        if normalized in seen:
            duplicate_count += 1
            continue
        seen.add(normalized)
        lines.append(normalized)
        class_counts[class_id] += 1
    return lines, duplicate_count, class_counts


def allocate_groups(groups: dict[str, list[Path]], rng: random.Random) -> dict[str, list[Path]]:
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    total = sum(len(paths) for _, paths in items)
    targets = {
        "val": round(total * SPLIT_RATIOS["val"]),
        "test": round(total * SPLIT_RATIOS["test"]),
    }
    allocated = {"train": [], "val": [], "test": []}

    for _, paths in items:
        candidates = [
            split
            for split in ("val", "test")
            if len(allocated[split]) < targets[split]
        ]
        if candidates:
            split = max(candidates, key=lambda name: targets[name] - len(allocated[name]))
        else:
            split = "train"
        allocated[split].extend(paths)
    return allocated


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)

    for split in SPLIT_RATIOS:
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    split_images: dict[str, list[Path]] = defaultdict(list)
    image_counts: dict[str, dict[str, int]] = {}
    box_counts: Counter = Counter()
    duplicates_removed = 0
    empty_labels = 0

    for class_id, class_name in enumerate(CLASS_NAMES):
        image_dir = SOURCE / class_name / "images"
        label_dir = SOURCE / class_name / "labels"
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        groups: dict[str, list[Path]] = defaultdict(list)
        for image in images:
            groups[group_key(image)].append(image)
        allocated = allocate_groups(groups, rng)
        image_counts[class_name] = {split: len(paths) for split, paths in allocated.items()}

        for split, paths in allocated.items():
            for image in paths:
                source_label = label_dir / f"{image.stem}.txt"
                if not source_label.is_file():
                    raise FileNotFoundError(f"Missing label for {image}")
                clean_lines, removed, classes = clean_label(source_label)
                duplicates_removed += removed
                box_counts.update(classes)
                if not clean_lines:
                    empty_labels += 1

                safe_name = f"{class_id}_{class_name.replace(' ', '_')}_{image.name}"
                destination_image = OUTPUT / "images" / split / safe_name
                destination_label = OUTPUT / "labels" / split / f"{Path(safe_name).stem}.txt"
                link_or_copy(image, destination_image)
                destination_label.write_text(
                    "\n".join(clean_lines) + ("\n" if clean_lines else ""),
                    encoding="utf-8",
                )
                split_images[split].append(destination_image)

    yaml_text = "\n".join(
        [
            f"path: {OUTPUT.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "images": {split: len(paths) for split, paths in split_images.items()},
        "images_by_source_class": image_counts,
        "unique_boxes_by_class": {
            CLASS_NAMES[class_id]: box_counts[class_id] for class_id in range(len(CLASS_NAMES))
        },
        "duplicate_annotation_rows_removed": duplicates_removed,
        "empty_labels": empty_labels,
    }
    (OUTPUT.parent / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
