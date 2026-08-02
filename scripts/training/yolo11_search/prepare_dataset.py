"""Validate and stratify exactly 1000 images into YOLO train/val/test sets."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from config import (
    CLASS_FILE,
    DATASET_DIR,
    EXPECTED_CLASSES,
    SEED,
    SOURCE_IMAGES,
    SOURCE_LABELS,
    SPLIT_SIZES,
    TOTAL_IMAGES,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    classes: frozenset[int]
    box_count: int


def load_classes() -> list[str]:
    names = [
        line.strip()
        for line in CLASS_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(names) != EXPECTED_CLASSES or len(set(names)) != len(names):
        raise ValueError(f"Expected {EXPECTED_CLASSES} unique classes, got: {names}")
    return names


def validate_label(path: Path, class_count: int) -> tuple[frozenset[int], int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty annotation: {path}")
    present: set[int] = set()
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class id {class_id}")
        cx, cy, width, height = values
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized box {values}")
        if cx - width / 2 < -1e-5 or cx + width / 2 > 1.00001:
            raise ValueError(f"{path}:{line_number}: box exceeds horizontal image bounds")
        if cy - height / 2 < -1e-5 or cy + height / 2 > 1.00001:
            raise ValueError(f"{path}:{line_number}: box exceeds vertical image bounds")
        present.add(class_id)
    return frozenset(present), len(lines)


def discover_samples(class_count: int) -> list[Sample]:
    if not SOURCE_IMAGES.is_dir() or not SOURCE_LABELS.is_dir():
        raise FileNotFoundError(f"Source dataset not found under {SOURCE_IMAGES.parent}")
    images = sorted(
        (path for path in SOURCE_IMAGES.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (int(path.stem) if path.stem.isdigit() else math.inf, path.name),
    )
    samples: list[Sample] = []
    for image in images:
        label = SOURCE_LABELS / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image.name}: {label}")
        classes, box_count = validate_label(label, class_count)
        samples.append(Sample(image, label, classes, box_count))
    if len(samples) < TOTAL_IMAGES:
        raise ValueError(f"Need at least {TOTAL_IMAGES} valid images, found {len(samples)}")
    return samples


def stratified_assignment(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Greedy multilabel stratification with exact split sizes.

    The unused group makes the 1000-image selection part of the same
    stratification problem instead of dropping 65 files arbitrarily.
    """
    rng = random.Random(SEED)
    split_limits = dict(SPLIT_SIZES)
    split_limits["unused"] = len(samples) - TOTAL_IMAGES
    total = len(samples)
    total_class_images = Counter(class_id for sample in samples for class_id in sample.classes)
    desired = {
        split: {
            class_id: total_class_images[class_id] * limit / total
            for class_id in range(EXPECTED_CLASSES)
        }
        for split, limit in split_limits.items()
    }
    current = {split: Counter() for split in split_limits}
    result: dict[str, list[Sample]] = {split: [] for split in split_limits}

    shuffled = list(samples)
    rng.shuffle(shuffled)
    # Multiclass and rare-class samples are allocated first.
    shuffled.sort(
        key=lambda sample: (
            sum(1 / total_class_images[class_id] for class_id in sample.classes),
            len(sample.classes),
        ),
        reverse=True,
    )

    for sample in shuffled:
        candidates = [split for split, limit in split_limits.items() if len(result[split]) < limit]
        best_score = -float("inf")
        best_splits: list[str] = []
        for split in candidates:
            class_deficit = sum(
                (desired[split][class_id] - current[split][class_id])
                / max(desired[split][class_id], 1.0)
                for class_id in sample.classes
            )
            size_deficit = (split_limits[split] - len(result[split])) / max(split_limits[split], 1)
            score = class_deficit + 0.20 * size_deficit
            if score > best_score + 1e-12:
                best_score, best_splits = score, [split]
            elif abs(score - best_score) <= 1e-12:
                best_splits.append(split)
        chosen = rng.choice(best_splits)
        result[chosen].append(sample)
        current[chosen].update(sample.classes)

    for split, limit in split_limits.items():
        if len(result[split]) != limit:
            raise AssertionError(f"{split}: expected {limit}, got {len(result[split])}")
    return result


def write_dataset(assignments: dict[str, list[Sample]], class_names: list[str]) -> None:
    # This directory is generated solely by this script; rebuilding it is safe.
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in SPLIT_SIZES:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seed": SEED,
        "total_selected": TOTAL_IMAGES,
        "source_label_set": str(SOURCE_LABELS),
        "class_names": class_names,
        "splits": {},
    }
    for split in SPLIT_SIZES:
        samples = sorted(assignments[split], key=lambda sample: sample.image.name)
        class_image_counts: Counter[int] = Counter()
        class_box_counts: Counter[int] = Counter()
        for sample in samples:
            shutil.copy2(sample.image, DATASET_DIR / "images" / split / sample.image.name)
            shutil.copy2(sample.label, DATASET_DIR / "labels" / split / f"{sample.image.stem}.txt")
            class_image_counts.update(sample.classes)
            for line in sample.label.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    class_box_counts[int(line.split()[0])] += 1
            rows.append(
                {
                    "filename": sample.image.name,
                    "split": split,
                    "classes": " ".join(map(str, sorted(sample.classes))),
                    "boxes": sample.box_count,
                }
            )
        summary["splits"][split] = {
            "images": len(samples),
            "boxes": sum(sample.box_count for sample in samples),
            "images_by_class": {class_names[i]: class_image_counts[i] for i in range(len(class_names))},
            "boxes_by_class": {class_names[i]: class_box_counts[i] for i in range(len(class_names))},
        }

    yaml_lines = [
        f"path: {DATASET_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(class_names)],
        "",
    ]
    (DATASET_DIR / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    with (DATASET_DIR / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "split", "classes", "boxes"])
        writer.writeheader()
        writer.writerows(rows)
    (DATASET_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare() -> dict[str, object]:
    class_names = load_classes()
    samples = discover_samples(len(class_names))
    assignments = stratified_assignment(samples)
    write_dataset(assignments, class_names)
    return json.loads((DATASET_DIR / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    summary = prepare()
    print(f"Prepared {summary['total_selected']} images in {DATASET_DIR}")
    for split, values in summary["splits"].items():
        print(f"  {split}: {values['images']} images, {values['boxes']} boxes")
        print(f"    images by class: {values['images_by_class']}")


if __name__ == "__main__":
    main()

