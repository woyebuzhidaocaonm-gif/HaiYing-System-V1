"""Convert the selected Blade30 LabelMe annotations into a reviewable YOLO staging set.

Only labels with an unambiguous mapping to the six classes of the local WT blade
defect dataset are exported.  The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


WORKSPACE_ROOT = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
SOURCE_ROOT = (
    WORKSPACE_ROOT
    / "3_blade_1_15_with_labeldata"
    / "3_blade_1_15_with_labeldata"
)
OUTPUT_ROOT = WORKSPACE_ROOT / "Blade30_yolo_staging"

# Keep exactly the class order used by the existing main dataset.
CLASS_NAMES = [
    "craze",
    "corrosion",
    "surface_injure",
    "thunderstrike",
    "crack",
    "hide_craze",
]

# Source LabelMe label -> (YOLO class id, target class name).
LABEL_MAPPING = {
    "leading edge;erosion;coating or LEP only": (1, "corrosion"),
    "leading edge;erosion;continuous or deep": (1, "corrosion"),
    "leading edge;erosion;spotty or laminate": (1, "corrosion"),
    "leading edge;erosion;eroded tip": (1, "corrosion"),
    "trailing edge;crack;superficial": (4, "crack"),
}


def yolo_box(points: list[list[float]], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Convert a LabelMe polygon to a clipped, normalized YOLO bounding box."""
    if len(points) < 2 or width <= 0 or height <= 0:
        return None
    xs = [max(0.0, min(float(width), float(point[0]))) for point in points]
    ys = [max(0.0, min(float(height), float(point[1]))) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def output_stem(json_path: Path) -> str:
    """Avoid filename collisions between Blade_N / flight folders."""
    relative = json_path.relative_to(SOURCE_ROOT)
    return "__".join((*relative.parts[:-1], json_path.stem))


def create_preview(image_path: Path, label_path: Path, preview_path: Path) -> bool:
    """Draw a small set of YOLO boxes for visual review, when OpenCV is available."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    # cv2.imread/imwrite on Windows cannot reliably handle non-ASCII paths.
    raw_image = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
    if image is None:
        return False
    height, width = image.shape[:2]
    colors = {1: (0, 165, 255), 4: (0, 0, 255)}  # BGR: orange / red
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, bw, bh = map(float, line.split())
        class_id = int(class_id)
        x1 = max(0, int((cx - bw / 2) * width))
        y1 = max(0, int((cy - bh / 2) * height))
        x2 = min(width - 1, int((cx + bw / 2) * width))
        y2 = min(height - 1, int((cy + bh / 2) * height))
        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(3, width // 1200))
        cv2.putText(image, CLASS_NAMES[class_id], (x1, max(35, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    scale = min(1.0, 1600 / max(width, height))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    saved, encoded = cv2.imencode(".jpg", image)
    if not saved:
        return False
    encoded.tofile(str(preview_path))
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT, help="Blade30 LabelMe root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT, help="YOLO staging output.")
    return parser.parse_args()


def main() -> None:
    global SOURCE_ROOT, OUTPUT_ROOT
    args = parse_args()
    SOURCE_ROOT = args.source.expanduser().resolve()
    OUTPUT_ROOT = args.output.expanduser().resolve()

    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Blade30 source folder not found: {SOURCE_ROOT}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_ROOT}. Review it or remove it before running again."
        )

    images_dir = OUTPUT_ROOT / "images"
    labels_dir = OUTPUT_ROOT / "labels"
    previews_dir = OUTPUT_ROOT / "preview"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    previews_dir.mkdir()

    class_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    ignored_counter: Counter[str] = Counter()
    rows: list[dict[str, str | int]] = []
    exported_images = 0
    invalid_shapes = 0
    preview_budget = 8

    for json_path in sorted(SOURCE_ROOT.rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        image_name = data.get("imagePath", "")
        image_path = json_path.parent / image_name
        width = int(data.get("imageWidth", 0))
        height = int(data.get("imageHeight", 0))
        boxes: list[tuple[int, tuple[float, float, float, float], str]] = []

        for shape in data.get("shapes", []):
            source_label = str(shape.get("label", "")).strip()
            mapping = LABEL_MAPPING.get(source_label)
            if mapping is None:
                ignored_counter[source_label] += 1
                continue
            box = yolo_box(shape.get("points", []), width, height)
            if box is None:
                invalid_shapes += 1
                continue
            class_id, target_name = mapping
            boxes.append((class_id, box, source_label))
            class_counter[target_name] += 1
            source_counter[source_label] += 1

        if not boxes:
            continue
        if not image_path.is_file():
            raise FileNotFoundError(f"Paired image missing for {json_path}: {image_path}")

        stem = output_stem(json_path)
        destination_image = images_dir / f"{stem}{image_path.suffix.lower()}"
        destination_label = labels_dir / f"{stem}.txt"
        shutil.copy2(image_path, destination_image)
        destination_label.write_text(
            "".join(
                f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                for class_id, (cx, cy, bw, bh), _ in boxes
            ),
            encoding="utf-8",
        )
        exported_images += 1
        for class_id, _, source_label in boxes:
            rows.append(
                {
                    "output_image": destination_image.name,
                    "blade": json_path.relative_to(SOURCE_ROOT).parts[0],
                    "source_json": str(json_path.relative_to(SOURCE_ROOT)),
                    "source_label": source_label,
                    "target_class": CLASS_NAMES[class_id],
                }
            )
        if preview_budget:
            if create_preview(destination_image, destination_label, previews_dir / f"{stem}.jpg"):
                preview_budget -= 1

    (OUTPUT_ROOT / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    with (OUTPUT_ROOT / "mapping_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_image", "blade", "source_json", "source_label", "target_class"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": str(SOURCE_ROOT),
        "exported_images": exported_images,
        "exported_objects_by_target_class": dict(class_counter),
        "exported_objects_by_source_label": dict(source_counter),
        "ignored_objects_by_source_label": dict(ignored_counter),
        "invalid_mapped_shapes": invalid_shapes,
        "class_order": CLASS_NAMES,
        "mapping": {source: {"class_id": cid, "class_name": name} for source, (cid, name) in LABEL_MAPPING.items()},
        "note": "This is a staging set. Inspect preview images before merging into the main dataset.",
    }
    (OUTPUT_ROOT / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "README.txt").write_text(
        "Blade30 YOLO staging set\n\n"
        "Only unambiguous mappings were exported:\n"
        "- leading-edge erosion -> corrosion (class 1)\n"
        "- superficial trailing-edge crack -> crack (class 4)\n\n"
        "Excluded: contamination, markings, OK add-ons, and LPS worn/burnt.\n"
        "Inspect the preview folder and conversion_summary.json before merging with the main dataset.\n",
        encoding="utf-8",
    )
    print(f"Exported {exported_images} images and {sum(class_counter.values())} objects to {OUTPUT_ROOT}")
    print("Objects by class:", dict(class_counter))
    print(f"Preview images created: {8 - preview_budget}")


if __name__ == "__main__":
    main()
