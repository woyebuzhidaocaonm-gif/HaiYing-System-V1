import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd())) / "WT blade defect dataset"
CLASS_FILE = ROOT / "class_definitions.txt"
JOBS = (
    (ROOT / "Annotations", ROOT / "label1"),
    (ROOT / "annotation_second_person", ROOT / "label2"),
)


def load_classes():
    classes = [
        line.strip()
        for line in CLASS_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"No classes found in {CLASS_FILE}")
    if len(classes) != len(set(classes)):
        raise ValueError("Duplicate class names found")
    return classes


def convert_xml(xml_path, class_to_id):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size: {xml_path}")

    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    lines = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in class_to_id:
            raise ValueError(f"Unknown class '{class_name}' in {xml_path}")

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {xml_path}")

        xmin = float(box.findtext("xmin", "nan"))
        ymin = float(box.findtext("ymin", "nan"))
        xmax = float(box.findtext("xmax", "nan"))
        ymax = float(box.findtext("ymax", "nan"))

        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(
                f"Invalid box in {xml_path}: "
                f"({xmin}, {ymin}, {xmax}, {ymax}) for {width}x{height}"
            )

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(
            f"{class_to_id[class_name]} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the two VOC annotation sets to YOLO labels.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=ROOT,
        help="Root containing Annotations, annotation_second_person and class_definitions.txt.",
    )
    return parser.parse_args()


def main():
    global ROOT, CLASS_FILE, JOBS
    ROOT = parse_args().dataset_root.expanduser().resolve()
    CLASS_FILE = ROOT / "class_definitions.txt"
    JOBS = (
        (ROOT / "Annotations", ROOT / "label1"),
        (ROOT / "annotation_second_person", ROOT / "label2"),
    )
    classes = load_classes()
    class_to_id = {name: index for index, name in enumerate(classes)}

    converted = []
    for source_dir, output_dir in JOBS:
        xml_files = sorted(source_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {source_dir}")
        results = [(xml.stem, convert_xml(xml, class_to_id)) for xml in xml_files]
        converted.append((source_dir, output_dir, results))

    for source_dir, output_dir, results in converted:
        output_dir.mkdir(parents=True, exist_ok=True)
        for stem, lines in results:
            text = "\n".join(lines)
            if text:
                text += "\n"
            (output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        box_count = sum(len(lines) for _, lines in results)
        print(f"{source_dir.name} -> {output_dir.name}: {len(results)} files, {box_count} boxes")

    print("Class mapping:")
    for class_name, class_id in class_to_id.items():
        print(f"  {class_id}: {class_name}")


if __name__ == "__main__":
    main()
