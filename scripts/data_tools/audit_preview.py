import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
BASE = ROOT / "WT blade defect dataset"
WTBS = ROOT / "WTBs2025"
OUTPUT = ROOT / "dataset_audit_preview.jpg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BASE_CLASSES = [
    "craze",
    "corrosion",
    "surface_injure",
    "thunderstrike",
    "crack",
    "hide_craze",
]
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
COLORS = [
    "#ff3b30",
    "#34c759",
    "#007aff",
    "#ff9500",
    "#af52de",
    "#00c7be",
    "#ff2d55",
    "#5856d6",
    "#a2845e",
]


def font(size: int):
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


TITLE_FONT = font(26)
LABEL_FONT = font(18)


def find_image(folder: Path, stem: str):
    for suffix in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = [p for p in folder.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return matches[0] if matches else None


def draw_box(draw, box, name, color, source_size, target_size):
    sx = target_size[0] / source_size[0]
    sy = target_size[1] / source_size[1]
    x1, y1, x2, y2 = box
    scaled = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    draw.rectangle(scaled, outline=color, width=3)
    draw.rectangle(
        (scaled[0], max(0, scaled[1] - 22), scaled[0] + 9 * len(name) + 8, scaled[1]),
        fill=color,
    )
    draw.text((scaled[0] + 3, max(0, scaled[1] - 21)), name, fill="white", font=LABEL_FONT)


def render_base_sample(xml_path: Path, wanted_class: str, size=(320, 240)):
    root = ET.parse(xml_path).getroot()
    image_path = find_image(BASE / "JPEGImages", xml_path.stem)
    if image_path is None:
        return None
    image = Image.open(image_path).convert("RGB")
    source_size = image.size
    image = image.resize(size)
    draw = ImageDraw.Draw(image)
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        box = tuple(float(bnd.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
        color = COLORS[BASE_CLASSES.index(name)] if name in BASE_CLASSES else "white"
        draw_box(draw, box, name, color, source_size, size)
    return image


def base_samples(class_name: str, count=3):
    candidates = []
    for xml_path in sorted((BASE / "Annotations").glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        if any((obj.findtext("name") or "").strip() == class_name for obj in root.findall("object")):
            candidates.append(xml_path)
    if not candidates:
        return []
    indexes = sorted(set(round(i * (len(candidates) - 1) / max(1, count - 1)) for i in range(count)))
    return [render_base_sample(candidates[i], class_name) for i in indexes]


def render_wtbs_sample(image_path: Path, size=(320, 240)):
    image = Image.open(image_path).convert("RGB")
    source_size = image.size
    image = image.resize(size)
    draw = ImageDraw.Draw(image)
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:5])
            box = (
                (cx - width / 2) * source_size[0],
                (cy - height / 2) * source_size[1],
                (cx + width / 2) * source_size[0],
                (cy + height / 2) * source_size[1],
            )
            name = WTBS_CLASSES[class_id] if 0 <= class_id < len(WTBS_CLASSES) else str(class_id)
            color = COLORS[class_id % len(COLORS)]
            draw_box(draw, box, name, color, source_size, size)
    return image


def wtbs_samples(class_name: str, count=3):
    image_dir = WTBS / class_name / "images"
    candidates = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not candidates:
        return []
    indexes = sorted(set(round(i * (len(candidates) - 1) / max(1, count - 1)) for i in range(count)))
    return [render_wtbs_sample(candidates[i]) for i in indexes]


def paste_row(canvas, y, heading, samples):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, y, canvas.width, y + 34), fill="#202124")
    draw.text((10, y + 3), heading, fill="white", font=TITLE_FONT)
    for index, sample in enumerate(samples):
        if sample is not None:
            canvas.paste(sample, (index * 320, y + 34))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an annotation audit contact sheet.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT,
        help="Directory containing 'WT blade defect dataset' and 'WTBs2025'.",
    )
    parser.add_argument("--output", type=Path, help="Output JPEG path.")
    return parser.parse_args()


def main():
    global ROOT, BASE, WTBS, OUTPUT
    args = parse_args()
    ROOT = args.workspace.expanduser().resolve()
    BASE = ROOT / "WT blade defect dataset"
    WTBS = ROOT / "WTBs2025"
    OUTPUT = (args.output or ROOT / "dataset_audit_preview.jpg").expanduser().resolve()

    if not BASE.is_dir() or not WTBS.is_dir():
        raise FileNotFoundError(
            f"Expected both source datasets under {ROOT}; got BASE={BASE}, WTBS={WTBS}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = len(BASE_CLASSES) + len(WTBS_CLASSES) + 2
    row_height = 274
    canvas = Image.new("RGB", (960, rows * row_height), "white")
    y = 0
    paste_row(canvas, y, "BASE DATASET — authoritative classes", [])
    y += row_height
    for class_name in BASE_CLASSES:
        paste_row(canvas, y, f"BASE: {class_name}", base_samples(class_name))
        y += row_height
    paste_row(canvas, y, "WTBs2025 — candidate source classes", [])
    y += row_height
    for class_name in WTBS_CLASSES:
        paste_row(canvas, y, f"WTBs2025: {class_name}", wtbs_samples(class_name))
        y += row_height
    canvas.crop((0, 0, canvas.width, y)).save(OUTPUT, quality=90)
    print(OUTPUT)


if __name__ == "__main__":
    main()
