"""Simple image/folder inference entry point for PyCharm."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import DEVICE, OUTPUT_DIR, RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image, directory, video path, URL, or camera index.")
    parser.add_argument("--weights", type=Path, default=OUTPUT_DIR / "best.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Run train_yolo.py first; weights not found: {weights}")
    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        imgsz=640,
        conf=args.confidence,
        iou=0.60,
        device=DEVICE,
        save=True,
        project=str(RUNS_DIR / "predict"),
        name="result",
        exist_ok=True,
    )
    print(f"Predictions saved to {RUNS_DIR / 'predict' / 'result'}")


if __name__ == "__main__":
    main()
