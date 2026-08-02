from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
PROJECT = Path(
    os.environ.get("HAIYING_RUN_ROOT", Path.cwd() / "runs" / "yolo11_baseline")
)
DATA = ROOT / "WT_blade_merged_dataset" / "data.yaml"
PRETRAINED = os.environ.get("HAIYING_YOLO11_WEIGHTS", "yolo11s.pt")
RUNS = PROJECT / "runs"
WEIGHTS = PROJECT / "weights"

EPOCHS = 20
BATCH = 12
IMAGE_SIZE = 640
PATIENCE = 7
WORKERS = 4
SEED = 2026
CLASS_NAMES = [
    "craze",
    "corrosion",
    "surface_injure",
    "thunderstrike",
    "crack",
    "hide_craze",
]


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    if not DATA.is_file():
        raise FileNotFoundError(f"Merged dataset config not found: {DATA}")

    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=1.0,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        amp=True,
        cos_lr=True,
        close_mosaic=5,
        cache=False,
        project=str(RUNS),
        name="train",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best = RUNS / "train" / "weights" / "best.pt"
    last = RUNS / "train" / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")

    test_result = YOLO(str(best)).val(
        data=str(DATA),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        project=str(RUNS),
        name="test",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )
    box = test_result.box
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(box.p[index]),
            "recall": float(box.r[index]),
            "map50": float(box.ap50[index]),
            "map50_95": float(box.maps[index]),
        }
    metrics = {
        "model": "yolo11s",
        "dataset": str(DATA),
        "split": "test",
        "epochs_requested": EPOCHS,
        "batch": BATCH,
        "imgsz": IMAGE_SIZE,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        "speed_ms": {key: float(value) for key, value in test_result.speed.items()},
    }
    shutil.copy2(best, WEIGHTS / "best.pt")
    if last.is_file():
        shutil.copy2(last, WEIGHTS / "last.pt")
    (PROJECT / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"YOLO11 training complete: {PROJECT}", flush=True)


if __name__ == "__main__":
    main()
