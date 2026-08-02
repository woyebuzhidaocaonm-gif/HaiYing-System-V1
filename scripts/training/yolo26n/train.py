from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


WORKSPACE = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
PROJECT = Path(os.environ.get("HAIYING_RUN_ROOT", Path.cwd() / "runs" / "yolo26n"))
DATA = PROJECT / "dataset" / "data.yaml"
PRETRAINED = os.environ.get("HAIYING_YOLO26_WEIGHTS", "yolo26n.pt")
RUNS = PROJECT / "runs"
EXPORTS = PROJECT / "weights"

EPOCHS = 30
BATCH = 12
IMAGE_SIZE = 640
PATIENCE = 8
WORKERS = 4
SEED = 2026


def main() -> None:
    if not DATA.is_file():
        raise FileNotFoundError(f"Run prepare_dataset.py first: {DATA}")
    RUNS.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)

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
        save_period=5,
        plots=True,
        verbose=True,
    )

    best = RUNS / "train" / "weights" / "best.pt"
    last = RUNS / "train" / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")

    result = YOLO(str(best)).val(
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
    box = result.box
    names = result.names
    per_class = {
        names[index]: {
            "precision": float(box.p[index]),
            "recall": float(box.r[index]),
            "map50": float(box.ap50[index]),
            "map50_95": float(box.maps[index]),
        }
        for index in range(len(names))
    }
    metrics = {
        "model": "yolo26n",
        "split": "test",
        "epochs_requested": EPOCHS,
        "epochs_completed": len(result.results_dict) if False else None,
        "batch": BATCH,
        "imgsz": IMAGE_SIZE,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        "speed_ms": {key: float(value) for key, value in result.speed.items()},
    }
    shutil.copy2(best, EXPORTS / "best.pt")
    if last.is_file():
        shutil.copy2(last, EXPORTS / "last.pt")
    (PROJECT / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
