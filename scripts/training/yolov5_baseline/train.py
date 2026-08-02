from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
PROJECT = Path(
    os.environ.get("HAIYING_RUN_ROOT", Path.cwd() / "runs" / "yolov5_baseline")
)
YOLOV5 = Path(os.environ.get("YOLOV5_ROOT", ROOT / "yolov5"))
DATA = ROOT / "WT_blade_merged_dataset" / "data.yaml"
HYP = Path(__file__).resolve().parent / "hyp.yaml"
RUNS = PROJECT / "runs"
TRAIN_RUN = RUNS / "train"
TEST_RUN = RUNS / "test"
WEIGHTS = PROJECT / "weights"

EPOCHS = 20
BATCH = 12
IMAGE_SIZE = 640
PATIENCE = 7
WORKERS = 4
SEED = 2026


def train() -> Path:
    if not (YOLOV5 / "train.py").is_file():
        raise FileNotFoundError(
            f"YOLOv5 source not found at {YOLOV5}. Set the YOLOV5_ROOT environment variable."
        )
    if not DATA.is_file():
        raise FileNotFoundError(f"Merged dataset config not found: {DATA}")
    command = [
        sys.executable,
        "-u",
        str(YOLOV5 / "train.py"),
        "--weights",
        str(YOLOV5 / "yolov5s.pt"),
        "--data",
        str(DATA),
        "--hyp",
        str(HYP),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH),
        "--imgsz",
        str(IMAGE_SIZE),
        "--optimizer",
        "AdamW",
        "--device",
        "0",
        "--workers",
        str(WORKERS),
        "--project",
        str(RUNS),
        "--name",
        "train",
        "--exist-ok",
        "--cos-lr",
        "--patience",
        str(PATIENCE),
        "--seed",
        str(SEED),
    ]
    print("Starting YOLOv5s training:")
    print(subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment["YOLOV5_DISABLE_TENSORBOARD"] = "1"
    subprocess.run(command, cwd=YOLOV5, env=environment, check=True)
    best = TRAIN_RUN / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")
    return best


def evaluate(best: Path) -> dict:
    os.environ["YOLOV5_DISABLE_TENSORBOARD"] = "1"
    sys.path.insert(0, str(YOLOV5))
    from val import run as validate

    results, maps, times = validate(
        data=str(DATA),
        weights=str(best),
        batch_size=BATCH,
        imgsz=IMAGE_SIZE,
        task="test",
        device="0",
        workers=WORKERS,
        project=str(RUNS),
        name="test",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )
    metrics = {
        "model": "yolov5s",
        "dataset": str(DATA),
        "split": "test",
        "epochs_requested": EPOCHS,
        "batch": BATCH,
        "imgsz": IMAGE_SIZE,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "precision": float(results[0]),
        "recall": float(results[1]),
        "map50": float(results[2]),
        "map50_95": float(results[3]),
        "per_class_map50_95": {
            name: float(maps[index])
            for index, name in enumerate(
                ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]
            )
        },
        "timings_ms": {
            "preprocess": float(times[0]),
            "inference": float(times[1]),
            "nms": float(times[2]),
        },
    }
    return metrics


def main():
    for path in (RUNS, WEIGHTS):
        path.mkdir(parents=True, exist_ok=True)
    best = train()
    metrics = evaluate(best)
    shutil.copy2(best, WEIGHTS / "best.pt")
    last = TRAIN_RUN / "weights" / "last.pt"
    if last.is_file():
        shutil.copy2(last, WEIGHTS / "last.pt")
    (PROJECT / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"YOLOv5 training complete: {PROJECT}", flush=True)


if __name__ == "__main__":
    main()
