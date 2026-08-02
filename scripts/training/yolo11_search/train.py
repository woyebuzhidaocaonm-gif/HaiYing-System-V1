"""Run validation-based YOLO parameter search, final training, and one test."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from config import (
    DATASET_DIR,
    DATA_YAML,
    DEVICE,
    FINAL_EPOCHS,
    OUTPUT_DIR,
    PATIENCE,
    RUNS_DIR,
    SEARCH_EPOCHS,
    SEED,
    TRIALS,
    WORKERS,
    Trial,
)
from prepare_dataset import prepare


def ensure_environment() -> None:
    if not torch.cuda.is_available() and DEVICE != "cpu":
        raise RuntimeError(
            "CUDA is unavailable. In PyCharm select the Python 3.9 interpreter "
            "that reports PyTorch + cu118, or set DEVICE='cpu' in config.py."
        )
    if not DATA_YAML.is_file():
        print("Prepared dataset not found; creating it now...")
        prepare()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def metrics_dict(metrics: Any) -> dict[str, float]:
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def common_train_args(trial: Trial, epochs: int, project: Path, name: str) -> dict[str, Any]:
    return {
        "data": str(DATA_YAML),
        "epochs": epochs,
        "imgsz": trial.imgsz,
        "batch": trial.batch,
        "device": DEVICE,
        "workers": WORKERS,
        "optimizer": trial.optimizer,
        "lr0": trial.lr0,
        "patience": PATIENCE,
        "seed": SEED,
        "deterministic": True,
        "amp": True,
        "cos_lr": True,
        "close_mosaic": min(10, max(0, epochs // 3)),
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }


def run_trial(trial: Trial) -> dict[str, Any]:
    result_file = RUNS_DIR / "search" / trial.name / "trial_result.json"
    if result_file.is_file():
        print(f"Skipping completed trial: {trial.name}")
        return json.loads(result_file.read_text(encoding="utf-8"))

    print(f"\n=== Screening {trial.name} ===")
    model = YOLO(trial.model)
    model.train(**common_train_args(trial, SEARCH_EPOCHS, RUNS_DIR / "search", trial.name))
    best_path = RUNS_DIR / "search" / trial.name / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Training did not produce {best_path}")
    # Explicit validation makes all trials comparable and avoids reading a
    # version-dependent results.csv column layout.
    val_metrics = YOLO(str(best_path)).val(
        data=str(DATA_YAML),
        split="val",
        imgsz=trial.imgsz,
        batch=trial.batch,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_DIR / "search_validation"),
        name=trial.name,
        exist_ok=True,
        plots=True,
    )
    result: dict[str, Any] = {**asdict(trial), "epochs": SEARCH_EPOCHS, **metrics_dict(val_metrics)}
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_search_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(results, key=lambda row: (row["map50_95"], row["map50"], row["recall"]), reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "name", "model", "imgsz", "batch", "optimizer", "lr0", "epochs", "precision", "recall", "map50", "map50_95"]
    with (OUTPUT_DIR / "search_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({"rank": rank, **row})
    (OUTPUT_DIR / "search_summary.json").write_text(
        json.dumps({"selection_metric": "validation map50_95", "results": ranked}, indent=2),
        encoding="utf-8",
    )
    print("\nValidation ranking:")
    for rank, row in enumerate(ranked, 1):
        print(f"  {rank}. {row['name']}: mAP50-95={row['map50_95']:.4f}, mAP50={row['map50']:.4f}")
    return ranked[0]


def search() -> dict[str, Any]:
    ensure_environment()
    return write_search_summary([run_trial(trial) for trial in TRIALS])


def find_trial(name: str) -> Trial:
    for trial in TRIALS:
        if trial.name == name:
            return trial
    raise KeyError(f"Unknown trial: {name}")


def train_final(best_result: dict[str, Any]) -> Path:
    trial = find_trial(best_result["name"])
    final_dir = RUNS_DIR / "final" / trial.name
    final_result_file = final_dir / "final_result.json"
    best_path = final_dir / "weights" / "best.pt"
    if final_result_file.is_file() and best_path.is_file():
        print(f"Skipping completed final training: {trial.name}")
        return best_path

    print(f"\n=== Final training: {trial.name}, up to {FINAL_EPOCHS} epochs ===")
    model = YOLO(trial.model)
    model.train(**common_train_args(trial, FINAL_EPOCHS, RUNS_DIR / "final", trial.name))
    if not best_path.is_file():
        raise FileNotFoundError(f"Final training did not produce {best_path}")

    # The held-out test split is evaluated exactly once, after all choices.
    test_metrics = YOLO(str(best_path)).val(
        data=str(DATA_YAML),
        split="test",
        imgsz=trial.imgsz,
        batch=trial.batch,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_DIR / "test"),
        name=trial.name,
        exist_ok=True,
        plots=True,
        save_json=True,
    )
    report = {
        "selected_by": "highest validation map50_95 in screening",
        "selected_trial": asdict(trial),
        "screening_validation": {key: best_result[key] for key in ("precision", "recall", "map50", "map50_95")},
        "held_out_test": metrics_dict(test_metrics),
        "dataset": {"train": 800, "val": 100, "test": 100},
        "seed": SEED,
    }
    final_result_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "final_test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(best_path, OUTPUT_DIR / "best.pt")
    last_path = final_dir / "weights" / "last.pt"
    if last_path.is_file():
        shutil.copy2(last_path, OUTPUT_DIR / "last.pt")
    print(f"Best deployable weights: {OUTPUT_DIR / 'best.pt'}")
    print(f"Held-out test mAP50-95: {report['held_out_test']['map50_95']:.4f}")
    return best_path


def run_all() -> None:
    best_result = search()
    train_final(best_result)


if __name__ == "__main__":
    # In PyCharm, right-click this file and choose Run 'train_yolo'.
    run_all()
