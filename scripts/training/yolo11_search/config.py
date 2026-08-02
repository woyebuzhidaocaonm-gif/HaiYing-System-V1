"""Central configuration for the wind-turbine-blade YOLO project.

Edit this file when the project is moved or when training settings need to be
changed.  Every path is derived from this file, so the project works in
PyCharm without depending on its working-directory setting.
"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.environ.get("HAIYING_DATA_ROOT", Path.cwd()))
ARTIFACT_DIR = Path(
    os.environ.get("HAIYING_RUN_ROOT", Path.cwd() / "runs" / "yolo11_search")
)
SOURCE_DIR = WORKSPACE_DIR / "WT blade defect dataset"
SOURCE_IMAGES = SOURCE_DIR / "JPEGImages"
# label1 is used because it contains no empty annotations and has 1584 boxes;
# label2 contains 28 empty annotations and has 1543 boxes.
SOURCE_LABELS = SOURCE_DIR / "label1"
CLASS_FILE = SOURCE_DIR / "class_definitions.txt"

DATASET_DIR = ARTIFACT_DIR / "dataset_1000"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = ARTIFACT_DIR / "runs"
OUTPUT_DIR = ARTIFACT_DIR / "output"

SEED = 2026
TOTAL_IMAGES = 1000
SPLIT_SIZES = {"train": 800, "val": 100, "test": 100}
EXPECTED_CLASSES = 6


@dataclass(frozen=True)
class Trial:
    """One validation-set screening experiment."""

    name: str
    model: str
    imgsz: int
    batch: int
    optimizer: str
    lr0: float


# These trials cover the useful speed/accuracy range of an 8 GB RTX 4060.
# Test-set results are deliberately not used during this search.
SEARCH_EPOCHS = 30
TRIALS = (
    Trial("yolo11n_640_adamw", "yolo11n.pt", 640, 16, "AdamW", 0.001),
    Trial("yolo11s_640_adamw", "yolo11s.pt", 640, 12, "AdamW", 0.001),
    Trial("yolo11s_768_adamw", "yolo11s.pt", 768, 8, "AdamW", 0.001),
)

FINAL_EPOCHS = 100
PATIENCE = 20
WORKERS = 4
DEVICE = 0  # RTX 4060; use "cpu" only when CUDA is unavailable.
