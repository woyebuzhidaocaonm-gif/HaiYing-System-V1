"""Verify that Python is using the expected GPU-enabled YOLO environment."""

import sys

import cv2
import torch
import ultralytics


def main() -> None:
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"Ultralytics       : {ultralytics.__version__}")
    print(f"OpenCV version    : {cv2.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Select a CUDA-enabled PyTorch environment "
            "before running GPU training."
        )

    print(f"GPU device        : {torch.cuda.get_device_name(0)}")
    print("YOLO GPU environment is ready.")


if __name__ == "__main__":
    main()
