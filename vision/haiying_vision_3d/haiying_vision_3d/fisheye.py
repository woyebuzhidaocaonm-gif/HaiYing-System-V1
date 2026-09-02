"""Pure OpenCV helpers for ROS equidistant/fisheye calibration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class FisheyeCalibration:
    width: int
    height: int
    camera_name: str
    camera_matrix: np.ndarray
    distortion: np.ndarray


def _matrix(data: dict, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.asarray(data.get("data", []), dtype=np.float64)
    if values.size != shape[0] * shape[1]:
        raise ValueError(f"{name} must contain {shape[0] * shape[1]} values")
    matrix = values.reshape(shape)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def load_fisheye_calibration(path: str | Path) -> FisheyeCalibration:
    """Load and validate a ROS camera calibration YAML file."""
    calibration_path = Path(path).expanduser()
    with calibration_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("calibration file must contain a YAML mapping")
    if data.get("distortion_model") != "equidistant":
        raise ValueError("AR0234 fisheye calibration must use distortion_model=equidistant")

    width = int(data.get("image_width", 0))
    height = int(data.get("image_height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("image_width and image_height must be positive")

    camera_matrix = _matrix(data.get("camera_matrix", {}), (3, 3), "camera_matrix")
    distortion = _matrix(
        data.get("distortion_coefficients", {}), (1, 4), "distortion_coefficients"
    ).reshape(4, 1)
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    return FisheyeCalibration(
        width=width,
        height=height,
        camera_name=str(data.get("camera_name", "fisheye_camera")),
        camera_matrix=camera_matrix,
        distortion=distortion,
    )

def make_rectified_camera_matrix(
    width: int, height: int, focal_length: float
) -> np.ndarray:
    """Create a centered pinhole matrix for the rectified output."""
    if width <= 0 or height <= 0 or focal_length <= 0.0:
        raise ValueError("output dimensions and focal_length must be positive")
    return np.asarray(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def create_rectification_maps(
    calibration: FisheyeCalibration,
    output_size: tuple[int, int],
    rectified_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-point remap tables once for low-overhead real-time use."""
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size must be positive")
    return cv2.fisheye.initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.distortion,
        np.eye(3, dtype=np.float64),
        np.asarray(rectified_camera_matrix, dtype=np.float64),
        (width, height),
        cv2.CV_16SC2,
    )


def rectify_image(
    image: np.ndarray, map1: np.ndarray, map2: np.ndarray
) -> np.ndarray:
    """Apply a precomputed fisheye-to-pinhole mapping."""
    if image is None or image.size == 0:
        raise ValueError("input image is empty")
    return cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
