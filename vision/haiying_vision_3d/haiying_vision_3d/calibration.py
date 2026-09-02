"""Validated rigid-transform helpers for camera/LiDAR calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import quaternion_to_rotation_matrix


def normalize_quaternion(quaternion_xyzw) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("quaternion norm must be non-zero")
    return quaternion / norm


def transform_to_matrix(transform: dict[str, Any]) -> np.ndarray:
    """Convert a translation/quaternion mapping to a 4x4 transform matrix."""
    if not isinstance(transform, dict):
        raise ValueError("transform must be a mapping")
    translation = np.asarray(transform.get("translation"), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain finite values")
    quaternion = normalize_quaternion(transform.get("rotation"))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_matrix(*quaternion)
    matrix[:3, 3] = translation
    return matrix


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("rotation matrix must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError("rotation matrix determinant must be +1")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return normalize_quaternion([x, y, z, w])


def matrix_to_transform(matrix: np.ndarray) -> dict[str, list[float]]:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform matrix must be finite and 4x4")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("invalid homogeneous transform bottom row")
    quaternion = rotation_matrix_to_quaternion(transform[:3, :3])
    return {
        "translation": transform[:3, 3].astype(float).tolist(),
        "rotation": quaternion.astype(float).tolist(),
    }


def compute_lidar_to_camera(
    camera_in_base: dict[str, Any], lidar_in_base: dict[str, Any]
) -> np.ndarray:
    """Return T_camera_lidar, mapping LiDAR XYZ into camera coordinates."""
    base_from_camera = transform_to_matrix(camera_in_base)
    base_from_lidar = transform_to_matrix(lidar_in_base)
    return np.linalg.inv(base_from_camera) @ base_from_lidar


def load_calibration_file(
    path: str, *, simulation_mode: bool, max_reprojection_error_px: float
) -> dict[str, Any]:
    """Load and validate a real or simulation calibration file."""
    calibration_path = Path(path).expanduser()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"calibration file not found: {calibration_path}")
    with calibration_path.open("r", encoding="utf-8") as stream:
        if calibration_path.suffix.lower() == ".json":
            data = json.load(stream)
        else:
            data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("calibration file root must be a mapping")

    metadata = data.get("calibration", {})
    status = str(metadata.get("status", data.get("status", ""))).lower()
    allowed = {"simulation", "calibrated"} if simulation_mode else {"calibrated"}
    if status not in allowed:
        raise ValueError(
            f"calibration status must be one of {sorted(allowed)}, received {status!r}"
        )

    camera_in_base = data.get("camera_in_base")
    lidar_in_base = data.get("lidar_in_base")
    transform_to_matrix(camera_in_base)
    transform_to_matrix(lidar_in_base)

    validation = data.get("validation", {})
    reprojection_error = validation.get("reprojection_error_px")
    if not simulation_mode:
        if reprojection_error is None or not math.isfinite(float(reprojection_error)):
            raise ValueError("real calibration requires finite reprojection_error_px")
        if float(reprojection_error) > max_reprojection_error_px:
            raise ValueError(
                f"reprojection error {float(reprojection_error):.3f}px exceeds "
                f"limit {max_reprojection_error_px:.3f}px"
            )
    return data
