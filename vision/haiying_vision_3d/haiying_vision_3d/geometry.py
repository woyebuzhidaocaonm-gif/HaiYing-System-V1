"""Pure geometry routines used by the ROS 2 target-point node."""

from __future__ import annotations

import numpy as np


def deproject_depth_pixel(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Deproject one rectified depth pixel into an optical-frame XYZ point."""
    values = np.asarray([u, v, depth_m, fx, fy, cx, cy], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("pixel, depth, and intrinsics must be finite")
    if depth_m <= 0.0:
        raise ValueError("depth must be positive")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")
    return np.asarray(
        [
            (u - cx) * depth_m / fx,
            (v - cy) * depth_m / fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def median_depth_in_window(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
) -> float | None:
    """Return the median valid depth around a target pixel."""
    if depth_m.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if radius < 0:
        raise ValueError("window radius cannot be negative")
    x = int(round(u))
    y = int(round(v))
    height, width = depth_m.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    values = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    valid = values[
        np.isfinite(values)
        & (values >= min_depth_m)
        & (values <= max_depth_m)
    ]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def select_lidar_point_for_pixel(
    points_lidar: np.ndarray,
    u: float,
    v: float,
    camera_matrix: np.ndarray,
    lidar_to_camera: np.ndarray,
    search_radius_px: float,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray | None:
    """Project LiDAR points into the image and select the closest pixel match.

    The returned XYZ remains in the LiDAR frame. ``lidar_to_camera`` must be a
    calibrated 4x4 transform that maps LiDAR coordinates into the rectified
    camera optical frame.
    """
    points = np.asarray(points_lidar, dtype=np.float64)
    k = np.asarray(camera_matrix, dtype=np.float64)
    transform = np.asarray(lidar_to_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_lidar must have shape (N, 3)")
    if k.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if transform.shape != (4, 4):
        raise ValueError("lidar_to_camera must have shape (4, 4)")
    if search_radius_px <= 0.0:
        raise ValueError("search radius must be positive")

    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.size == 0:
        return None

    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    points_camera = (transform @ homogeneous.T).T[:, :3]
    z = points_camera[:, 2]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid):
        return None
    points = points[valid]
    points_camera = points_camera[valid]
    z = points_camera[:, 2]

    projected = (k @ points_camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    squared_distance = (pixels[:, 0] - u) ** 2 + (pixels[:, 1] - v) ** 2
    candidates = squared_distance <= search_radius_px**2
    if not np.any(candidates):
        return None

    candidate_indices = np.flatnonzero(candidates)
    # Pixel agreement is the primary criterion. For a tie, prefer the nearer
    # surface so a background point does not win at an occlusion boundary.
    order = np.lexsort((z[candidate_indices], squared_distance[candidate_indices]))
    return points[candidate_indices[order[0]]].copy()


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a normalized or non-normalized quaternion to a 3x3 matrix."""
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.dot(quaternion, quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion norm must be non-zero")
    quaternion *= np.sqrt(2.0 / norm)
    q = np.outer(quaternion, quaternion)
    return np.asarray(
        [
            [1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]],
            [q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]],
            [q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1]],
        ],
        dtype=np.float64,
    )


def apply_transform(
    point_xyz: np.ndarray,
    translation_xyz: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Apply a ROS Transform translation and quaternion to one XYZ point."""
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    translation = np.asarray(translation_xyz, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    rotation = quaternion_to_rotation_matrix(*quaternion)
    return rotation @ point + translation
