"""Pure camera/LiDAR projection and robust box association helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxRangeEstimate:
    """Representative LiDAR point and range associated with one image box."""

    point_lidar: np.ndarray
    distance_m: float
    point_count: int
    points_lidar: np.ndarray
    pixels: np.ndarray
    depths_m: np.ndarray


def project_lidar_to_image(
    points_lidar: np.ndarray,
    camera_matrix: np.ndarray,
    lidar_to_camera: np.ndarray,
    image_shape: tuple[int, int],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project valid LiDAR points into a rectified camera image.

    Returns matching arrays containing LiDAR-frame XYZ, camera-frame XYZ, and
    floating-point image coordinates. Points outside the image are discarded.
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
    if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("image_shape must be (height, width)")
    if min_depth_m <= 0.0 or max_depth_m <= min_depth_m:
        raise ValueError("depth limits are invalid")

    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.size == 0:
        empty3 = np.empty((0, 3), dtype=np.float64)
        return empty3, empty3.copy(), np.empty((0, 2), dtype=np.float64)

    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    points_camera = (transform @ homogeneous.T).T[:, :3]
    depth = points_camera[:, 2]
    valid = (
        np.all(np.isfinite(points_camera), axis=1)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    points = points[valid]
    points_camera = points_camera[valid]
    if points.size == 0:
        empty3 = np.empty((0, 3), dtype=np.float64)
        return empty3, empty3.copy(), np.empty((0, 2), dtype=np.float64)

    projected = (k @ points_camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    height, width = image_shape
    inside = (
        np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    return points[inside], points_camera[inside], pixels[inside]


def estimate_range_in_box(
    points_lidar: np.ndarray,
    points_camera: np.ndarray,
    pixels: np.ndarray,
    box_xyxy: np.ndarray,
    min_points: int = 3,
    inner_margin_ratio: float = 0.08,
    fallback_expand_px: float = 8.0,
    mad_scale: float = 3.0,
    minimum_depth_gate_m: float = 0.12,
) -> BoxRangeEstimate | None:
    """Estimate a robust target point from projected points inside a YOLO box."""
    lidar = np.asarray(points_lidar, dtype=np.float64)
    camera = np.asarray(points_camera, dtype=np.float64)
    uv = np.asarray(pixels, dtype=np.float64)
    box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
    if lidar.ndim != 2 or lidar.shape[1] != 3:
        raise ValueError("points_lidar must have shape (N, 3)")
    if camera.shape != lidar.shape or uv.shape != (lidar.shape[0], 2):
        raise ValueError("projected point arrays must have matching lengths")
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError("box_xyxy must contain four finite values")
    if min_points < 1:
        raise ValueError("min_points must be positive")
    if not 0.0 <= inner_margin_ratio < 0.5:
        raise ValueError("inner_margin_ratio must be in [0, 0.5)")

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    margin_x = (x2 - x1) * inner_margin_ratio
    margin_y = (y2 - y1) * inner_margin_ratio

    def select(left: float, top: float, right: float, bottom: float) -> np.ndarray:
        return (
            (uv[:, 0] >= left)
            & (uv[:, 0] <= right)
            & (uv[:, 1] >= top)
            & (uv[:, 1] <= bottom)
        )

    selected = select(x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y)
    if int(selected.sum()) < min_points:
        selected = select(
            x1 - fallback_expand_px,
            y1 - fallback_expand_px,
            x2 + fallback_expand_px,
            y2 + fallback_expand_px,
        )
    if int(selected.sum()) < min_points:
        return None

    selected_lidar = lidar[selected]
    selected_camera = camera[selected]
    selected_uv = uv[selected]
    depths = selected_camera[:, 2]
    median_depth = float(np.median(depths))
    mad = float(np.median(np.abs(depths - median_depth)))
    robust_sigma = 1.4826 * mad
    depth_gate = max(minimum_depth_gate_m, mad_scale * robust_sigma)
    inliers = np.abs(depths - median_depth) <= depth_gate
    if int(inliers.sum()) < min_points:
        # Sparse clouds can legitimately have different depths across a sloped
        # surface. In that case retain the original box points.
        inliers = np.ones(depths.shape, dtype=bool)

    inlier_lidar = selected_lidar[inliers]
    representative = np.median(inlier_lidar, axis=0)
    distance = float(np.linalg.norm(representative))
    return BoxRangeEstimate(
        point_lidar=representative,
        distance_m=distance,
        point_count=int(inliers.sum()),
        points_lidar=inlier_lidar,
        pixels=selected_uv[inliers],
        depths_m=depths[inliers],
    )
