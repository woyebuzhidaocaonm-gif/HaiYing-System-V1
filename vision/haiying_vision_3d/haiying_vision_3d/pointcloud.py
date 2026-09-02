"""PointCloud2 parsing helpers, including Livox mixed-field layouts."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def pointcloud2_to_xyz(message: PointCloud2) -> np.ndarray:
    """Return Nx3 XYZ without requiring every PointCloud2 field to share a type."""
    records = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )
    if len(records) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.column_stack(
        (
            np.asarray(records["x"], dtype=np.float64),
            np.asarray(records["y"], dtype=np.float64),
            np.asarray(records["z"], dtype=np.float64),
        )
    )
