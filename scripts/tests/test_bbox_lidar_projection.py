#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from lidar_projection import query_lidar_depth_in_bbox


def point_for_pixel(u, v, depth, k):
    return np.array([
        (u - k[0, 2]) / k[0, 0] * depth,
        (v - k[1, 2]) / k[1, 1] * depth,
        depth,
    ])


class TestBBoxLidarProjection(unittest.TestCase):
    def setUp(self):
        self.k = np.array([
            [100.0, 0.0, 320.0],
            [0.0, 100.0, 240.0],
            [0.0, 0.0, 1.0],
        ])

    def test_point_far_from_bbox_center_is_selected(self):
        point = point_for_pixel(500.0, 300.0, 5.0, self.k)
        depth = query_lidar_depth_in_bbox(
            np.array([point]),
            300.0, 200.0, 520.0, 320.0,
            self.k, margin=0.0,
        )
        self.assertAlmostEqual(depth, 5.0)

    def test_margin_includes_nearby_point(self):
        point = point_for_pixel(525.0, 250.0, 6.0, self.k)
        depth = query_lidar_depth_in_bbox(
            np.array([point]),
            400.0, 200.0, 520.0, 300.0,
            self.k, margin=10.0,
        )
        self.assertAlmostEqual(depth, 6.0)

    def test_outside_point_is_rejected(self):
        point = point_for_pixel(550.0, 250.0, 6.0, self.k)
        depth = query_lidar_depth_in_bbox(
            np.array([point]),
            400.0, 200.0, 520.0, 300.0,
            self.k, margin=10.0,
        )
        self.assertIsNone(depth)

    def test_iqr_rejects_far_outlier(self):
        points = np.array([
            point_for_pixel(450.0, 260.0, 5.0, self.k),
            point_for_pixel(451.0, 260.0, 5.1, self.k),
            point_for_pixel(452.0, 260.0, 4.9, self.k),
            point_for_pixel(453.0, 260.0, 5.0, self.k),
            point_for_pixel(454.0, 260.0, 50.0, self.k),
        ])
        depth = query_lidar_depth_in_bbox(
            points,
            430.0, 240.0, 470.0, 280.0,
            self.k, margin=0.0,
        )
        self.assertAlmostEqual(depth, 4.9)

    def test_sparse_points_use_nearest_fallback(self):
        points = np.array([
            point_for_pixel(450.0, 260.0, 7.0, self.k),
            point_for_pixel(451.0, 260.0, 5.0, self.k),
        ])
        depth = query_lidar_depth_in_bbox(
            points,
            430.0, 240.0, 470.0, 280.0,
            self.k, margin=0.0,
        )
        self.assertAlmostEqual(depth, 5.0)


if __name__ == "__main__":
    unittest.main()
