import math
import unittest

import numpy as np

from haiying_vision_3d.detector import decode_yolov5_predictions, letterbox
from haiying_vision_3d.geometry import (
    apply_transform,
    deproject_depth_pixel,
    median_depth_in_window,
    select_lidar_point_for_pixel,
)


class GeometryTests(unittest.TestCase):
    def test_deproject_depth_pixel(self):
        point = deproject_depth_pixel(
            420.0, 290.0, 2.0, 500.0, 500.0, 320.0, 240.0
        )
        np.testing.assert_allclose(point, [0.4, 0.2, 2.0])

    def test_median_depth_ignores_invalid_values(self):
        depth = np.asarray(
            [[0.0, math.nan, 2.0], [1.8, 2.2, 100.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
        self.assertEqual(
            median_depth_in_window(depth, 1, 1, 1, 0.2, 10.0), 2.0
        )

    def test_select_lidar_point_uses_calibrated_projection(self):
        points = np.asarray(
            [[0.0, 0.0, 2.0], [0.4, 0.2, 2.0], [5.0, 5.0, -1.0]]
        )
        camera_matrix = np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0, 0, 1]]
        )
        selected = select_lidar_point_for_pixel(
            points,
            420.0,
            290.0,
            camera_matrix,
            np.eye(4),
            3.0,
            0.2,
            30.0,
        )
        np.testing.assert_allclose(selected, [0.4, 0.2, 2.0])

    def test_select_lidar_point_returns_none_without_pixel_match(self):
        selected = select_lidar_point_for_pixel(
            np.asarray([[0.0, 0.0, 2.0]]),
            600.0,
            400.0,
            np.asarray(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0, 0, 1]]
            ),
            np.eye(4),
            3.0,
            0.2,
            30.0,
        )
        self.assertIsNone(selected)

    def test_apply_transform_translation_and_rotation(self):
        half = math.sqrt(0.5)
        transformed = apply_transform([1, 0, 0], [1, 2, 3], [0, 0, half, half])
        np.testing.assert_allclose(transformed, [1, 3, 3], atol=1e-12)

    def test_letterbox_and_yolov5_decode_restore_original_pixel(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        prepared, scale, pad_x, pad_y = letterbox(image, 640)
        self.assertEqual(prepared.shape, (640, 640, 3))
        self.assertEqual((scale, pad_x, pad_y), (1.0, 0, 80))
        prediction = np.zeros((1, 1, 11), dtype=np.float32)
        prediction[0, 0, :4] = [420.0, 370.0, 100.0, 80.0]
        prediction[0, 0, 4] = 0.9
        prediction[0, 0, 7] = 0.8
        detections = decode_yolov5_predictions(
            prediction, (480, 640), scale, pad_x, pad_y, 0.25, 0.45
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_id"], 2)
        np.testing.assert_allclose(detections[0]["center"], [420.0, 290.0])
        self.assertAlmostEqual(detections[0]["confidence"], 0.72, places=6)


if __name__ == "__main__":
    unittest.main()
