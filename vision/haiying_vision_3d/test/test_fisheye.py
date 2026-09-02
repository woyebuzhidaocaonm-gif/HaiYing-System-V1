import os
import unittest

import cv2
import numpy as np

from haiying_vision_3d.fisheye import (
    create_rectification_maps,
    load_fisheye_calibration,
    make_rectified_camera_matrix,
    rectify_image,
)


class FisheyeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "ar0234_fisheye_1920x1080.yaml"
        )

    def test_loads_measured_ar0234_calibration(self):
        calibration = load_fisheye_calibration(self.calibration_path)
        self.assertEqual((calibration.width, calibration.height), (1920, 1080))
        self.assertEqual(calibration.distortion.shape, (4, 1))
        self.assertAlmostEqual(calibration.camera_matrix[0, 0], 287.9494481646608)

    def test_rectification_map_and_output_shape(self):
        calibration = load_fisheye_calibration(self.calibration_path)
        rectified_matrix = make_rectified_camera_matrix(1920, 1080, 700.0)
        map1, map2 = create_rectification_maps(
            calibration, (1920, 1080), rectified_matrix
        )
        output = rectify_image(
            np.zeros((1080, 1920, 3), dtype=np.uint8), map1, map2
        )
        self.assertEqual(output.shape, (1080, 1920, 3))
        self.assertEqual(map1.dtype, np.int16)

    def test_fisheye_undistortion_makes_checkerboard_rows_straight(self):
        calibration = load_fisheye_calibration(self.calibration_path)
        rectified_matrix = make_rectified_camera_matrix(1920, 1080, 700.0)
        # Synthetic points lying on horizontal rays in the rectified pinhole image.
        rectified_points = np.asarray(
            [[x, 300.0] for x in np.linspace(300.0, 1620.0, 20)], dtype=np.float64
        ).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(
            rectified_points, rectified_matrix, np.zeros(5)
        )
        distorted = cv2.fisheye.distortPoints(
            rays, calibration.camera_matrix, calibration.distortion
        )
        restored = cv2.fisheye.undistortPoints(
            distorted,
            calibration.camera_matrix,
            calibration.distortion,
            P=rectified_matrix,
        ).reshape(-1, 2)
        self.assertLess(float(np.max(np.abs(restored[:, 1] - 300.0))), 1e-8)


if __name__ == "__main__":
    unittest.main()
