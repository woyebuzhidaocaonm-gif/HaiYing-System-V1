import json
import tempfile
import unittest

import numpy as np

from haiying_vision_3d.calibration import (
    compute_lidar_to_camera,
    load_calibration_file,
    matrix_to_transform,
    rotation_matrix_to_quaternion,
    transform_to_matrix,
)


class CalibrationTests(unittest.TestCase):
    def test_compute_lidar_to_camera_composes_back_to_base(self):
        camera = {
            "translation": [0.12, 0.03, 0.242],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        lidar = {
            "translation": [0.0, 0.0, 0.30],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        camera_from_lidar = compute_lidar_to_camera(camera, lidar)
        np.testing.assert_allclose(
            camera_from_lidar[:3, 3], [-0.12, -0.03, 0.058], atol=1e-12
        )
        np.testing.assert_allclose(
            transform_to_matrix(camera) @ camera_from_lidar,
            transform_to_matrix(lidar),
            atol=1e-12,
        )

    def test_rotation_matrix_to_quaternion_handles_180_degrees(self):
        rotation = np.diag([1.0, -1.0, -1.0])
        quaternion = rotation_matrix_to_quaternion(rotation)
        self.assertAlmostEqual(abs(quaternion[0]), 1.0, places=12)
        restored = transform_to_matrix(
            {"translation": [0, 0, 0], "rotation": quaternion.tolist()}
        )
        np.testing.assert_allclose(restored[:3, :3], rotation, atol=1e-12)

    def test_matrix_transform_roundtrip(self):
        source = {
            "translation": [1.0, -2.0, 3.0],
            "rotation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
        }
        matrix = transform_to_matrix(source)
        np.testing.assert_allclose(
            transform_to_matrix(matrix_to_transform(matrix)), matrix, atol=1e-12
        )

    def test_real_file_requires_valid_status_and_error(self):
        data = {
            "calibration": {"status": "calibrated"},
            "camera_in_base": {
                "translation": [0, 0, 0],
                "rotation": [0, 0, 0, 1],
            },
            "lidar_in_base": {
                "translation": [0.1, 0, 0],
                "rotation": [0, 0, 0, 1],
            },
            "validation": {"reprojection_error_px": 1.2},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
            json.dump(data, stream)
            path = stream.name
        loaded = load_calibration_file(
            path, simulation_mode=False, max_reprojection_error_px=3.0
        )
        self.assertEqual(loaded["calibration"]["status"], "calibrated")

        data["validation"]["reprojection_error_px"] = 4.0
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
            json.dump(data, stream)
            bad_path = stream.name
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_calibration_file(
                bad_path, simulation_mode=False, max_reprojection_error_px=3.0
            )


if __name__ == "__main__":
    unittest.main()
