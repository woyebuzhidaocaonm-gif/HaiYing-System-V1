import os
import unittest

import numpy as np
import yaml


class IdealExtrinsicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "target_point_ideal_extrinsic.yaml",
        )
        with open(path, "r", encoding="utf-8") as stream:
            parameters = yaml.safe_load(stream)["target_point_node"]["ros__parameters"]
        cls.transform = np.asarray(parameters["lidar_to_camera"]).reshape(4, 4)

    def test_rotation_is_right_handed_and_orthonormal(self):
        rotation = self.transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_lidar_origin_has_expected_camera_optical_position(self):
        origin = self.transform @ np.asarray([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(origin[:3], [0.0, -0.20, -0.15], atol=1e-12)

    def test_lidar_forward_axis_maps_to_camera_forward_axis(self):
        forward_point = self.transform @ np.asarray([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(forward_point[:3], [0.0, -0.20, 0.85], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
