import numpy as np

from haiying_vision_3d.fusion import estimate_range_in_box, project_lidar_to_image


def test_project_lidar_to_rectified_image():
    points = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    k = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    lidar, camera, pixels = project_lidar_to_image(
        points, k, np.eye(4), (80, 120), 0.2, 10.0
    )
    assert lidar.shape == (2, 3)
    assert np.allclose(camera, lidar)
    assert np.allclose(pixels, [[50.0, 40.0], [100.0, 40.0]])


def test_box_range_uses_point_cluster_and_rejects_depth_outlier():
    lidar = np.asarray(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.02], [-0.1, 0.0, 1.98], [0.0, 0.0, 8.0]]
    )
    camera = lidar.copy()
    pixels = np.asarray([[50.0, 50.0], [52.0, 50.0], [48.0, 50.0], [51.0, 51.0]])
    estimate = estimate_range_in_box(
        lidar,
        camera,
        pixels,
        np.asarray([40.0, 40.0, 60.0, 60.0]),
        min_points=3,
    )
    assert estimate is not None
    assert estimate.point_count == 3
    assert np.allclose(estimate.point_lidar, [0.0, 0.0, 2.0], atol=0.03)
    assert abs(estimate.distance_m - 2.0) < 0.03


def test_box_range_returns_none_for_sparse_box():
    points = np.asarray([[0.0, 0.0, 2.0]])
    pixels = np.asarray([[50.0, 50.0]])
    assert (
        estimate_range_in_box(
            points,
            points,
            pixels,
            np.asarray([40.0, 40.0, 60.0, 60.0]),
            min_points=3,
        )
        is None
    )
