from glob import glob
from setuptools import find_packages, setup


package_name = "haiying_vision_3d"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="HaiYing Team",
    maintainer_email="jokei@example.com",
    description="Publish measured 3D target coordinates from depth or LiDAR data.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "calibration_node = haiying_vision_3d.calibration_node:main",
            "fisheye_rectifier_node = haiying_vision_3d.fisheye_rectifier_node:main",
            "target_point_node = haiying_vision_3d.target_point_node:main",
            "yolo_target_pixel_node = haiying_vision_3d.yolo_target_pixel_node:main",
            "yolo_lidar_fusion_node = haiying_vision_3d.yolo_lidar_fusion_node:main",
            "verify_runtime = haiying_vision_3d.runtime_verifier:main",
        ],
    },
)
