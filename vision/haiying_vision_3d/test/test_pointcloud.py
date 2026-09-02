import struct
import unittest

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from haiying_vision_3d.pointcloud import pointcloud2_to_xyz


class PointCloudTests(unittest.TestCase):
    def test_reads_livox_mixed_field_pointcloud2(self):
        message = PointCloud2()
        message.height = 1
        message.width = 2
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="tag", offset=16, datatype=PointField.UINT8, count=1),
            PointField(name="line", offset=17, datatype=PointField.UINT8, count=1),
            PointField(name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 26
        message.row_step = 52
        message.is_dense = True
        message.data = b"".join(
            [
                struct.pack("<ffffBBd", 1.0, 2.0, 3.0, 42.0, 1, 2, 10.0),
                struct.pack("<ffffBBd", -1.0, -2.0, -3.0, 7.0, 3, 4, 11.0),
            ]
        )

        xyz = pointcloud2_to_xyz(message)

        np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])


if __name__ == "__main__":
    unittest.main()
