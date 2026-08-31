#!/usr/bin/env python3
"""
点云→图像投影模块（V9.2 视觉链核心，PM 冻结方案「RGB + 雷达点云定位」）
======================================================================

本模块负责三步，对应 PM 冻结结论：
1. 雷达点云从 mid360_link 变换到 ar0234_camera_link（X前/Y左/Z上，FLU）；
2. **ar0234_camera_link → ar0234_camera_optical_frame 坐标转换**
   （FLU → RDF：Z前/X右/Y下）——使用 CameraInfo 内参投影前的必备步骤，
   PM 要求提供实际代码，见 link_to_optical()；
3. 用 /drone/camera/camera_info 的内参 K 把 optical 系 3D 点投影到像素，
   并在 YOLO 框中心附近筛选点云返回稳健深度。

本模块不依赖 ROS（纯 numpy），可在无 ROS 环境离线单测（scripts/tests/）。
"""

import numpy as np

# --------------------------------------------------------------------------
# 坐标系定义（PM 冻结 / REP-103）
# --------------------------------------------------------------------------
# ar0234_camera_link（FLU，前-左-上）:
#   X 前,  Y 左,  Z 上
# ar0234_camera_optical_frame（RDF，右-下-前）:
#   X 右,  Y 下,  Z 前
#
# FLU → RDF 转换（正交旋转，无平移）:
#   x_opt = -y_link     （左 → 右）
#   y_opt = -z_link     （上 → 下）
#   z_opt =  x_link     （前 → 前）
#
# 与 V9.2 launch 中静态 TF 完全一致：
#   ar0234_camera_link → ar0234_camera_optical_frame
#   rpy = (-pi/2, 0, -pi/2)
# 即 R_link_to_optical = Rz(-pi/2) @ Rx(-pi/2) = [[0,0,1],[-1,0,0],[0,-1,0]]
# 其转置即为下方矩阵（离线测试 1 有等价性断言）。
LINK_TO_OPTICAL_R = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)


def link_to_optical(points):
    """把 ar0234_camera_link(FLU) 坐标批量转到 ar0234_camera_optical_frame(RDF)。

    实际代码（PM 要求）：p_opt = LINK_TO_OPTICAL_R @ p_link。
    points: (N,3) ndarray，单位米。
    返回:   (N,3) optical 坐标（X右/Y下/Z前）。
    """
    points = np.asarray(points, dtype=np.float64)
    return points @ LINK_TO_OPTICAL_R.T


def optical_to_link(points):
    """逆变换 RDF → FLU（LINK_TO_OPTICAL_R 的转置，供调试/验证用）。"""
    points = np.asarray(points, dtype=np.float64)
    return points @ LINK_TO_OPTICAL_R


def project_optical_to_pixel(points_optical, K):
    """用 CameraInfo 内参 K 把 optical 系 3D 点投影到像素坐标。

    points_optical: (N,3)（X右/Y下/Z前），Z 必须 > 0（相机前方）。
    K: (3,3) 内参矩阵 [fx 0 cx; 0 fy cy; 0 0 1]。
    返回: (N,2) 像素 (u,v)；Z <= 0 或非法点的对应行返回 (-1,-1)。
    """
    pts = np.asarray(points_optical, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    out = np.full((len(pts), 2), -1.0, dtype=np.float64)
    if len(pts) == 0:
        return out
    z = pts[:, 2]
    valid = z > 1e-6
    if np.any(valid):
        u = pts[valid, 0] / z[valid] * K[0, 0] + K[0, 2]
        v = pts[valid, 1] / z[valid] * K[1, 1] + K[1, 2]
        out[valid, 0] = u
        out[valid, 1] = v
    return out


def pc2_to_xyz(pc2):
    """从 sensor_msgs/PointCloud2（或其离线替身）解析 xyz，不假设字段顺序。

    按 msg.fields 的 name/offset/datatype 读取；只支持 FLOAT32(7)，
    不支持的类型抛出 ValueError（V9.2 mid360 输出 xyz+intensity 均为 FLOAT32）。
    返回: (N,3) ndarray。
    """
    data = pc2.data
    point_step = pc2.point_step
    n = len(data) // point_step
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # 按每个字段的 offset 批量读取（FLOAT32 通用快路径）
    offsets = {}
    for f in pc2.fields:
        if f.name in ('x', 'y', 'z'):
            if f.datatype != 7:  # sensor_msgs PointField.FLOAT32
                raise ValueError(
                    f"字段 {f.name} datatype={f.datatype} 不支持，"
                    "仅支持 FLOAT32(7)")
            offsets[f.name] = f.offset

    if not offsets:
        raise ValueError("PointCloud2 缺少 x/y/z 字段")

    out = np.zeros((n, 3), dtype=np.float64)
    if point_step % 4 == 0 and all(o % 4 == 0 for o in offsets.values()):
        raw = np.frombuffer(
            data, dtype=np.float32,
            count=n * (point_step // 4)).reshape(n, point_step // 4)
        for name, off in offsets.items():
            col = {'x': 0, 'y': 1, 'z': 2}[name]
            out[:, col] = raw[:, off // 4]
    else:
        # 非 4 字节对齐的罕见布局：逐点 struct 读取
        import struct
        for i in range(n):
            base = i * point_step
            for name, off in offsets.items():
                col = {'x': 0, 'y': 1, 'z': 2}[name]
                (out[i, col],) = struct.unpack_from(
                    '<f', data, base + off)
    return out


def query_lidar_depth(points_optical, u, v, K, radius=15.0):
    """在像素 (u,v) 半径 radius 像素内筛选点云，返回稳健深度。

    稳健策略（抗离群点）：
      1. 取 optical Z > 0.1m（V9.2 mid360 最近量程 0.1m）的点；
      2. 投影后按像素距离 <= radius 筛选；
      3. 中位数深度 median_z，再在中位数 ±50% 窗口内取最近点深度。
    points_optical: (N,3) 已转到 optical 系的点云。
    返回: 深度(米, optical Z)；区域内无有效点返回 None。
    """
    points_optical = np.asarray(points_optical, dtype=np.float64)
    if len(points_optical) == 0:
        return None
    z = points_optical[:, 2]
    front = (z > 0.1) & np.isfinite(points_optical[:, 0]) & \
            np.isfinite(points_optical[:, 1]) & np.isfinite(z)
    if not np.any(front):
        return None

    pts = points_optical[front]
    zf = pts[:, 2]
    px = project_optical_to_pixel(pts, K)
    dist = np.hypot(px[:, 0] - u, px[:, 1] - v)
    in_win = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (dist <= radius)
    if not np.any(in_win):
        return None

    zs = zf[in_win]
    median_z = float(np.median(zs))
    lo, hi = median_z * 0.5, median_z * 1.5
    robust = zs[(zs >= lo) & (zs <= hi)]
    return float(np.min(robust)) if len(robust) else median_z


def stamps_within_skew(stamp_a, stamp_b, max_skew):
    """图像-点云时间差检查（PM 冻结：超过 0.5s 拒绝本次定位）。

    stamp_a/stamp_b: 带 .sec/.nanosec 的时间戳（rclpy Time 或离线替身）。
    任一时间戳为零（未携带采集时间）视为不可考 → 返回 False（拒绝）。
    返回: True=时间差在 max_skew 内；False=超差或时间不可考。
    """
    if stamp_a is None or stamp_b is None:
        return False
    ta = float(stamp_a.sec) + float(stamp_a.nanosec) * 1e-9
    tb = float(stamp_b.sec) + float(stamp_b.nanosec) * 1e-9
    if ta <= 0.0 or tb <= 0.0:
        return False
    return abs(ta - tb) <= float(max_skew)
