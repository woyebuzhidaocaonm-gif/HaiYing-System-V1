"""TF 查询工具：world→base_footprint 回退链 + 位姿坐标换算。

正式链路（V9.2）中 world→base_footprint 动态 TF 由仿真组补齐；
冻结链（attitude_cmd）下位姿 frame=map（ENU 世界对齐）。本模块
提供统一入口，查询失败返回 None，由调用方按 tf_failure_policy 处理。
"""
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point


def lookup_world_to_frame(tf_lookup, world_frame, target_frame):
    """查 world→target_frame 的 TransformStamped；失败返回 None。

    tf_lookup: 可调用 tf_lookup(target_frame, source_frame) ->
        TransformStamped，失败抛异常或返回 None（单测可注入替身）。
    """
    try:
        t = tf_lookup(target_frame, world_frame)
        return t
    except Exception:
        return None


def lookup_base_fallback(tf_lookup, world_frame, arm_frame,
                         fallback_frames=('base_link', 'drone_base_link')):
    """world→arm_frame（默认 base_footprint）回退链。

    顺序: arm_frame 直达 → base_link → drone_base_link（旧仿真命名，
    恒等偏移近似）。任一成功即返回其 TransformStamped，全失败返回 None。
    """
    for frame in (arm_frame,) + tuple(fallback_frames):
        t = lookup_world_to_frame(tf_lookup, world_frame, frame)
        if t is not None:
            return t
    return None


def point_to_frame(point, transform):
    """把 PointStamped（world 系）换算到 transform 的目标系。

    point: PointStamped；transform: TransformStamped(world→目标系)。
    返回 (x, y, z) 元组。
    """
    stamped = PointStamped()
    stamped.header = point.header
    stamped.header.frame_id = transform.header.frame_id
    stamped.point = point.point
    out = do_transform_point(stamped, transform)
    return (out.point.x, out.point.y, out.point.z)


def pose_position(pose):
    """返回 PoseStamped 的位置 (x, y, z)，None 表示消息缺失。"""
    if pose is None:
        return None
    p = pose.pose.position
    return (p.x, p.y, p.z)
