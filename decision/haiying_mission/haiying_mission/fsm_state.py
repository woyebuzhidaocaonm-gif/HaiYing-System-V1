"""六态定义与目标样本校验（与 docs/FROZEN_CONTROL_CHAIN.md 冻结六态一致）"""
import math

SEARCHING = 'SEARCHING'
TARGET_FOUND = 'TARGET_FOUND'
APPROACHING = 'APPROACHING'
BRUSHING = 'BRUSHING'
RETURNING = 'RETURNING'
ERROR = 'ERROR'

# 链路层（attitude_cmd_node）视为正常、可解除 FSM HOLD 的状态集
NORMAL_STATES = (SEARCHING, TARGET_FOUND, APPROACHING, BRUSHING, RETURNING)
ALL_STATES = NORMAL_STATES + (ERROR,)


def target_is_valid(point, max_target_pos):
    """目标样本校验（接口底线防御，视觉侧契约是"丢失不发、不发 NaN"，
    本节点仍独立校验，绝不把脏数据喂给状态机）。

    point: geometry_msgs/Point（或其离线替身），None 视为无效。
    max_target_pos: 单轴绝对值上限（米），<=0 表示不限。
    返回 False 时调用方必须：丢弃样本、且不刷新看门狗计时。
    """
    if point is None:
        return False
    x, y, z = point.x, point.y, point.z
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return False
    if max_target_pos > 0.0:
        if abs(x) > max_target_pos or abs(y) > max_target_pos \
                or abs(z) > max_target_pos:
            return False
    return True
