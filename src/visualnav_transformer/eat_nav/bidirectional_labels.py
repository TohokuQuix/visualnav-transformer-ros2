"""XY + auxiliary yaw labels in the same initial body frame for both directions."""

from __future__ import annotations

import math

from .motion_vnext import motion_label_from_trace, _yaw_from_root


SCHEMA = 'eatnav_body_frame_delta_xy_yaw_2s_v1'


def motion_label_with_yaw(rows) -> dict:
    result = motion_label_from_trace(rows)
    result['schema'] = SCHEMA
    result['delta_body_yaw_rad'] = None
    if result['available']:
        first_yaw = _yaw_from_root(rows[0]['root_state'])
        last_yaw = _yaw_from_root(rows[39]['post_root_state'])
        difference = last_yaw - first_yaw
        result['delta_body_yaw_rad'] = math.atan2(math.sin(difference), math.cos(difference))
    return result
