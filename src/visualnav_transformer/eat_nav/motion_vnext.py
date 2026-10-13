"""Goal-independent two-second displacement label and geometric Goal score."""

from __future__ import annotations

import math

import numpy as np


SCHEMA = 'eatnav_body_frame_delta_xy_2s_v1'
STEPS = 40
STEP_NS = 50_000_000


def _yaw_from_root(root) -> float:
    w, x, y, z = map(float, root[3:7])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def motion_label_from_trace(rows) -> dict:
    """Use start yaw and step-39 post-root; preserve short/terminal traces."""
    if not rows:
        raise ValueError('empty motion trace')
    observed = rows[:STEPS]
    first_stamp = int(observed[0]['logical_sim_time_ns'])
    if any(row['control_step'] != i or
           int(row['logical_sim_time_ns']) != first_stamp + i * STEP_NS
           for i, row in enumerate(observed)):
        raise ValueError('motion trace step/stamp discontinuity')
    terminal = next((row['terminal'] for row in observed if row.get('terminal') is not None), None)
    reason = terminal.get('termination_reason', 'unknown') if terminal is not None else None
    result = {'schema': SCHEMA, 'available': False, 'delta_body_xy_m': None,
              'window_steps': len(observed), 'last_control_step': int(observed[-1]['control_step']),
              'termination_reason': reason}
    if len(observed) != STEPS or terminal is not None:
        return result
    start = np.asarray(observed[0]['root_state'][:2], dtype=np.float64)
    end = np.asarray(observed[-1]['post_root_state'][:2], dtype=np.float64)
    yaw0 = _yaw_from_root(observed[0]['root_state'])
    if not np.isfinite(start).all() or not np.isfinite(end).all() or not math.isfinite(yaw0):
        raise ValueError('nonfinite motion root/yaw')
    delta = end - start
    c, s = math.cos(yaw0), math.sin(yaw0)
    body = np.asarray([c * delta[0] + s * delta[1],
                       -s * delta[0] + c * delta[1]], dtype=np.float64)
    result.update(available=True, delta_body_xy_m=body.tolist())
    return result


def goal_progress_scores(delta_body_xy, goal_body_xy, candidate_mask=None):
    """Score fixed motion predictions for a Goal without calling the model."""
    delta = np.asarray(delta_body_xy, dtype=np.float64)
    goal = np.asarray(goal_body_xy, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 2 or goal.shape != (2,) or not np.isfinite(
            delta).all() or not np.isfinite(goal).all():
        raise ValueError('Goal/motion shape or nonfinite')
    scores = np.linalg.norm(goal) - np.linalg.norm(goal[None, :] - delta, axis=1)
    if candidate_mask is not None:
        mask = np.asarray(candidate_mask)
        if mask.shape != (len(delta),) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError('candidate mask shape/dtype/empty')
        scores[~mask] = -np.inf
    return scores
