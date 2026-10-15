"""Body-frame position and terminal-heading score for a fixed candidate bank."""

from __future__ import annotations

import numpy as np


def score_body_goal(xy, delta_yaw, goal_body_xy, heading_weight_m,
                    *, goal_radius_m=0.05, mask=None):
    xy = np.asarray(xy, dtype=np.float64)
    yaw = np.asarray(delta_yaw, dtype=np.float64)
    goal = np.asarray(goal_body_xy, dtype=np.float64)
    valid = np.ones(len(xy), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    weight = float(heading_weight_m)
    radius = float(goal_radius_m)
    if xy.ndim != 2 or xy.shape[1:] != (2,) or yaw.shape != (len(xy),) or \
            goal.shape != (2,) or valid.shape != (len(xy),) or not valid.any() or \
            not np.isfinite(xy).all() or not np.isfinite(yaw).all() or \
            not np.isfinite(goal).all() or not np.isfinite((weight, radius)).all() or \
            weight < 0 or radius <= 0 or np.linalg.norm(goal) <= 0:
        raise ValueError('heading score shape/finite/goal contract')
    residual = goal[None, :] - xy
    distance = np.linalg.norm(residual, axis=1)
    position = np.linalg.norm(goal) - distance
    heading = np.zeros(len(xy), dtype=np.float64)
    outside = distance > radius
    if np.linalg.norm(goal) > radius:
        # Forward and reverse arrival are equally acceptable. The heading
        # penalty is largest at 90 degrees and recedes toward 180 degrees.
        initial_alignment = abs(goal[0] / np.linalg.norm(goal))
        target_bearing = np.arctan2(residual[outside, 1], residual[outside, 0])
        heading[outside] = np.abs(np.cos(target_bearing - yaw[outside])) - initial_alignment
    total = position + weight * heading
    total[~valid] = -np.inf
    return {'index': int(np.argmax(total)), 'position_score_m': position,
            'heading_score': heading, 'weighted_heading_score_m': weight * heading,
            'total_score_m': total, 'goal_reached_neighborhood': ~outside}
