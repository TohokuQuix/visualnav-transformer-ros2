"""Goal scoring on measured map pose at the inference stamp."""

import math
import numpy as np


def score_global_goal(predicted_xy, robot_pose_map, goal_map, mask):
    predicted = np.asarray(predicted_xy, dtype=np.float32)
    pose = np.asarray(robot_pose_map, dtype=np.float64)
    goal = np.asarray(goal_map, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.ndim != 2 or predicted.shape[1] != 2 or \
            pose.shape != (3,) or goal.shape != (2,) or \
            valid.shape != (len(predicted),) or not valid.any() or \
            not np.isfinite(predicted).all() or not np.isfinite(pose).all() or \
            not np.isfinite(goal).all():
        raise ValueError('global goal scoring shape/finite/mask')
    delta = goal - pose[:2]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    goal_body = np.asarray([c*delta[0]+s*delta[1],
                            -s*delta[0]+c*delta[1]], dtype=np.float32)
    if np.linalg.norm(goal_body) <= 0:
        raise ValueError('global goal already reached')
    score = (np.linalg.norm(goal_body) -
             np.linalg.norm(goal_body[None]-predicted, axis=1)).astype(np.float32)
    score[~valid] = -np.inf
    return {'index': int(np.argmax(score)), 'scores_m': score,
            'goal_body_xy_m': goal_body}
