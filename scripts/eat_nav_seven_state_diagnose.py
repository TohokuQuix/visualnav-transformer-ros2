"""Evidence-only diagnosis for follower command startup attempts.

A ROS log cannot prove a callback did not run. Counts inferred from log lines
are lower bounds; command publication requires bridge/topic instrumentation.
"""
from __future__ import annotations

import math
import re

_TIMESTAMP = re.compile(r'^\[[^]]+\] \[([0-9]+(?:\.[0-9]+)?)\]')


def classify_command_delivery(*, goal_distance_m, tolerance_m, before, after, commands, goal_status):
    if not (len(before) == len(after) == len(commands) == len(goal_status)) or not before:
        raise ValueError('branch arrays must have equal nonzero length')
    if not all(isinstance(x, int) and x >= 0 for x in [*before, *after]):
        raise ValueError('sequence counters must be nonnegative integers')
    if any(new < old for old, new in zip(before, after)):
        raise ValueError('sequence counter went backwards')
    if not all(math.isfinite(x) for x in (goal_distance_m, tolerance_m)) or tolerance_m < 0:
        raise ValueError('invalid goal distance or tolerance')
    fresh = [new > old for old, new in zip(before, after)]
    if all(fresh):
        if any(cmd is None or len(cmd) != 3 or not all(math.isfinite(v) for v in cmd) for cmd in commands):
            raise ValueError('fresh sequence without finite 3D command')
        delivery = 'fresh_zero' if all(all(v == 0 for v in cmd) for cmd in commands) else 'fresh_nonzero'
    elif any(fresh):
        delivery = 'partial_fresh'
    elif all(cmd is None for cmd in commands):
        delivery = 'no_message'
    else:
        delivery = 'stale_only'
    near_goal = goal_distance_m <= tolerance_m
    if near_goal and delivery == 'fresh_zero' and all(goal_status):
        classification = 'no_action_required'
    elif near_goal and delivery in ('no_message', 'stale_only', 'partial_fresh'):
        classification = 'pipeline_unresolved'
    elif delivery in ('no_message', 'stale_only', 'partial_fresh'):
        classification = 'infrastructure_diagnosis_required'
    else:
        classification = 'valid_command_delivery'
    return {'goal_distance_m': float(goal_distance_m), 'goal_tolerance_m': float(tolerance_m),
            'near_goal': near_goal, 'delivery': delivery, 'classification': classification,
            'fresh_branches': sum(fresh), 'branch_count': len(fresh)}


def summarize_follower_log(text: str):
    first = {}
    counts = {'started': 0, 'path_callback': 0, 'tf_failure': 0, 'control_debug': 0}
    for line in text.splitlines():
        stamp = _TIMESTAMP.match(line)
        if stamp is None:
            continue
        t = float(stamp.group(1))
        checks = {'started': 'Started:', 'path_callback': 'Activated default path',
                  'tf_failure': 'Failed to lookup robot pose', 'control_debug': 'control:'}
        for name, needle in checks.items():
            if needle in line:
                counts[name] += 1
                first.setdefault(name, t)
    return {'started_count': counts['started'],
            'path_callback_count_lower_bound': counts['path_callback'],
            'tf_failure_count': counts['tf_failure'],
            'control_debug_count': counts['control_debug'],
            'control_callback_count_lower_bound': counts['tf_failure'] + counts['control_debug'],
            'first_started_s': first.get('started'),
            'first_path_callback_s': first.get('path_callback'),
            'first_tf_failure_s': first.get('tf_failure'),
            'first_control_debug_s': first.get('control_debug'),
            'command_publish_count': None,
            'timer_callback_exact_count': None}


def map_path_goal_geometry(map_path, relative_goal, tolerance_m):
    if len(map_path) < 2 or len(relative_goal) != 2:
        raise ValueError('Path or relative goal shape mismatch')
    start, endpoint = map_path[0], map_path[-1]
    if any(len(point) != 2 or not all(math.isfinite(float(v)) for v in point) for point in (start, endpoint)):
        raise ValueError('Path coordinates must be finite 2D points')
    published = math.dist(start, endpoint)
    relative = math.hypot(*relative_goal)
    return {'robot_pose_map_xy': list(start), 'path_endpoint_map_xy': list(endpoint),
            'relative_goal_base_xy': list(relative_goal),
            'published_goal_distance_m': published, 'relative_goal_distance_m': relative,
            'goal_tolerance_m': tolerance_m, 'within_goal_tolerance': published <= tolerance_m}
