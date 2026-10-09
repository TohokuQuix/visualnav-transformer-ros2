"""Deterministic 28-slot layout inside a 16x7 Isaac scene."""

from __future__ import annotations

import math
import re


def build_plan(manifest: dict, source_contract: dict) -> dict:
    rows = manifest['rows']
    anchors = source_contract['anchors']
    paths = {item['path_sha256']: item for item in source_contract['paths']}
    if len(rows) != 28 or len(anchors) != 4 or len(paths) != 5:
        raise ValueError('frozen P1 28/4/5 input count changed')
    targets = []
    for group, anchor in enumerate(anchors):
        match = re.search(r'-r([0-9]+\.[0-9]+)-', anchor['pose_id'])
        if not match:
            raise ValueError('source anchor radius missing')
        radius = float(match.group(1))
        x, y, theta = anchor['requested_xy_yaw']
        local = [radius * math.cos(theta), radius * math.sin(theta)]
        origin = [x-local[0], y-local[1]]
        subset = rows[group*7:(group+1)*7]
        if len(subset) != 7 or any(row['anchor_id'] != anchor['anchor_id'] for row in subset):
            raise ValueError('P1 manifest order/group changed')
        for column, row in enumerate(subset):
            path_sha = row['candidate_path_sha256']
            if path_sha not in paths:
                raise ValueError('unrecognized candidate Path')
            targets.append({
                'physical_run_id': row['physical_run_id'],
                'anchor_id': anchor['anchor_id'],
                'source_pose_id': anchor['pose_id'],
                'source_env_index': anchor['environment_index'],
                'source_tile_origin_xy': origin,
                'source_anchor_local_xy': local,
                'source_anchor_yaw': theta,
                'target_env_index': group*7+column,
                'target_tile_row': group,
                'target_tile_column': column,
                'terrain_type': anchor['terrain'],
                'terrain_seed': anchor['terrain_seed'],
                'difficulty': 0.5,
                'candidate_path_sha256': path_sha,
                'base_path': paths[path_sha]['waypoints'],
            })
    if len({r['target_env_index'] for r in targets}) != 28:
        raise ValueError('duplicate target slot')
    return {'schema': 'p1_batch_112_scene_plan_v1', 'num_envs': 112,
            'terrain_rows': 16, 'terrain_columns': 7, 'targets': targets}
