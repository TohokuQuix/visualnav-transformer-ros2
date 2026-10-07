"""Frozen first-seed radial inventory and strict per-shard completeness gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .radial_inventory import build_pose_schedule

SHARD_SIZE = 5
INTERLEAVE_STRIDE = 15
POSE_COUNT = 112


def build_inventory(batch_seed: int = 20261007) -> dict:
    if batch_seed != 20261007:
        raise ValueError('first-seed inventory is frozen at 20261007')
    poses = build_pose_schedule([batch_seed])
    ordered = [poses[(i * INTERLEAVE_STRIDE) % POSE_COUNT] for i in range(POSE_COUNT)]
    shards = [{'shard_id': f'shard-{j:02d}', 'order': j,
               'poses': ordered[j*SHARD_SIZE:(j+1)*SHARD_SIZE]}
              for j in range((POSE_COUNT+SHARD_SIZE-1)//SHARD_SIZE)]
    result = {'schema':'eatnav_radial_112_sharded_v1','batch_seed':batch_seed,
              'difficulty':0.5,'scene_envs':112,'radius_m':0.3,
              'direction_count':16,'shard_size_max':SHARD_SIZE,
              'interleave_stride':INTERLEAVE_STRIDE,'poses':poses,'shards':shards}
    validate_inventory(result)
    return result


def validate_inventory(inventory: dict) -> None:
    expected = build_pose_schedule([20261007])
    if (inventory.get('batch_seed') != 20261007 or inventory.get('difficulty') != 0.5 or
        inventory.get('scene_envs') != 112 or inventory.get('radius_m') != 0.3 or
        inventory.get('direction_count') != 16 or inventory.get('shard_size_max') != 5 or
        inventory.get('interleave_stride') != 15 or inventory.get('poses') != expected):
        raise ValueError('frozen 112-pose inventory metadata or geometry drift')
    shards = inventory.get('shards')
    if not isinstance(shards, list) or len(shards) != 23:
        raise ValueError('expected 23 shards')
    ordered = [expected[(i*15)%112] for i in range(112)]
    flatten = []
    for index, shard in enumerate(shards):
        poses = shard.get('poses', [])
        if (shard.get('shard_id') != f'shard-{index:02d}' or shard.get('order') != index or
            not 1 <= len(poses) <= 5):
            raise ValueError('shard identity or size drift')
        flatten.extend(poses)
    if flatten != ordered or len({p['pose_id'] for p in flatten}) != 112:
        raise ValueError('shard assignment duplicated, missing, or reordered poses')


def sensor_paths_for_env(template: list[str], env_index: int) -> list[str]:
    if not 0 <= env_index < 112 or not template:
        raise ValueError('invalid environment or sensor template')
    result = []
    for path in template:
        match = re.fullmatch(r'(/World/envs/)env_\d+(/onix_chassis/[^/]+)', path)
        if not match:
            raise ValueError('sensor template must be an Onix chassis path')
        result.append(f'{match.group(1)}env_{env_index}{match.group(2)}')
    if len(set(result)) != len(result):
        raise ValueError('duplicate expanded sensor path')
    return result


def select_shard_rows(manifest_poses: list[dict], schedule: list[dict]) -> list[dict]:
    if not 1 <= len(manifest_poses) <= 5:
        raise ValueError('a shard must contain one to five poses')
    by_id = {row['pose_id']:row for row in schedule}
    if len(by_id) != 112:
        raise ValueError('expected the frozen 112-pose schedule')
    ids = [pose['pose_id'] for pose in manifest_poses]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate selected pose')
    try:
        selected = [by_id[pose_id] for pose_id in ids]
    except KeyError as exc:
        raise ValueError('unknown selected pose') from exc
    for expected, actual in zip(manifest_poses,selected):
        if any(expected.get(key) != value for key,value in actual.items()):
            raise ValueError('selected pose geometry or seed drift')
    return selected


def verify_shard_artifacts(run: Path, pose_ids: list[str]) -> dict:
    run = Path(run)
    if not 1 <= len(pose_ids) <= 5 or len(set(pose_ids)) != len(pose_ids):
        raise ValueError('invalid pose set')
    for pose_id in pose_ids:
        stream = run/'trace'/f'{pose_id}.jsonl'
        data = stream.read_bytes()
        if not data.endswith(b'\n'):
            raise ValueError('interrupted step stream')
        rows = [json.loads(line) for line in data.splitlines()]
        if len(rows) != 200 or [row['physics_step'] for row in rows] != list(range(1,201)):
            raise ValueError('missing, repeated, or out-of-order physics step')
        if any((not row['contact']['bounds_valid'] or row['contact']['buffer_full_or_truncation_suspected'] or
                not row['contact']['finite'] or row['contact']['buffer_capacity'] != 4096) for row in rows):
            raise ValueError('contact integrity gate failed')
        for index, expected_step in enumerate((50,100,150,200)):
            path = run/'dem'/pose_id/f'frame-{index:02d}.json'
            if not path.exists():
                raise ValueError('DEM history incomplete')
            frame = json.loads(path.read_text())
            if frame['physics_step'] != expected_step or frame['shape'] != [2,20,20] or not frame['finite']:
                raise ValueError('DEM history incomplete')
        for phase in ('prephysics','settled_step_200'):
            path = run/'actor'/pose_id/f'{phase}.json'
            if not path.exists():
                raise ValueError('Actor observation incomplete')
            actor = json.loads(path.read_text())
            if actor['shape'] != [22] or not actor['finite'] or len(actor['field_order']) != 22:
                raise ValueError('Actor observation incomplete')
    for raw in (run/'raw').rglob('*.npz'):
        sidecar = json.loads(raw.with_suffix('.json').read_text())
        if hashlib.sha256(raw.read_bytes()).hexdigest() != sidecar['raw_sha256']:
            raise ValueError('raw contact SHA mismatch')
    if (run/'partial').exists():
        raise ValueError('cumulative partial snapshots forbidden')
    return {'pose_count':len(pose_ids),'physics_steps':200,'raw_count':len(list((run/'raw').rglob('*.npz')))}


def numerical_exceedance_stop(exceeded: list[bool]) -> bool:
    """Prospective 'many poses' stop: strict majority in a completed shard."""
    if not exceeded:
        raise ValueError('empty shard')
    return sum(exceeded) > len(exceeded)/2


def settled_exceedance_reasons(rows: list[dict], dem_fractions: list[float]) -> list[str]:
    if len(rows) != 200 or len(dem_fractions) != 4:
        raise ValueError('settled measurement incomplete')
    limits = {'xy_drift_m':0.10,'h1_signed_rad':0.10,
              'linear_speed_mps':0.10,'angular_speed_radps':0.20}
    reasons = []
    for key,limit in limits.items():
        if any(abs(row['state'][key]) > limit for row in rows[160:]):
            reasons.append(key)
    if min(dem_fractions) < 0.70:
        reasons.append('dem_valid_fraction')
    return reasons
