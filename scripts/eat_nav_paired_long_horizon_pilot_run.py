"""Run the frozen passive paired long-horizon pilot in manifest order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from eat_nav_motion_14_collection_host import ACTOR_ENV, SHARED_RL, execute, sha


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO/'docs/eat-nav/paired-long-horizon-pilot-spec-v0.2.md'
EAT_CONFIG = REPO/'config/eat_nav.yaml'
OUTPUT_ROOT = Path('/home/ubuntu/Projects/onix_isaac_lab/'
                   '.eatnav-paired-long-horizon-pilot-v0.2-20261019-01')
CONTROL_STEPS = 200
REPLAN_EVERY_STEPS = 40
CONDITION = 'front_folded'
REPEAT = 0
NUM_ENVS = 4
CANDIDATES = (
    {'candidate_id': 'forward_direct', 'direction': 1, 'angle': 0},
    {'candidate_id': 'forward_-30', 'direction': 1, 'angle': -30},
    {'candidate_id': 'forward_+30', 'direction': 1, 'angle': 30},
    {'candidate_id': 'reverse_direct', 'direction': -1, 'angle': 0},
)
TERRAIN_LEVELS = (
    ('k_rail_step_field', 1), ('k_rail_step_field', 4),
    ('pyramid_stairs', 1), ('pyramid_stairs', 4),
)


def terrain_seed(terrain_type: str, level: int, replicate: int) -> int:
    token = f'eatnav-pilot-v0.2|{terrain_type}|{level}|{replicate}'.encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'big')


def build_state_groups() -> list[dict]:
    groups = []
    for terrain_type, level in TERRAIN_LEVELS:
        for replicate in range(3):
            seed = terrain_seed(terrain_type, level, replicate)
            groups.append({
                'state_group_id': f'{terrain_type}-level{level}-rep{replicate}',
                'terrain_type': terrain_type,
                'terrain_level': level,
                'replicate': replicate,
                'terrain_seed': seed,
                'env_tile_count': NUM_ENVS,
            })
    seeds = [group['terrain_seed'] for group in groups]
    if len(groups) != 12 or len(seeds) != len(set(seeds)):
        raise ValueError('pilot state group assignment is not unique and balanced')
    return groups


def build_schedule(groups: list[dict]) -> list[dict]:
    schedule = []
    for group in groups:
        common = {key: group[key] for key in
                  ('state_group_id', 'terrain_type', 'terrain_level', 'terrain_seed')}
        common['failure_induction'] = False
        schedule.append({**common, 'stage': 'prepath', 'candidate_id': None,
                         'direction': 1, 'angle': 0})
        for candidate in CANDIDATES:
            schedule.append({**common, 'stage': 'candidate', **candidate})
    if len(schedule) != 60:
        raise ValueError('pilot schedule must contain 12 baselines and 48 candidates')
    return schedule


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    sources = {
        'coordinator': Path(__file__),
        'host_runner': REPO/'scripts/eat_nav_motion_14_collection_host.py',
        'sim_runner': REPO/'scripts/eat_nav_motion_14_collection_sim.py',
        'terrain': REPO/'src/visualnav_transformer/eat_nav/demo_terrain.py',
        'environment': REPO/'src/visualnav_transformer/eat_nav/demo_isaac_env_cfg.py',
        'follower': REPO/'src/visualnav_transformer/eat_nav/p1_follower_core.cpp',
        'labels': REPO/'src/visualnav_transformer/eat_nav/long_horizon_labels.py',
        'config': EAT_CONFIG,
        'protocol': PROTOCOL,
        'actor': SHARED_RL,
        'actor_env': ACTOR_ENV,
    }
    return {name: sha(path) for name, path in sources.items()}


def row_root(root: Path, index: int, row: dict) -> Path:
    suffix = 'prepath' if row['stage'] == 'prepath' else row['candidate_id']
    return root/f'{index:03d}-{row["state_group_id"]}-{suffix}'


def run(root: Path = OUTPUT_ROOT) -> Path:
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f'new pilot output root required: {root}')
    if not SHARED_RL.is_file() or not ACTOR_ENV.is_file() or not PROTOCOL.is_file():
        raise FileNotFoundError('pilot checkpoint, Actor env YAML, or protocol missing')
    groups = build_state_groups()
    schedule = build_schedule(groups)
    root.mkdir(parents=True)
    provenance = {
        'schema': 'eatnav_passive_paired_long_horizon_pilot_v0.2',
        'created_unix_ns': time.time_ns(),
        'source_git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_sha256': source_hashes(),
        'state_groups': groups,
        'candidates': list(CANDIDATES),
        'control_hz': 20,
        'control_steps': CONTROL_STEPS,
        'replan_every_steps': REPLAN_EVERY_STEPS,
        'candidate_duration_sec': CONTROL_STEPS/20,
        'candidate_cycle_sec': REPLAN_EVERY_STEPS/20,
        'condition': CONDITION,
        'repeat': REPEAT,
        'num_envs_per_process': NUM_ENVS,
        'process_count': len(schedule),
        'candidate_process_count': sum(row['stage'] == 'candidate' for row in schedule),
        'candidate_trajectory_count': sum(row['stage'] == 'candidate' for row in schedule)*NUM_ENVS,
        'failure_induction': False,
        'event_policy': 'passive_observation_only; no forced fall/stuck actions or resets',
    }
    atomic_json(root/'pilot-manifest.json', provenance)
    atomic_json(root/'assignment.json', {'schema': provenance['schema'], 'rows': schedule})
    prepaths: dict[str, Path] = {}
    completed = 0
    started = time.monotonic()
    atomic_json(root/'status.json', {'state': 'running', 'completed_processes': 0,
                                     'total_processes': len(schedule)})
    for index, row in enumerate(schedule):
        attempt = row_root(root, index, row)
        reference = prepaths.get(row['state_group_id'])
        atomic_json(root/'current.json', {'index': index, 'row': row,
                                          'output_root': str(attempt),
                                          'started_unix_ns': time.time_ns()})
        try:
            execute(
                attempt,
                direction=row['direction'],
                angle=row['angle'],
                seed=row['terrain_seed'],
                terrain_type=row['terrain_type'],
                terrain_level=row['terrain_level'],
                condition=CONDITION,
                repeat=REPEAT,
                protocol=PROTOCOL,
                reference=reference,
                prepath_only=row['stage'] == 'prepath',
                rl_checkpoint=SHARED_RL,
                actor_env_yaml=ACTOR_ENV,
                control_steps=CONTROL_STEPS,
                replan_every_steps=REPLAN_EVERY_STEPS,
            )
            decision = json.loads((attempt/'decision.json').read_text())
            if decision.get('pass') is not True:
                raise RuntimeError(decision.get('failure') or 'host decision failed')
            if row['stage'] == 'prepath':
                prepaths[row['state_group_id']] = attempt/'sim-output'
            completed += 1
            atomic_json(root/'status.json', {
                'state': 'running', 'completed_processes': completed,
                'total_processes': len(schedule), 'last_completed': row,
                'elapsed_wall_sec': time.monotonic()-started,
            })
        except Exception as exc:
            atomic_json(root/'status.json', {
                'state': 'stopped', 'completed_processes': completed,
                'total_processes': len(schedule), 'failed_row': row,
                'failure': f'{type(exc).__name__}: {exc}',
                'elapsed_wall_sec': time.monotonic()-started,
            })
            raise
    atomic_json(root/'status.json', {
        'state': 'completed', 'completed_processes': completed,
        'total_processes': len(schedule), 'elapsed_wall_sec': time.monotonic()-started,
    })
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(run(args.root))