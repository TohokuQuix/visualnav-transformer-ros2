"""Freeze candidate-free radial inventory inputs before any simulator launch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from visualnav_transformer.eat_nav.radial_inventory import build_pose_schedule, TERRAIN_ORDER, VALIDITY_LIMITS

ROOT_DEFAULT = Path('/tmp/eatnav-radial-terrain-inventory-20261007')
PRIOR = Path('/tmp/eatnav-radial-boundary-review-20261007')
REPO = Path(__file__).resolve().parents[1]
REMOTE = ROOT_DEFAULT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def prepare(root=ROOT_DEFAULT):
    root = Path(root)
    if root.exists():
        raise FileExistsError('new output root required: ' + str(root))
    prior = json.loads((PRIOR / 'review-artifact-sha-ledger.json').read_text())
    if len(prior['entries']) != 38:
        raise ValueError('prior radial artifact count changed')
    drift = [name for name, meta in prior['entries'].items() if sha(PRIOR / name) != meta['sha256']]
    if drift:
        raise ValueError('prior radial artifact SHA drift: ' + repr(drift))
    src = json.loads((PRIOR / 'source-provenance.json').read_text())
    if src['prior_ledger_mismatch_count'] or src['direct_input_mismatch_count'] or src['execution_source_mismatch_count']:
        raise ValueError('old source audit not clean')
    for path, meta in src['direct_input_checks'].items():
        if sha(path) != meta['expected']:
            raise ValueError('old direct input drift: ' + path)
    for meta in src['execution_source_checks'].values():
        if sha(meta['path']) != meta['expected']:
            raise ValueError('old execution source drift: ' + meta['path'])
    host_paths = {
        'eat_config': REPO / 'config/eat_nav.yaml',
        'radial_module': REPO / 'src/visualnav_transformer/eat_nav/radial_inventory.py',
        'sim_source': REPO / 'scripts/eat_nav_radial_inventory_sim.py',
        'rough_cfg': Path('/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py'),
        'eat_env_cfg': Path('/tmp/eatnav-isaac/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/eat_nav_env_cfg.py'),
        'terrain_generator': Path('/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/terrains/terrain_generator.py'),
        'checkpoint_env': Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml'),
        'rl_checkpoint': Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'),
        'follower_config_readonly': Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'),
    }
    expected_checkpoint = 'b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d'
    if sha(host_paths['rl_checkpoint']) != expected_checkpoint:
        raise ValueError('RL checkpoint drift')
    remote_paths = {
        'eat_config': str(REMOTE / 'inputs/eat_nav.yaml'),
        'radial_module': '/tmp/eatnav-train/src/visualnav_transformer/eat_nav/radial_inventory.py',
        'sim_source': str(REMOTE / 'inputs/eat_nav_radial_inventory_sim.py'),
        'rough_cfg': '/workspace/isaaclab_onix/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py',
        'eat_env_cfg': '/tmp/eatnav-extended/isaac-source/onix_isaac_lab/tasks/locomotion/velocity/config/onix/eat_nav_env_cfg.py',
        'terrain_generator': '/workspace/isaaclab/source/isaaclab/isaaclab/terrains/terrain_generator.py',
        'checkpoint_env': '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml',
        'rl_checkpoint': '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
        'follower_config_readonly': '/workspace/onix_dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml',
    }
    schedule = build_pose_schedule((20261007, 20261008))
    scene = {'num_rows': 16, 'num_cols': 7, 'num_envs': 112, 'terrain_order': TERRAIN_ORDER,
             'difficulty': .5, 'curriculum': False, 'radius_m': .30,
             'angles_deg': [22.5 * k for k in range(16)], 'initial_root_height_above_terrain_origin_m': .55,
             'settling_control_steps': 20, 'dem_frame_control_steps': [4, 9, 14, 19]}
    limits = {'gpu_device_total_cap_mib': 12288, 'anon_plus_shmem_cap_bytes': 10 * 1024**3,
              'host_mem_available_floor_bytes': 8 * 1024**3, 'total_runtime_cap_sec': 7200,
              'per_batch_startup_cap_sec': 720, 'heartbeat_cap_sec': 90,
              'storage_cap_bytes': 1_000_000_000, 'retry_count': 0,
              'max_pose_count': 224, 'cleanup_owned_only': True}
    freeze = {'schema': 'eatnav_radial_inventory_u0_v1', 'batch_seeds': [20261007, 20261008],
              'source_paths': remote_paths, 'source_sha256': {k: sha(v) for k, v in host_paths.items()},
              'sim_source_sha256': sha(host_paths['sim_source']),
              'prior_review_ledger_sha256': sha(PRIOR / 'review-artifact-sha-ledger.json'),
              'scene': scene, 'limits': limits, 'validity_limits': VALIDITY_LIMITS,
              'candidate_free': True, 'candidate_path_count': 0, 'external_zenoh_required': False,
              'bridge_follower_started': False, 'outcome_collection': False,
              'old_failed_attempts': 'retained in previous roots; excluded from this inventory'}
    root.mkdir(parents=True)
    write_once(root / 'input-freeze.json', freeze)
    write_once(root / 'pose-schedule.json', schedule)
    write_once(root / 'source-provenance.json', {'prior_review_root': str(PRIOR),
        'prior_review_ledger_sha256': sha(PRIOR / 'review-artifact-sha-ledger.json'),
        'prior_28_groups': src['existing_state_groups'], 'prior_252_traces': src['existing_protocol_valid_traces'],
        'prior_attestation_count': src['prior_ledger_count'], 'prior_live_checks': src['direct_input_count'] + src['execution_source_count'],
        'source_sha256': freeze['source_sha256'], 'git_base': '35026a9124cce58a1f5d752ba9962a4a555aa2e1',
        'branch': 'exp/eatnav-radial-terrain-inventory'})
    print(root)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    prepare(ap.parse_args().root)
