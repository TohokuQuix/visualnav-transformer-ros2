"""Materialize the already sealed, outcome-blind R3 transition reserve as one shard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/tmp/eatnav-r3-transition-reserve-20261008')
OLD = Path('/tmp/eatnav-r3-settling-pilot-20261008')
WORKTREE = Path('/tmp/eatnav-vnt-r3-transition-reserve')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('frozen artifact already exists: ' + str(path))
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def selected_rows(manifest, original_primary):
    primary_ids = {x['pose_id'] for x in original_primary}
    items = manifest['selected']
    if len(items) != 2 or [x['selection_rank_in_r3'] for x in items] != [3, 4]:
        raise ValueError('frozen reserve rank changed')
    if len({x['source_family'] for x in items}) != 2:
        raise ValueError('reserve source families changed')
    rows = []
    from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER, terrain_seed
    for item in items:
        if item['pose_id'] in primary_ids or item['classification'] != 'transition':
            raise ValueError('reserve includes primary or non-transition pose')
        col = TERRAIN_ORDER.index(item['terrain'])
        angle = item['direction_index']
        if item['terrain_seed'] != terrain_seed(20261007, col):
            raise ValueError('terrain seed drift')
        if item['env_id'] != angle * len(TERRAIN_ORDER) + col:
            raise ValueError('environment index drift')
        rows.append(dict(pose_id=item['pose_id'], batch_seed=20261007,
            terrain=item['terrain'], terrain_seed=item['terrain_seed'],
            source_family=item['source_family'], row=angle, column=col,
            env_index=item['env_id'], angle_index=angle, theta_rad=item['theta_rad'],
            radius_m=item['radius_m'], requested_xy_yaw=item['requested_xy_yaw'],
            virtual_dem_sha256=item['dem_sha256'], virtual_class='transition',
            virtual_interaction_onset_m=item['virtual_interaction_onset_m'],
            rank=item['selection_rank_in_r3']))
    return rows


def main():
    ledger = json.loads((ROOT / 'freeze-sha.json').read_text())
    if (sha(ROOT / 'reserve-preaudit-selection.json') != ledger['reserve_preaudit_sha256'] or
        sha(ROOT / 'reserve-manifest.json') != ledger['reserve_manifest_sha256']):
        raise RuntimeError('pre-audit reserve manifest SHA drift')
    manifest = json.loads((ROOT / 'reserve-manifest.json').read_text())
    if sha(OLD / 'reserve-rank-freeze.json') != manifest['reserve_input_sha256']:
        raise RuntimeError('original reserve SHA drift')
    for path, expected in manifest['offline_scan_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('offline scan SHA drift: ' + path)
    old_primary = json.loads((OLD / 'primary-manifest.json').read_text())['poses']
    rows = selected_rows(manifest, old_primary)
    from visualnav_transformer.eat_nav.r3_anchor_contract import requested_world
    for row, item in zip(rows, manifest['selected']):
        requested_world(row, item['env_origin_xyz'])
    old_freeze = json.loads((OLD / 'shard-A/input-freeze.json').read_text())
    shard = ROOT / 'shard-reserve'
    shard.mkdir(exist_ok=False)
    write_new(ROOT / 'primary-manifest.json', {'schema':'eatnav_r3_primary_v1',
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip(),
        'reserve_manifest_sha256': ledger['reserve_manifest_sha256'], 'poses':rows})
    write_new(shard / 'pose-manifest.json', {'schema':'eatnav_r3_shard_poses_v1',
        'shard_id':'shard-reserve','poses':rows})
    freeze = json.loads(json.dumps(old_freeze))
    freeze.update(schema='eatnav_r3_transition_reserve_freeze_v1', shard_id='shard-reserve',
        new_top_root=str(ROOT), inventory_root=str(ROOT), expected_pose_count=len(rows),
        pose_manifest_sha256=sha(shard/'pose-manifest.json'),
        primary_manifest_path=str(ROOT/'primary-manifest.json'),
        primary_manifest_sha256=sha(ROOT/'primary-manifest.json'),
        reserve_manifest_path=str(ROOT/'reserve-manifest.json'),
        reserve_manifest_sha256=ledger['reserve_manifest_sha256'],
        sim_source_sha256=sha(WORKTREE/'scripts/eat_nav_r3_settling_sim.py'),
        remediation_cycle=0, remediation_cycles_allowed=2,
        no_further_automatic_retry=True)
    freeze['contact_contract']['views'] = len(rows)
    for key in ('sim_source','r3_anchor_contract','radial_lean_log',
                'radial_shard_inventory','radial_termination_capture'):
        old_path=Path(freeze['source_paths'][key])
        freeze['source_paths'][key]=str(shard/old_path.relative_to(OLD/'shard-A'))
    freeze['source_sha256']['sim_source']=freeze['sim_source_sha256']
    for key in ('r3_anchor_contract','radial_lean_log','radial_shard_inventory','radial_termination_capture'):
        freeze['source_sha256'][key]=sha(WORKTREE/'src/visualnav_transformer/eat_nav'/ (key+'.py'))
    write_new(shard/'input-freeze.json',freeze)
    pythonpath=':'.join((str(shard/'src'),'/tmp/eatnav-train/src',
        '/tmp/eatnav-extended/isaac-source','/workspace/isaaclab/source/isaaclab',
        '/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks',
        '/workspace/isaaclab_onix/source/onix_isaac_lab'))
    command=['docker','exec','-e','PYTHONPATH='+pythonpath,'isaac-lab-onix',
        '/isaac-sim/python.sh',str(shard/'inputs/eat_nav_r3_settling_sim.py'),
        '--headless','--freeze',str(shard/'input-freeze.json'),
        '--pose-manifest',str(shard/'pose-manifest.json'),'--batch-seed','20261007',
        '--output',str(shard/'run'),'--mode','full']
    source_files=[WORKTREE/'scripts/eat_nav_r3_settling_sim.py',
        WORKTREE/'scripts/eat_nav_r3_transition_run.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/r3_anchor_contract.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/radial_shard_inventory.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/radial_termination_capture.py']
    write_new(shard/'prelaunch.json', {'schema':'eatnav_r3_transition_prelaunch_v1',
        'worktree':str(WORKTREE), 'shard_id':'shard-reserve',
        'source_and_test_sha256':{str(p):sha(p) for p in source_files},
        'freeze_sha256':sha(shard/'input-freeze.json'),
        'pose_manifest_sha256':sha(shard/'pose-manifest.json'),
        'reserve_manifest_sha256':ledger['reserve_manifest_sha256'],
        'command':command,'candidate_free':True})
    write_new(ROOT/'preflight.json',{'schema':'eatnav_r3_transition_preflight_v1',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip(),
        'worktree':str(WORKTREE),'reserve_manifest_sha256':ledger['reserve_manifest_sha256'],
        'primary_manifest_sha256':sha(ROOT/'primary-manifest.json'),
        'pose_manifest_sha256':sha(shard/'pose-manifest.json'),
        'freeze_sha256':sha(shard/'input-freeze.json'),
        'terrain_mesh_sha256':freeze['terrain_mesh_sha256'],
        'source_and_test_sha256':{str(p):sha(p) for p in source_files}})
    print(json.dumps({'pose_ids':[x['pose_id'] for x in rows],
        'reserve_manifest_sha256':ledger['reserve_manifest_sha256'],
        'freeze_sha256':sha(shard/'input-freeze.json')},indent=2))

if __name__ == '__main__':
    main()
