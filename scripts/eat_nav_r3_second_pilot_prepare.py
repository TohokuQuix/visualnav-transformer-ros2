"""Freeze two candidate-free second-seed settling shards after offline preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/tmp/eatnav-r3-second-seed-20261008')
WORKTREE=Path('/tmp/eatnav-vnt-r3-second-seed')
OLD=Path('/tmp/eatnav-r3-settling-pilot-20261008')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.anchor_placement_rules import CLASS_ORDER


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('frozen artifact already exists: '+str(path))
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    selected=ROOT/'selection';pilot_path=selected/'pilot-manifest.json'
    pilot=json.loads(pilot_path.read_text());ledger=json.loads((selected/'selection-sha.json').read_text())
    seed=json.loads((ROOT/'seed-freeze.json').read_text());scan=json.loads((ROOT/'scan-v1/scan-summary.json').read_text())
    rows=pilot['poses']
    if (sha(pilot_path)!=ledger['pilot_manifest_sha256'] or
        pilot['seed_freeze_sha256']!=sha(ROOT/'seed-freeze.json') or
        pilot['robustness_freeze_sha256']!=sha(ROOT/'robustness-freeze.json') or
        pilot['selection_uses_robustness'] is not False or
        pilot['r3_source_sha256']!='60da9ea1a4a7d6cfdde812ac49bc09561518710f7127d1d8cf86111b28ecb3de' or
        pilot['classifier_source_sha256']!='fc12a07a372c3597402bf6002e6ef000c63bd7a50db1717c47f2d0c18a3addf8' or
        len(rows)!=8 or [x['virtual_class'] for x in rows]!=list(CLASS_ORDER)*2 or
        len({x['pose_id'] for x in rows})!=8):
        raise RuntimeError('second-seed pilot selection drift')
    for i,cls in enumerate(CLASS_ORDER):
        if rows[i]['source_family']==rows[i+4]['source_family']:
            raise RuntimeError('class does not have distinct source families: '+cls)
    if (scan['candidate_count']!=448 or scan['dem_complete']!=448 or
        scan['robustness_complete']!=448 or scan['physics_steps_after_scene_init']!=0):
        raise RuntimeError('offline scan incomplete')
    for rad,expected in pilot['scan_sha256'].items():
        if sha(ROOT/'scan-v1'/rad)!=expected:raise RuntimeError('scan input SHA drift: '+rad)
    if json.loads((ROOT/'scan-status.json').read_text())['stage']!='complete':
        raise RuntimeError('scan was not sealed')
    original_freeze=json.loads((OLD/'shard-A/input-freeze.json').read_text())
    if (original_freeze['contact_contract']['view_capacity']!=4096 or
        original_freeze['planned_physics_steps']!=200 or
        original_freeze['numeric_diagnostics']!={'angular_speed_max_radps':.2,
            'dem_valid_fraction_min':.7,'h1_abs_max_rad':.1,
            'linear_speed_max_mps':.1,'xy_drift_max_m':.1}):
        raise RuntimeError('first-seed physical/numeric contract drift')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip()
    write_new(ROOT/'primary-manifest.json',{'schema':'eatnav_r3_primary_v1',
        'source_commit':commit,'second_seed_pilot_sha256':sha(pilot_path),'poses':rows})
    reserves=json.loads((selected/'reserve-rank-freeze.json').read_text())['poses']
    write_new(ROOT/'execution-manifest.json',{'schema':'eatnav_r3_second_two_shard_v1',
        'shard_order':['shard-A','shard-B'],'primary_pose_ids':[x['pose_id'] for x in rows],
        'shards':[{'shard_id':name,'pose_ids':[x['pose_id'] for x in group]}
                  for name,group in (('shard-A',rows[:4]),('shard-B',rows[4:]))],
        'reserve_pose_ids_excluded':[x['pose_id'] for x in reserves],
        'candidate_free':True,'physics_steps_per_shard':200})
    source_files=[WORKTREE/'scripts/eat_nav_r3_second_settle_sim.py',
        WORKTREE/'scripts/eat_nav_r3_second_settle_run.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/r3_anchor_contract.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/radial_shard_inventory.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/radial_termination_capture.py']
    source_shas={str(p):sha(p) for p in source_files}
    for name,group in (('shard-A',rows[:4]),('shard-B',rows[4:])):
        shard=ROOT/name;shard.mkdir(exist_ok=False)
        write_new(shard/'pose-manifest.json',{'schema':'eatnav_r3_shard_poses_v1',
            'shard_id':name,'poses':group})
        freeze=json.loads(json.dumps(original_freeze))
        freeze.update(schema='eatnav_r3_second_settling_freeze_v1',shard_id=name,
            new_top_root=str(ROOT),inventory_root=str(ROOT),
            inventory_manifest_sha256=sha(pilot_path),batch_seeds=[seed['batch_seed']],
            expected_pose_count=4,pose_manifest_sha256=sha(shard/'pose-manifest.json'),
            primary_manifest_path=str(ROOT/'primary-manifest.json'),
            primary_manifest_sha256=sha(ROOT/'primary-manifest.json'),
            second_seed_pilot_sha256=sha(pilot_path),
            terrain_mesh_sha256=scan['terrain_mesh_sha256'],
            sim_source_sha256=sha(WORKTREE/'scripts/eat_nav_r3_second_settle_sim.py'),
            remediation_cycle=0,remediation_cycles_allowed=2,
            no_further_automatic_retry=True,candidate_free=True)
        freeze['contact_contract']['views']=4
        for key in ('sim_source','r3_anchor_contract','radial_lean_log',
                    'radial_shard_inventory','radial_termination_capture'):
            old_path=Path(freeze['source_paths'][key])
            new_path=shard/old_path.relative_to(OLD/'shard-A')
            if key=='sim_source':new_path=shard/'inputs/eat_nav_r3_second_settle_sim.py'
            freeze['source_paths'][key]=str(new_path)
        for key in ('r3_anchor_contract','radial_lean_log','radial_shard_inventory','radial_termination_capture'):
            freeze['source_sha256'][key]=sha(WORKTREE/'src/visualnav_transformer/eat_nav'/f'{key}.py')
        freeze['source_sha256']['sim_source']=freeze['sim_source_sha256']
        write_new(shard/'input-freeze.json',freeze)
        pythonpath=':'.join((str(shard/'src'),'/tmp/eatnav-train/src',
            '/tmp/eatnav-extended/isaac-source','/workspace/isaaclab/source/isaaclab',
            '/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks',
            '/workspace/isaaclab_onix/source/onix_isaac_lab'))
        command=['docker','exec','-e','PYTHONPATH='+pythonpath,'isaac-lab-onix',
            '/isaac-sim/python.sh',str(shard/'inputs/eat_nav_r3_second_settle_sim.py'),
            '--headless','--freeze',str(shard/'input-freeze.json'),
            '--pose-manifest',str(shard/'pose-manifest.json'),
            '--batch-seed',str(seed['batch_seed']),'--output',str(shard/'run'),'--mode','full']
        write_new(shard/'prelaunch.json',{'schema':'eatnav_r3_second_prelaunch_v1',
            'source_commit':commit,'worktree':str(WORKTREE),'shard_id':name,
            'source_and_test_sha256':source_shas,
            'freeze_sha256':sha(shard/'input-freeze.json'),
            'pose_manifest_sha256':sha(shard/'pose-manifest.json'),
            'second_seed_pilot_sha256':sha(pilot_path),
            'command':command,'candidate_free':True})
    write_new(ROOT/'pilot-preflight.json',{'schema':'eatnav_r3_second_pilot_preflight_v1',
        'source_commit':commit,'worktree':str(WORKTREE),
        'seed_freeze_sha256':sha(ROOT/'seed-freeze.json'),
        'robustness_freeze_sha256':sha(ROOT/'robustness-freeze.json'),
        'pilot_manifest_sha256':sha(pilot_path),
        'primary_manifest_sha256':sha(ROOT/'primary-manifest.json'),
        'execution_manifest_sha256':sha(ROOT/'execution-manifest.json'),
        'terrain_mesh_sha256':scan['terrain_mesh_sha256'],
        'source_and_test_sha256':source_shas,
        'runtime_forecast_sec':180,'runtime_cap_per_shard_sec':900,
        'storage_forecast_bytes':60_000_000,'storage_cap_bytes':500_000_000,
        'external_router_follower_bridge_required':False,
        'candidate_free':True})
    print(json.dumps({'pilot_manifest_sha256':sha(pilot_path),'pose_ids':[x['pose_id'] for x in rows],
        'preflight_sha256':sha(ROOT/'pilot-preflight.json')},indent=2))

if __name__=='__main__':main()
