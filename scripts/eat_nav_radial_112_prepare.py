"""Freeze the complete sharded schedule and all inputs before the first result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

WORKTREE=Path('/tmp/eatnav-vnt-radial-112-sharded')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_shard_inventory import build_inventory, validate_inventory, sensor_paths_for_env

ROOT=Path('/tmp/eatnav-radial-112-sharded-20261007')
PRIOR=Path('/tmp/eatnav-radial-lean-smoke-20261007/cycle-1')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def main():
    if ROOT.exists():raise RuntimeError('inventory output root already exists')
    if subprocess.run(['git','status','--porcelain'],cwd=WORKTREE,capture_output=True,text=True,check=True).stdout:
        raise RuntimeError('inventory worktree must be clean and committed before freezing')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=WORKTREE,capture_output=True,text=True,check=True).stdout.strip()
    prior_freeze=json.loads((PRIOR/'input-freeze.json').read_text())
    templates=list(prior_freeze['contact_contract']['sensor_paths_by_pose'].values())
    if len(templates)!=5 or any([p.rsplit('/onix_chassis/',1)[-1] for p in t]!=
        [p.rsplit('/onix_chassis/',1)[-1] for p in templates[0]] for t in templates):
        raise ValueError('previous five views do not share an identical frozen body set')
    inventory=build_inventory();validate_inventory(inventory)
    source_paths=[WORKTREE/'scripts/eat_nav_radial_five_pose_sim.py',
        WORKTREE/'scripts/eat_nav_radial_five_pose_run.py',
        WORKTREE/'scripts/eat_nav_radial_112_prepare.py',
        WORKTREE/'scripts/eat_nav_radial_112_run.py',
        WORKTREE/'scripts/eat_nav_radial_112_analyze.py',
        *sorted((WORKTREE/'src/visualnav_transformer/eat_nav').glob('*contact*.py')),
        WORKTREE/'src/visualnav_transformer/eat_nav/radial_shard_inventory.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/radial_interaction.py',
        WORKTREE/'src/visualnav_transformer/eat_nav/radial_five_pose_contract.py',
        *sorted((WORKTREE/'tests').glob('test_radial_*.py')),
        WORKTREE/'tests/test_contact_host_gate.py',WORKTREE/'tests/test_contact_raw_capture.py']
    source_sha={str(p):sha(p) for p in source_paths}
    ROOT.mkdir(parents=True,exist_ok=False)
    write(ROOT/'inventory-manifest.json',inventory)
    prior_files=[PRIOR/'input-freeze.json',PRIOR/'pose-manifest.json',PRIOR/'status.json',PRIOR/'analysis.json',
                 PRIOR/'final-output-sha-ledger.json',Path('/tmp/eatnav-contact-index-20261007/cycle-2-full/input-freeze.json'),
                 Path('/tmp/eatnav-radial-terrain-inventory-20261007/input-freeze.json')]
    write(ROOT/'prior-root-readonly-sha.json',[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in prior_files if p.exists()])
    sim_sha=sha(WORKTREE/'scripts/eat_nav_radial_five_pose_sim.py')
    for shard in inventory['shards']:
        sr=ROOT/'cycle-0'/shard['shard_id'];sr.mkdir(parents=True,exist_ok=False)
        poses=[{**p,'shard_id':shard['shard_id'],'expected_root_z_rule':'scene.env_origins.z + 0.35 m',
                'requested_xy_yaw_rule':'pose_world(origin_xyz, angle_index)'} for p in shard['poses']]
        manifest={'schema':'eatnav_radial_shard_pose_v1','shard_id':shard['shard_id'],
                  'inventory_sha256':sha(ROOT/'inventory-manifest.json'),
                  'source_sim_sha256':sim_sha,
                  'eat_config_sha256':prior_freeze['source_sha256']['eat_config'],
                  'batch_seed':20261007,'terrain_difficulty':0.5,'poses':poses}
        write(sr/'pose-manifest.json',manifest)
        freeze=json.loads(json.dumps(prior_freeze))
        freeze.update({'schema':'eatnav_radial_112_shard_freeze_v1','new_top_root':str(ROOT),
                       'inventory_root':str(ROOT),'inventory_manifest_sha256':sha(ROOT/'inventory-manifest.json'),
                       'shard_id':shard['shard_id'],'expected_pose_count':len(poses),
                       'pose_manifest_sha256':sha(sr/'pose-manifest.json'),
                       'sim_source_sha256':sim_sha,'remediation_cycles_allowed':2,
                       'numeric_exceedance_stop':'strict_majority_of_completed_shard',
                       'run_kind':'full','planned_physics_steps':200})
        freeze['contact_contract']['sensor_path_template']=templates[0]
        freeze['contact_contract']['sensor_paths_by_pose']={}
        freeze['contact_contract']['view_capacity']=4096
        freeze['source_paths']['sim_source']=str(sr/'inputs/eat_nav_radial_five_pose_sim.py')
        freeze['source_sha256']['sim_source']=sim_sha
        freeze['source_paths']['radial_lean_log']=str(sr/'src/visualnav_transformer/eat_nav/radial_lean_log.py')
        freeze['source_sha256']['radial_lean_log']=sha(WORKTREE/'src/visualnav_transformer/eat_nav/radial_lean_log.py')
        freeze['source_paths']['radial_shard_inventory']=str(sr/'src/visualnav_transformer/eat_nav/radial_shard_inventory.py')
        freeze['source_sha256']['radial_shard_inventory']=sha(WORKTREE/'src/visualnav_transformer/eat_nav/radial_shard_inventory.py')
        write(sr/'input-freeze.json',freeze)
        command=['docker','exec','-e',f'PYTHONPATH={sr}/src:/tmp/eatnav-train/src:/tmp/eatnav-extended/isaac-source:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab_onix/source/onix_isaac_lab',
                 'isaac-lab-onix','/isaac-sim/python.sh',str(sr/'inputs/eat_nav_radial_five_pose_sim.py'),
                 '--headless','--freeze',str(sr/'input-freeze.json'),'--pose-manifest',str(sr/'pose-manifest.json'),
                 '--batch-seed','20261007','--output',str(sr/'run'),'--mode','full']
        write(sr/'prelaunch.json',{'schema':'eatnav_radial_shard_prelaunch_v1','shard_id':shard['shard_id'],
            'implementation_commit':commit,'branch':'exp/eatnav-radial-112-sharded','worktree':str(WORKTREE),
            'worktree_status':'clean','source_and_test_sha256':source_sha,
            'freeze_sha256':sha(sr/'input-freeze.json'),'pose_manifest_sha256':sha(sr/'pose-manifest.json'),
            'command':command,'source_family_note':'directions within a terrain share one terrain seed/family'})
    write(ROOT/'preflight.json',{'schema':'eatnav_radial_112_preflight_v1','implementation_commit':commit,
          'branch':'exp/eatnav-radial-112-sharded','worktree':str(WORKTREE),'worktree_status':'clean',
          'inventory_manifest_sha256':sha(ROOT/'inventory-manifest.json'),
          'prior_root_sha_ledger_sha256':sha(ROOT/'prior-root-readonly-sha.json'),
          'pose_count':112,'shard_count':23,'max_selected_views':5,
          'forecast_20pct_margin':{'runtime_minutes':17.7,'storage_bytes':254_000_000},
          'source_and_test_sha256':source_sha,
          'prospective_interaction_thresholds':{'height_change_m':.04,'step_change_m':.05,'grade_slope':.12,
              'differential_m':.04,'min_corridor_valid':.70,'min_band_valid':.50,
              'two_second_reach_proxy_m':.60,'assumed_speed_mps':.30},
          'numeric_exceedance_stop':'strict majority of completed shard',
          'candidate_free':True,'second_seed_executed':False})
    print(ROOT,sha(ROOT/'inventory-manifest.json'),commit)


if __name__=='__main__':main()
