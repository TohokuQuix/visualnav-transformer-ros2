"""Freeze the second-seed zero-step scan's exact source/input contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/tmp/eatnav-r3-second-seed-20261008')
WORKTREE=Path('/tmp/eatnav-vnt-r3-second-seed')
OLD=Path('/tmp/eatnav-anchor-placement-20261008')
MODULES=('config','dem','radial_inventory','radial_interaction',
         'anchor_placement_geometry','r3_robustness')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('frozen input already exists: '+str(path))
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    seed=json.loads((ROOT/'seed-freeze.json').read_text())
    if sha(ROOT/'seed-freeze.json')!=json.loads((ROOT/'seed-freeze-sha.json').read_text())['sha256']:
        raise RuntimeError('second seed SHA drift')
    robustness=json.loads((ROOT/'robustness-freeze.json').read_text())
    if robustness['condition_count']!=15 or robustness['selection_use'] is not False:
        raise RuntimeError('robustness definition drift')
    old=json.loads((OLD/'inputs/pose-frames.json').read_text())
    keys=('checkpoint_env','eat_config','rl_checkpoint','terrain_generator','eat_env_cfg',
          'rough_cfg','raycaster','installed_source_4','radial_module')
    sources={key:old['frozen_source_paths'][key] for key in keys}
    source_sha={key:old['frozen_source_sha256'][key] for key in keys}
    for name in MODULES:
        label='second_'+name
        local=WORKTREE/'src/visualnav_transformer/eat_nav'/f'{name}.py'
        sources[label]=str(ROOT/'src/visualnav_transformer/eat_nav'/f'{name}.py')
        source_sha[label]=sha(local)
    sim=WORKTREE/'scripts/eat_nav_r3_second_scan_sim.py'
    sources['second_scan_sim']=str(ROOT/'inputs'/sim.name)
    source_sha['second_scan_sim']=sha(sim)
    if (source_sha['second_radial_inventory']!='fdb1edfb9ef056926c164e3a0b9ea49584f6357d6fe4a2e917280864af62996c' or
        source_sha['second_radial_interaction']!='fc12a07a372c3597402bf6002e6ef000c63bd7a50db1717c47f2d0c18a3addf8' or
        sha(WORKTREE/'src/visualnav_transformer/eat_nav/anchor_placement_rules.py')!='60da9ea1a4a7d6cfdde812ac49bc09561518710f7127d1d8cf86111b28ecb3de'):
        raise RuntimeError('first-seed geometry/classifier/R3 source changed')
    write_new(ROOT/'inputs/scan-inputs.json',{
        'schema':'eatnav_r3_second_scan_inputs_v1','batch_seed':seed['batch_seed'],
        'seed_freeze_sha256':sha(ROOT/'seed-freeze.json'),
        'robustness_freeze_sha256':sha(ROOT/'robustness-freeze.json'),
        'frozen_source_paths':sources,'frozen_source_sha256':source_sha,
        'terrain_count':7,'direction_count':16,'radii_m':[.30,.40,.50,.60],
        'terrain_difficulty':.5,'physics_steps':0})
    pythonpath=':'.join((str(ROOT/'src'),'/tmp/eatnav-train/src',
        '/tmp/eatnav-extended/isaac-source','/workspace/isaaclab/source/isaaclab',
        '/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks',
        '/workspace/isaaclab_onix/source/onix_isaac_lab'))
    command=['docker','exec','-e','PYTHONPATH='+pythonpath,'isaac-lab-onix',
        '/isaac-sim/python.sh',str(ROOT/'inputs'/sim.name),'--headless',
        '--inputs',str(ROOT/'inputs/scan-inputs.json'),
        '--seed-freeze',str(ROOT/'seed-freeze.json'),
        '--robustness-freeze',str(ROOT/'robustness-freeze.json'),
        '--output',str(ROOT/'scan-v1')]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip()
    write_new(ROOT/'scan-preflight.json',{'schema':'eatnav_r3_second_scan_preflight_v1',
        'worktree':str(WORKTREE),'execution_commit':commit,
        'seed_freeze_sha256':sha(ROOT/'seed-freeze.json'),
        'robustness_freeze_sha256':sha(ROOT/'robustness-freeze.json'),
        'scan_inputs_sha256':sha(ROOT/'inputs/scan-inputs.json'),
        'source_sha256':source_sha,'command':command,
        'forecast_sec':300,'runtime_cap_sec':900,'storage_forecast_bytes':120_000_000,
        'storage_cap_bytes':500_000_000,'candidate_free':True})
    print(json.dumps({'commit':commit,'batch_seed':seed['batch_seed'],
        'scan_inputs_sha256':sha(ROOT/'inputs/scan-inputs.json')},indent=2))

if __name__=='__main__':main()
