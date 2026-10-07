"""Create an immutable-input continuation for shards 15–22; never edit the old root."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WORKTREE = Path('/tmp/eatnav-vnt-radial-112-sharded')
OLD = Path('/tmp/eatnav-radial-112-sharded-20261007')
ROOT = Path(os.environ.get('EATNAV_TERMINATION_ROOT', '/tmp/eatnav-radial-termination-20261008'))
sys.path.insert(0, str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_shard_inventory import validate_inventory, verify_shard_artifacts


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def validate_reuse(inventory):
    if json.loads((OLD/'status.json').read_text())['completed_shards']!=15:
        raise ValueError('old run did not seal exactly 15 shards')
    reused=[]
    for shard in inventory['shards'][:15]:
        name=shard['shard_id'];sr=OLD/'cycle-0'/name
        seal=json.loads((sr/'seal.json').read_text())
        status=json.loads((sr/'status.json').read_text())
        if (status['stage']!='full_complete' or status['seal_sha256']!=sha(sr/'seal.json') or
            seal['run_sha_ledger_sha256']!=sha(sr/'sha-ledger.json') or
            seal['pose_ids']!=[p['pose_id'] for p in shard['poses']] or
            seal['native_exit_code']!=0 or seal['owned_residual'] is not False):
            raise ValueError('old sealed shard invalid: '+name)
        ledger=json.loads((sr/'sha-ledger.json').read_text())
        if any(sha(entry['path'])!=entry['sha256'] for entry in ledger['files']):
            raise ValueError('old run artifact SHA mismatch: '+name)
        verify_shard_artifacts(sr/'run',seal['pose_ids'])
        reused.append({'shard_id':name,'old_root':str(sr),
                       'pose_ids':seal['pose_ids'],'seal_sha256':sha(sr/'seal.json'),
                       'run_sha_ledger_sha256':sha(sr/'sha-ledger.json'),
                       'input_freeze_sha256':sha(sr/'input-freeze.json'),
                       'pose_manifest_sha256':sha(sr/'pose-manifest.json'),
                       'summary_sha256':sha(sr/'shard-summary.json')})
    failed=OLD/'cycle-0/shard-15'
    if (failed/'seal.json').exists() or not (failed/'run/failure.json').exists():
        raise ValueError('old partial shard unexpectedly promoted or missing')
    return {'schema':'eatnav_radial_readonly_reuse_v1','old_root':str(OLD),
            'old_inventory_sha256':sha(OLD/'inventory-manifest.json'),
            'old_partial_analysis_sha256':sha(OLD/'partial-analysis.json'),
            'old_partial_shard_15_failure_sha256':sha(failed/'run/failure.json'),
            'reused_pose_count':75,'reused_shards':reused,
            'old_partial_shard_15_reused':False}


def main():
    if ROOT.exists():raise RuntimeError('continuation root already exists')
    if subprocess.run(['git','status','--porcelain'],cwd=WORKTREE,capture_output=True,text=True,check=True).stdout:
        raise RuntimeError('continuation source must be clean and committed')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=WORKTREE,capture_output=True,text=True,check=True).stdout.strip()
    inventory=json.loads((OLD/'inventory-manifest.json').read_text());validate_inventory(inventory)
    reuse=validate_reuse(inventory)
    old_freeze=json.loads((OLD/'cycle-0/shard-15/input-freeze.json').read_text())
    old_pre=json.loads((OLD/'cycle-0/shard-15/prelaunch.json').read_text())
    source_sha={path:sha(path) for path in old_pre['source_and_test_sha256']}
    for path in (WORKTREE/'src/visualnav_transformer/eat_nav/radial_termination_capture.py',
                 WORKTREE/'tests/test_radial_termination_capture.py',
                 WORKTREE/'scripts/eat_nav_radial_termination_prepare.py',
                 WORKTREE/'scripts/eat_nav_radial_termination_continue.py',
                 WORKTREE/'scripts/eat_nav_radial_termination_analyze.py'):
        source_sha[str(path)]=sha(path)
    ROOT.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(OLD/'inventory-manifest.json',ROOT/'inventory-manifest.json')
    write(ROOT/'reuse-ledger.json',reuse)
    sim_sha=sha(WORKTREE/'scripts/eat_nav_radial_five_pose_sim.py')
    for shard in inventory['shards'][15:]:
        name=shard['shard_id'];sr=ROOT/'cycle-1'/name;sr.mkdir(parents=True,exist_ok=False)
        original=OLD/'cycle-0'/name
        manifest=json.loads((original/'pose-manifest.json').read_text())
        if [p['pose_id'] for p in manifest['poses']] != [p['pose_id'] for p in shard['poses']]:
            raise ValueError('old pose order drift')
        manifest['source_sim_sha256']=sim_sha
        write(sr/'pose-manifest.json',manifest)
        freeze=json.loads((original/'input-freeze.json').read_text())
        if freeze['contact_contract'] != old_freeze['contact_contract']:
            raise ValueError('contact contract drift')
        freeze['new_top_root']=str(ROOT);freeze['inventory_root']=str(ROOT)
        freeze['pose_manifest_sha256']=sha(sr/'pose-manifest.json')
        freeze['sim_source_sha256']=sim_sha
        freeze['source_paths']['sim_source']=str(sr/'inputs/eat_nav_radial_five_pose_sim.py')
        freeze['source_sha256']['sim_source']=sim_sha
        for key,filename in [('radial_shard_inventory','radial_shard_inventory.py'),
                             ('radial_lean_log','radial_lean_log.py'),
                             ('radial_termination_capture','radial_termination_capture.py')]:
            source=WORKTREE/'src/visualnav_transformer/eat_nav'/filename
            freeze['source_paths'][key]=str(sr/'src/visualnav_transformer/eat_nav'/filename)
            freeze['source_sha256'][key]=sha(source)
        write(sr/'input-freeze.json',freeze)
        cmd=[str(v).replace(str(OLD/'cycle-0'/name),str(sr)) for v in json.loads((original/'prelaunch.json').read_text())['command']]
        write(sr/'prelaunch.json',{'schema':'eatnav_radial_termination_prelaunch_v1',
              'shard_id':name,'implementation_commit':commit,'worktree':str(WORKTREE),
              'source_and_test_sha256':source_sha,
              'freeze_sha256':sha(sr/'input-freeze.json'),
              'pose_manifest_sha256':sha(sr/'pose-manifest.json'),
              'command':cmd,'reused_old_shards':15,'candidate_free':True})
    write(ROOT/'preflight.json',{'schema':'eatnav_radial_termination_preflight_v1',
          'implementation_commit':commit,'inventory_manifest_sha256':sha(ROOT/'inventory-manifest.json'),
          'reuse_ledger_sha256':sha(ROOT/'reuse-ledger.json'),
          'new_shard_order':[s['shard_id'] for s in inventory['shards'][15:]],
          'source_and_test_sha256':source_sha,'old_partial_shard_15_promoted':False,
          'candidate_free':True,'second_seed_executed':False})
    print(ROOT,sha(ROOT/'preflight.json'),commit)


if __name__=='__main__':main()
