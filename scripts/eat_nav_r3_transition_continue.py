"""One-shot candidate-free reserve shard coordinator; no automatic retry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/tmp/eatnav-r3-transition-reserve-20261008')
WORKTREE=Path('/tmp/eatnav-vnt-r3-transition-reserve')
sys.path.insert(0,str(WORKTREE/'scripts'))
from eat_nav_r3_transition_prepare import sha, write_new


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    ledger=json.loads((ROOT/'freeze-sha.json').read_text())
    pre=json.loads((ROOT/'preflight.json').read_text())
    sr=ROOT/'shard-reserve'
    freeze=json.loads((sr/'input-freeze.json').read_text())
    launch=json.loads((sr/'prelaunch.json').read_text())
    if (sha(ROOT/'reserve-manifest.json')!=ledger['reserve_manifest_sha256'] or
        sha(ROOT/'primary-manifest.json')!=pre['primary_manifest_sha256'] or
        sha(sr/'pose-manifest.json')!=pre['pose_manifest_sha256'] or
        sha(sr/'input-freeze.json')!=pre['freeze_sha256'] or
        sha(sr/'input-freeze.json')!=launch['freeze_sha256'] or
        launch['reserve_manifest_sha256']!=ledger['reserve_manifest_sha256'] or
        freeze['contact_contract']['view_capacity']!=4096 or
        freeze['planned_physics_steps']!=200 or
        freeze['terrain_mesh_sha256']!=pre['terrain_mesh_sha256']):
        raise RuntimeError('frozen reserve input drift')
    if (ROOT/'status.json').exists() or (sr/'host-launch.json').exists():
        raise RuntimeError('one-shot reserve already launched')
    write_new(ROOT/'status.json',{'stage':'running','candidate_free':True,'completed_shards':0,
        'reserve_manifest_sha256':ledger['reserve_manifest_sha256']})
    started=time.monotonic()
    try:
        with (ROOT/'coordinator.log').open('wb') as log:
            run=subprocess.run(['/home/kojima/anaconda3/envs/tinyvla/bin/python','-u',
                str(WORKTREE/'scripts/eat_nav_r3_transition_run.py')],cwd=WORKTREE,
                env={**os.environ,'EATNAV_R3_TRANSITION_SHARD_ROOT':str(sr),'PYTHONPATH':str(WORKTREE/'src')},
                stdout=log,stderr=subprocess.STDOUT)
        if run.returncode!=0:
            raise RuntimeError(f'reserve shard host runner failed: {run.returncode}')
        seal=json.loads((sr/'seal.json').read_text())
        status=json.loads((sr/'status.json').read_text())
        poses=json.loads((sr/'pose-manifest.json').read_text())['poses']
        if (status['stage']!='full_complete' or status['owned_residual'] is not False or
            seal['native_exit_code']!=0 or seal['owned_residual'] is not False or
            seal['pose_ids']!=[x['pose_id'] for x in poses] or
            sha(sr/'seal.json')!=status['seal_sha256'] or
            sha(sr/'sha-ledger.json')!=seal['run_sha_ledger_sha256']):
            raise RuntimeError('reserve shard seal invalid')
        atomic(ROOT/'status.json',{'stage':'measurement_complete','candidate_free':True,
            'completed_shards':1,'elapsed_sec':time.monotonic()-started,
            'reserve_manifest_sha256':ledger['reserve_manifest_sha256'],
            'seal_sha256':sha(sr/'seal.json'),'decision':'offline_analysis_pending'})
    except BaseException as exc:
        reason=repr(exc)
        code='resource_or_integrity_stop' if any(w in reason.lower() for w in
            ('resource','integrity','sha','seal','cap','timeout','residual','bounds')) else 'r3_transition_pilot_measurement_invalid'
        atomic(ROOT/'status.json',{'stage':'stopped','candidate_free':True,'completed_shards':0,
            'decision':code,'error':reason,'traceback':traceback.format_exc(),
            'elapsed_sec':time.monotonic()-started,
            'reserve_manifest_sha256':ledger['reserve_manifest_sha256']})
        raise

if __name__=='__main__':main()
