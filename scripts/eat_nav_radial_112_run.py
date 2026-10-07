"""Detached sequential coordinator for the immutable first-seed radial shards."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORKTREE = Path('/tmp/eatnav-vnt-radial-112-sharded')
sys.path.insert(0, str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_shard_inventory import (
    validate_inventory, numerical_exceedance_stop, settled_exceedance_reasons,
)

ROOT = Path(os.environ['EATNAV_INVENTORY_ROOT'])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def summarize_shard(root, poses):
    rows=[]
    for pose in poses:
        pose_id=pose['pose_id']
        trace=[json.loads(line) for line in (root/'run/trace'/f'{pose_id}.jsonl').read_text().splitlines()]
        fractions=[json.loads((root/'run/dem'/pose_id/f'frame-{i:02d}.json').read_text())['valid_fraction'] for i in range(4)]
        reasons=settled_exceedance_reasons(trace,fractions)
        rows.append({'pose_id':pose_id,'terrain':pose['terrain'],'angle_index':pose['angle_index'],
                     'measurement_valid':True,'settled_anchor_eligible':not reasons,
                     'exceedance_reasons':reasons,'dem_valid_fractions':fractions})
    result={'shard_id':json.loads((root/'input-freeze.json').read_text())['shard_id'],
            'pose_results':rows,'numeric_exceedance_majority':numerical_exceedance_stop([bool(x['exceedance_reasons']) for x in rows])}
    atomic(root/'shard-summary.json',result)
    return result


def main():
    inventory=json.loads((ROOT/'inventory-manifest.json').read_text())
    validate_inventory(inventory)
    initial=json.loads((ROOT/'preflight.json').read_text())
    if sha(ROOT/'inventory-manifest.json')!=initial['inventory_manifest_sha256']:
        raise RuntimeError('frozen inventory SHA drift')
    status_path=ROOT/'status.json'
    if status_path.exists():
        raise RuntimeError('coordinator already started; use a new cycle root after failure')
    started=time.monotonic()
    atomic(status_path,{'stage':'running','completed_shards':0,'current_shard':None,
                        'physics_executions':0,'decision':None,'wall_ns':time.time_ns()})
    completed=[]
    try:
        for shard in inventory['shards']:
            shard_id=shard['shard_id'];root=ROOT/'cycle-0'/shard_id
            if root.joinpath('host-launch.json').exists() or root.joinpath('seal.json').exists():
                raise RuntimeError('shard root is not fresh: '+shard_id)
            pre=json.loads((root/'prelaunch.json').read_text())
            freeze=json.loads((root/'input-freeze.json').read_text())
            if sha(root/'input-freeze.json')!=pre['freeze_sha256'] or sha(root/'pose-manifest.json')!=freeze['pose_manifest_sha256']:
                raise RuntimeError('shard input SHA drift: '+shard_id)
            if [x['pose_id'] for x in json.loads((root/'pose-manifest.json').read_text())['poses']] != [x['pose_id'] for x in shard['poses']]:
                raise RuntimeError('shard order or membership drift: '+shard_id)
            atomic(status_path,{'stage':'running','completed_shards':len(completed),'current_shard':shard_id,
                                'physics_executions':len(completed),'elapsed_sec':time.monotonic()-started,'decision':None})
            env={**os.environ,'EATNAV_FIVE_POSE_ROOT':str(root)}
            with (root/'host-runner.stdout.log').open('wb') as stream:
                child=subprocess.run([sys.executable,'-u',str(WORKTREE/'scripts/eat_nav_radial_five_pose_run.py')],
                                     cwd=WORKTREE,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if child.returncode!=0:
                raise RuntimeError(f'shard host launcher failed: {shard_id}, exit {child.returncode}')
            seal=json.loads((root/'seal.json').read_text())
            host=json.loads((root/'status.json').read_text())
            if (host.get('stage')!='full_complete' or host.get('owned_residual') is not False or
                seal['shard_id']!=shard_id or seal['native_exit_code']!=0 or
                sha(root/'seal.json')!=host['seal_sha256'] or
                sha(root/'sha-ledger.json')!=seal['run_sha_ledger_sha256']):
                raise RuntimeError('invalid shard seal: '+shard_id)
            result=summarize_shard(root,shard['poses'])
            completed.append({'shard_id':shard_id,'seal_sha256':sha(root/'seal.json'),
                              'summary_sha256':sha(root/'shard-summary.json'),
                              'numeric_exceedance_majority':result['numeric_exceedance_majority']})
            atomic(ROOT/'completed-shards.json',{'completed':completed})
            if result['numeric_exceedance_majority']:
                raise RuntimeError('prospective majority numeric exceedance stop: '+shard_id)
        atomic(status_path,{'stage':'measurement_complete','completed_shards':len(completed),
                            'current_shard':None,'physics_executions':len(completed),
                            'elapsed_sec':time.monotonic()-started,'decision':'analysis_pending'})
        analysis=subprocess.run(['/usr/bin/python3',str(WORKTREE/'scripts/eat_nav_radial_112_analyze.py'),str(ROOT)],
                                cwd=WORKTREE,capture_output=True,text=True,env={**os.environ,'MPLCONFIGDIR':str(ROOT/'mpl-cache')})
        (ROOT/'analysis.stdout.log').write_text(analysis.stdout+analysis.stderr)
        if analysis.returncode!=0:
            raise RuntimeError('offline analysis failed after complete measurement')
        final=json.loads((ROOT/'final-decision.json').read_text())
        atomic(status_path,{'stage':'complete','completed_shards':23,'current_shard':None,
                            'physics_executions':23,'elapsed_sec':time.monotonic()-started,
                            'decision':final['decision']})
        ledger=[]
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and path.name!='final-output-sha-ledger.json':
                ledger.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
        atomic(ROOT/'final-output-sha-ledger.json',{'count':len(ledger),'files':ledger})
    except BaseException as exc:
        reason=repr(exc)
        decision='radial_inventory_partial_measurement_review_required'
        if any(s in reason for s in ('SHA','seal','resource','integrity','capacity','bounds','nonfinite','full-buffer')):
            decision='resource_or_integrity_stop'
        atomic(status_path,{'stage':'stopped','completed_shards':len(completed),
                            'current_shard':inventory['shards'][len(completed)]['shard_id'] if len(completed)<23 else None,
                            'physics_executions':len(completed),'elapsed_sec':time.monotonic()-started,
                            'decision':decision,'error':reason,'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
