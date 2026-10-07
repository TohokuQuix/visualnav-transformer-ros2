"""Sequential, fail-closed continuation of the frozen radial inventory."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORKTREE=Path('/tmp/eatnav-vnt-radial-112-sharded')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_shard_inventory import (
    validate_inventory, numerical_exceedance_stop, settled_exceedance_reasons)
from eat_nav_radial_termination_prepare import validate_reuse, sha, write

ROOT=Path(os.environ.get('EATNAV_TERMINATION_ROOT','/tmp/eatnav-radial-termination-20261008'))


def summarize(root,poses):
    rows=[]
    for pose in poses:
        pid=pose['pose_id'];run=root/'run';terminal=run/'termination'/f'{pid}.json'
        if terminal.exists():
            event=json.loads(terminal.read_text())
            rows.append({'pose_id':pid,'terrain':pose['terrain'],
                         'classification':'anchor_ineligible_physical_termination',
                         'termination_step':event['termination_step'],
                         'termination_reason':event['termination_reason'],
                         'exceedance_reasons':[]})
        else:
            trace=[json.loads(line) for line in (run/'trace'/f'{pid}.jsonl').read_text().splitlines()]
            fractions=[json.loads((run/'dem'/pid/f'frame-{i:02d}.json').read_text())['valid_fraction'] for i in range(4)]
            reasons=settled_exceedance_reasons(trace,fractions)
            rows.append({'pose_id':pid,'terrain':pose['terrain'],
                         'classification':'anchor_ineligible_numeric' if reasons else 'settled_anchor_eligible',
                         'exceedance_reasons':reasons,'dem_valid_fractions':fractions})
    result={'shard_id':root.name,'pose_results':rows,
            'numeric_exceedance_majority':numerical_exceedance_stop(
                [x['classification']=='anchor_ineligible_numeric' for x in rows])}
    write(root/'shard-summary.json',result)
    return result


def main():
    inventory=json.loads((ROOT/'inventory-manifest.json').read_text());validate_inventory(inventory)
    preflight=json.loads((ROOT/'preflight.json').read_text())
    if sha(ROOT/'inventory-manifest.json')!=preflight['inventory_manifest_sha256'] or \
       sha(ROOT/'reuse-ledger.json')!=preflight['reuse_ledger_sha256']:
        raise RuntimeError('continuation preflight SHA drift')
    actual_reuse=validate_reuse(inventory)
    if actual_reuse!=json.loads((ROOT/'reuse-ledger.json').read_text()):
        raise RuntimeError('old read-only reuse changed')
    if preflight['new_shard_order']!=[s['shard_id'] for s in inventory['shards'][15:]]:
        raise RuntimeError('frozen continuation order drift')
    if (ROOT/'status.json').exists():raise RuntimeError('continuation already started')
    started=time.monotonic();completed=[]
    write(ROOT/'status.json',{'stage':'running','reused_shards':15,'new_completed_shards':0,
                              'current_shard':'shard-15','decision':None})
    try:
        for shard in inventory['shards'][15:]:
            name=shard['shard_id'];sr=ROOT/'cycle-1'/name
            if sr.joinpath('host-launch.json').exists() or sr.joinpath('seal.json').exists():
                raise RuntimeError('shard root already launched: '+name)
            freeze=json.loads((sr/'input-freeze.json').read_text())
            pre=json.loads((sr/'prelaunch.json').read_text())
            if (sha(sr/'input-freeze.json')!=pre['freeze_sha256'] or
                sha(sr/'pose-manifest.json')!=pre['pose_manifest_sha256'] or
                [x['pose_id'] for x in json.loads((sr/'pose-manifest.json').read_text())['poses']] !=
                [x['pose_id'] for x in shard['poses']] or
                freeze['shard_id']!=name or freeze['batch_seeds']!=[20261007] or
                freeze['contact_contract']['view_capacity']!=4096 or
                freeze['planned_physics_steps']!=200):
                raise RuntimeError('frozen shard input drift: '+name)
            if time.monotonic()-started>freeze['limits']['total_runtime_cap_sec']:
                raise RuntimeError('total runtime cap before shard: '+name)
            write(ROOT/'status.json',{'stage':'running','reused_shards':15,
                    'new_completed_shards':len(completed),'current_shard':name,
                    'elapsed_sec':time.monotonic()-started,'decision':None})
            env={**os.environ,'EATNAV_FIVE_POSE_ROOT':str(sr)}
            with (sr/'host-runner.stdout.log').open('wb') as out:
                child=subprocess.run([sys.executable,'-u',str(WORKTREE/'scripts/eat_nav_radial_five_pose_run.py')],
                                     cwd=WORKTREE,env=env,stdout=out,stderr=subprocess.STDOUT)
            if child.returncode!=0:raise RuntimeError(f'shard host launcher failed: {name}, exit {child.returncode}')
            seal=json.loads((sr/'seal.json').read_text());status=json.loads((sr/'status.json').read_text())
            if (status.get('stage')!='full_complete' or status.get('owned_residual') is not False or
                seal['native_exit_code']!=0 or seal['shard_id']!=name or
                sha(sr/'seal.json')!=status['seal_sha256'] or
                sha(sr/'sha-ledger.json')!=seal['run_sha_ledger_sha256']):
                raise RuntimeError('invalid termination-aware shard seal: '+name)
            result=summarize(sr,shard['poses'])
            completed.append({'shard_id':name,'seal_sha256':sha(sr/'seal.json'),
                              'summary_sha256':sha(sr/'shard-summary.json'),
                              'classifications':[x['classification'] for x in result['pose_results']]})
            write(ROOT/'completed-new-shards.json',{'completed':completed})
            if result['numeric_exceedance_majority']:
                raise RuntimeError('frozen strict-majority numeric stop: '+name)
        if validate_reuse(inventory)!=actual_reuse:
            raise RuntimeError('old read-only reuse changed after continuation')
        write(ROOT/'status.json',{'stage':'measurement_complete','reused_shards':15,
            'new_completed_shards':8,'elapsed_sec':time.monotonic()-started,'decision':'analysis_pending'})
        with (ROOT/'analysis.log').open('wb') as out:
            analysis=subprocess.run([sys.executable,'-u',str(WORKTREE/'scripts/eat_nav_radial_termination_analyze.py'),str(ROOT)],
                cwd=WORKTREE,env={**os.environ,'MPLCONFIGDIR':str(ROOT/'mpl-cache')},stdout=out,stderr=subprocess.STDOUT)
        if analysis.returncode!=0:raise RuntimeError('offline analysis failed')
        final=json.loads((ROOT/'final-decision.json').read_text())
        write(ROOT/'status.json',{'stage':'complete','reused_shards':15,
            'new_completed_shards':8,'completed_shards':23,'elapsed_sec':time.monotonic()-started,
            'decision':final['decision']})
        files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
               for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='final-output-sha-ledger.json']
        write(ROOT/'final-output-sha-ledger.json',{'count':len(files),'files':files})
    except BaseException as exc:
        reason=repr(exc)
        decision='radial_inventory_partial_measurement_review_required'
        if any(x in reason.lower() for x in ('sha','seal','resource','integrity','cap','bounds','nonfinite','unknown','timeout')):
            decision='resource_or_integrity_stop'
        write(ROOT/'status.json',{'stage':'stopped','reused_shards':15,
             'new_completed_shards':len(completed),'current_shard':inventory['shards'][15+len(completed)]['shard_id']
             if len(completed)<8 else None,'elapsed_sec':time.monotonic()-started,
             'decision':decision,'error':reason,'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
