"""One-process period-2 continuation with external logs and sealed prior audit."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from eat_nav_p1_g3_crossover_outer import command_for
from eat_nav_p1_g3_material_diag_outer import run_outer
from eat_nav_p1_g3_crossover_v2_continuation import (
    SCHEMA, sha, verify_prior_pair, write_new)


def preflight(root: Path, host: Path) -> dict:
    root=Path(root)
    if root.name!='g3-crossover-v2-period2-continuation-attempt-0':
        raise ValueError('fresh continuation root identity')
    manifest=json.loads((root/'continuation-manifest.json').read_text())
    contract=json.loads((root/'manifest.json').read_text())
    batch=root/'batch-first-differential-0-period-2'
    if manifest['schema']!=SCHEMA or manifest['authorized_process_count']!=1 or \
            sha(Path(__file__))!=manifest['continuation_outer_source_sha256'] or \
            sha(root/'manifest.json')!=manifest['period2_manifest_sha256'] or \
            sha(root/'g3-assignment.json')!=manifest['assignment_sha256'] or \
            sha(host)!=manifest['period2_host_source_sha256'] or \
            sha(batch/'host-runner.py')!=manifest['period2_host_source_sha256'] or \
            sha(batch/'source.py')!=manifest['period2_simulator_source_sha256'] or \
            sha(batch/'input/g3-contract.json')!=manifest['period2_manifest_sha256'] or \
            contract['runtime_scope']!='authorized_period2_continuation_only':
        raise ValueError('continuation source/manifest drift')
    ledger=json.loads((batch/'input-sha-ledger.json').read_text())
    if any(sha(batch/name)!=expected for name,expected in ledger.items()):
        raise ValueError('continuation input ledger drift')
    prior=verify_prior_pair(root,manifest['prior_periods'])
    old_host=Path(manifest['prior_periods'][0]['root'])/'host-runner.py'
    patch=''.join(difflib.unified_diff(
        old_host.read_text().splitlines(keepends=True),host.read_text().splitlines(keepends=True),
        fromfile='period0_1_host.py',tofile='period2_host.py'))
    if hashlib.sha256(patch.encode()).hexdigest()!=manifest['validator_diff_sha256']:
        raise ValueError('validator source diff drift')
    key=lambda p:(p['anchor_id'],p['batch_index'],p['execution_index'])
    staged=json.loads((batch/'input/g3-assignment.json').read_text())
    frozen={key(p):p['trials'] for p in contract['processes']}
    assigned={key(p):p['trials'] for p in staged['processes']}
    wanted=('first-differential',0,2)
    if assigned.get(wanted)!=frozen.get(wanted) or \
            [r['path_role'] for r in assigned[wanted]]!=['right_20','left_20','right_20','left_20']:
        raise ValueError('period-2 Path assignment differs from frozen manifest')
    return {'schema':SCHEMA,'prior_seals':prior,
            'continuation_manifest_sha256':sha(root/'continuation-manifest.json'),
            'period2_manifest_sha256':sha(root/'manifest.json'),
            'period2_input_ledger_sha256':sha(batch/'input-sha-ledger.json'),
            'Path_publications':0}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--external-log-dir',type=Path,required=True)
    parser.add_argument('--host-script',type=Path,required=True)
    parser.add_argument('--router-log-dir',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    if (root/'continuation-preflight.json').exists() or args.external_log_dir.exists():
        raise FileExistsError('continuation already attempted')
    evidence=preflight(root,args.host_script)
    write_new(root/'continuation-preflight.json',evidence)
    batch=root/'batch-first-differential-0-period-2'
    cmd=command_for(batch,args.remote,args.host_script,args.router_log_dir,
                    'first-differential',0,2)
    result=run_outer(cmd,batch,args.external_log_dir)
    verify_prior_pair(root,json.loads((root/'continuation-manifest.json').read_text())['prior_periods'])
    if result['host_exit_code']!=0 or result['post_launcher_seal_error'] is not None or \
            result['post_launcher_seal_verification']['stage']!='G3_crossover_batch_protocol_valid':
        raise RuntimeError('continuation host/launcher failed: '+str(result))
    print('G3_PERIOD2_CONTINUATION_PASS',batch)


if __name__=='__main__':main()
