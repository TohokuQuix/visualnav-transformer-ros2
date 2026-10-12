"""Freeze and stage one P1 G3 period-2 cross-attempt continuation."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil

from eat_nav_p1_g1_lifecycle import verify_seal

SCHEMA='eatnav_p1_g3_crossover_v2_period2_continuation_v1'
PERIOD_NAME='batch-first-differential-0-period-{}'


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(data,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _frozen_roles(contract: dict, period: int) -> list[str]:
    rows=[p for p in contract['processes'] if p['anchor_id']=='first-differential'
          and p['batch_index']==0 and p['execution_index']==period]
    if len(rows)!=1 or len(rows[0]['trials'])!=4:
        raise ValueError('frozen period assignment missing')
    return [r['path_role'] for r in sorted(rows[0]['trials'],key=lambda x:x['slot'])]


def verify_prior_reference(root: Path, spec: dict) -> dict:
    """Read and hash the original sealed batch; never write into it."""
    root=Path(root)
    if not root.is_dir():raise ValueError('prior root missing')
    checked=verify_seal(root)
    if checked['stage']!='G3_crossover_batch_protocol_valid':
        raise ValueError('prior protocol seal invalid')
    names={'seal':'host-seal.json','manifest':'input/g3-contract.json',
           'host_source':'host-runner.py','simulator_source':'source.py',
           'snapshot':'output/prepath-crossover-snapshot.json'}
    for key,name in names.items():
        if sha(root/name)!=spec[key+'_sha256']:
            raise ValueError('prior '+key+' SHA mismatch')
    contract=json.loads((root/'input/g3-contract.json').read_text())
    if contract['g3_host_source_sha256']!=spec['host_source_sha256'] or \
            contract['g3_simulator_source_sha256']!=spec['simulator_source_sha256']:
        raise ValueError('prior source provenance mismatch')
    if _frozen_roles(contract,spec['period'])!=spec['path_roles']:
        raise ValueError('prior path_roles mismatch')
    if sha(root/'input/g3-assignment.json')!=contract['assignment_sha256']:
        raise ValueError('prior assignment SHA mismatch')
    assignment=json.loads((root/'input/g3-assignment.json').read_text())
    key=lambda p:(p['anchor_id'],p['batch_index'],p['execution_index'])
    frozen={key(p):p['trials'] for p in contract['processes']}
    assigned={key(p):p['trials'] for p in assignment['processes']}
    wanted=('first-differential',0,spec['period'])
    if wanted not in assigned or assigned[wanted]!=frozen.get(wanted):
        raise ValueError('prior assignment rows differ from frozen manifest')
    if json.loads((root/'g3-host-status.json').read_text())['failure'] is not None:
        raise ValueError('prior host failure')
    return checked


def verify_prior_pair(root: Path, specs: list[dict]) -> dict:
    if [p['period'] for p in specs]!=[0,1]:raise ValueError('prior period sequence')
    result={}
    for spec in specs:
        alias=Path(root)/PERIOD_NAME.format(spec['period'])
        if not alias.is_symlink() or alias.resolve()!=Path(spec['root']).resolve():
            raise ValueError('prior alias target mismatch')
        result[str(spec['period'])]=verify_prior_reference(alias,spec)
    if specs[0]['manifest_sha256']!=specs[1]['manifest_sha256'] or \
            specs[0]['simulator_source_sha256']!=specs[1]['simulator_source_sha256'] or \
            specs[0]['host_source_sha256']!=specs[1]['host_source_sha256'] or \
            specs[0]['snapshot_sha256']!=specs[1]['snapshot_sha256']:
        raise ValueError('prior period provenance mismatch')
    return result


def make_freeze(repo: Path, old_root: Path) -> tuple[dict,dict]:
    base=json.loads((repo/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment_path=repo/'docs/eat-nav/p1-g3-crossover-v2-assignment.json'
    if sha(assignment_path)!=base['assignment_sha256']:
        raise ValueError('continuation assignment SHA drift')
    specs=[]
    for period in (0,1):
        batch=old_root/PERIOD_NAME.format(period)
        old_contract=json.loads((batch/'input/g3-contract.json').read_text())
        spec={'period':period,'root':str(batch.resolve()),
              'manifest_sha256':sha(batch/'input/g3-contract.json'),
              'host_source_sha256':sha(batch/'host-runner.py'),
              'simulator_source_sha256':sha(batch/'source.py'),
              'seal_sha256':sha(batch/'host-seal.json'),
              'snapshot_sha256':sha(batch/'output/prepath-crossover-snapshot.json'),
              'path_roles':_frozen_roles(old_contract,period)}
        verify_prior_reference(batch,spec)
        specs.append(spec)
    if any(spec['simulator_source_sha256']!=base['g3_simulator_source_sha256'] for spec in specs):
        raise ValueError('simulator source changed across attempts')
    if any(spec['manifest_sha256']!=specs[0]['manifest_sha256'] for spec in specs):
        raise ValueError('old manifest changed across periods')
    patch=''.join(difflib.unified_diff(
        (old_root/PERIOD_NAME.format(0)/'host-runner.py').read_text().splitlines(keepends=True),
        (repo/'scripts/eat_nav_p1_g3_crossover_v2_host.py').read_text().splitlines(keepends=True),
        fromfile='period0_1_host.py',tofile='period2_host.py'))
    patch_path=repo/'docs/eat-nav/p1-g3-crossover-v2-period2-validator.patch'
    if patch_path.read_text()!=patch or patch.count('@@')!=4 or \
            'validate_period_assignment' not in patch or 'approve_prepath' in patch:
        raise ValueError('host source delta extends beyond assignment validator')
    contract={**base,'runtime_scope':'authorized_period2_continuation_only',
              'continuation_reference':{'schema':SCHEMA,'prior_periods':specs,
                  'reason':'host assignment validator wrongly required all three roles within period 2',
                  'single_source_across_periods_claimed':False}}
    manifest={'schema':SCHEMA,'scope':'first-differential/batch-0/period-2-only',
              'prior_periods':specs,'period2_manifest_sha256':None,
              'period2_host_source_sha256':contract['g3_host_source_sha256'],
              'period2_simulator_source_sha256':contract['g3_simulator_source_sha256'],
              'validator_diff_sha256':sha(patch_path),
              'continuation_stage_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_crossover_v2_continuation.py'),
              'continuation_outer_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_crossover_v2_continue_outer.py'),
              'continuation_analyzer_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_crossover_v2_continue_analyze.py'),
              'assignment_sha256':contract['assignment_sha256'],
              'path_roles_by_period':{str(i):_frozen_roles(base,i) for i in (0,1,2)},
              'original_manifest_sha256':specs[0]['manifest_sha256'],
              'host_validator_change':'per-period rows compared directly with frozen manifest; no per-period three-role requirement',
              'comparison_basis':'same simulator source, assignment, Path SHA, checkpoint, terrain, seed, physics, follower, material gate, strict pre-Path fields and Outcome contract; host validation source differs and is retained per period',
              'single_source_across_periods_claimed':False,
              'authorized_process_count':1}
    return contract,manifest


def stage_continuation(repo: Path, old_stage: Path, specs: list[dict],
                       contract: dict, manifest: dict, root: Path) -> dict:
    """Stage only period 2; links to periods 0/1 are read-only references."""
    root=Path(root)
    if root.exists():raise FileExistsError('continuation root must be fresh')
    if manifest['schema']!=SCHEMA or contract['continuation_reference']['prior_periods']!=specs:
        raise ValueError('continuation manifest mismatch')
    if sha(Path(__file__))!=manifest['continuation_stage_source_sha256']:
        raise ValueError('continuation stage source SHA mismatch')
    if manifest['period2_manifest_sha256']!=sha(repo/'docs/eat-nav/p1-g3-crossover-v2-continuation-contract.json'):
        raise ValueError('period-2 contract SHA mismatch')
    for spec in specs:verify_prior_reference(Path(spec['root']),spec)
    previous=old_stage/PERIOD_NAME.format(2)
    if (previous/'output').exists() or (previous/'host-seal.json').exists():
        raise ValueError('period-2 source stage was already executed')
    if sha(previous/'source.py')!=contract['g3_simulator_source_sha256'] or \
            sha(previous/'input/g3-assignment.json')!=contract['assignment_sha256']:
        raise ValueError('period-2 staged simulator/assignment drift')
    root.mkdir(parents=True)
    for spec in specs:
        (root/PERIOD_NAME.format(spec['period'])).symlink_to(spec['root'],target_is_directory=True)
    verify_prior_pair(root,specs)
    batch=root/PERIOD_NAME.format(2)
    shutil.copytree(previous,batch,symlinks=False)
    shutil.copy2(repo/'scripts/eat_nav_p1_g3_crossover_v2_host.py',batch/'host-runner.py')
    (batch/'input/g3-contract.json').write_text((repo/'docs/eat-nav/p1-g3-crossover-v2-continuation-contract.json').read_text())
    if sha(batch/'host-runner.py')!=contract['g3_host_source_sha256']:
        raise ValueError('new host source SHA mismatch')
    old_ledger=json.loads((batch/'input-sha-ledger.json').read_text())
    ledger={name:sha(batch/name) for name in old_ledger}
    (batch/'input-sha-ledger.json').unlink()
    write_new(batch/'input-sha-ledger.json',ledger)
    shutil.copy2(repo/'docs/eat-nav/p1-g3-crossover-v2-continuation-contract.json',root/'manifest.json')
    shutil.copy2(repo/'docs/eat-nav/p1-g3-crossover-v2-assignment.json',root/'g3-assignment.json')
    write_new(root/'continuation-manifest.json',manifest)
    write_new(root/'stage-complete.json',{'schema':SCHEMA,
              'period2_input_ledger_sha256':sha(batch/'input-sha-ledger.json'),
              'period2_contract_sha256':sha(batch/'input/g3-contract.json'),
              'continuation_manifest_sha256':sha(root/'continuation-manifest.json'),
              'prior_seals':[spec['seal_sha256'] for spec in specs]})
    return {'root':str(root),'period2_input_ledger_sha256':sha(batch/'input-sha-ledger.json')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--stage-root',type=Path)
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--old-stage',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    contract_path=repo/'docs/eat-nav/p1-g3-crossover-v2-continuation-contract.json'
    manifest_path=repo/'docs/eat-nav/p1-g3-crossover-v2-continuation-manifest.json'
    if args.freeze:
        contract,manifest=make_freeze(repo,args.old_root)
        write_new(contract_path,contract)
        manifest['period2_manifest_sha256']=sha(contract_path)
        write_new(manifest_path,manifest)
    if args.stage_root is not None:
        contract=json.loads(contract_path.read_text());manifest=json.loads(manifest_path.read_text())
        stage_continuation(repo,args.old_stage,manifest['prior_periods'],contract,manifest,args.stage_root)


if __name__=='__main__':main()
