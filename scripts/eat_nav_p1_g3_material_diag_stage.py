"""Stage one immutable candidate-free material diagnostic under a fresh root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


OLD=Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v1-pilot-attempt-0')
INPUT_NAMES=('assignment.json','statistical-config.json','source-contract.json',
             'anchor-fixtures.json','eat_nav.yaml')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def stage(repo: Path, old: Path, target: Path) -> dict:
    if target.exists():raise FileExistsError('diagnostic root already exists')
    reference=old/'batch-first-differential-0-period-0'
    old_manifest=json.loads((old/'manifest.json').read_text())
    if old_manifest['schema']!='eatnav_p1_g3_path_crossover_v1' or \
            sha(old/'manifest.json')!='6a87ec0f05fb334b7968435d6c7db871ebe0ed2c5786ecb2d2be616bcba80162':
        raise ValueError('frozen original manifest changed')
    if json.loads((reference/'host-seal.json').read_text())['stage']!= \
            'G3_crossover_batch_failed_or_inconclusive':
        raise ValueError('expected immutable failed predecessor missing')
    target.mkdir(parents=True)
    (target/'input').mkdir()
    shutil.copytree(reference/'src',target/'src',ignore=shutil.ignore_patterns('__pycache__'))
    module=target/'src/visualnav_transformer/eat_nav/p1_g3_material_diagnostic.py'
    shutil.copy2(repo/'src/visualnav_transformer/eat_nav/p1_g3_material_diagnostic.py',module)
    for source_name,target_name in (
            ('eat_nav_p1_g3_material_diag_sim.py','source.py'),
            ('eat_nav_p1_g3_material_diag_host.py','host-runner.py'),
            ('eat_nav_p1_g3_material_diag_outer.py','outer-runner.py')):
        shutil.copy2(repo/'scripts'/source_name,target/target_name)
    for name in INPUT_NAMES:
        source=reference/'input'/name
        if sha(source)!=old_manifest['source_input_sha256'][name]:
            raise ValueError('frozen diagnostic input changed: '+name)
        shutil.copy2(source,target/'input'/name)
    freeze=json.loads((reference/'input/freeze.json').read_text())
    if sha(reference/'input/freeze.json')!=old_manifest['source_input_sha256']['freeze.json']:
        raise ValueError('frozen source freeze changed')
    freeze['pilot_simulator_source_sha256']=sha(target/'source.py')
    freeze['material_diagnostic_derivation']={'schema':'eatnav_p1_g3_material_diagnostic_v1',
        'predecessor_freeze_sha256':sha(reference/'input/freeze.json'),
        'Path_publications':0,'physics_control_steps':0}
    write_new(target/'input/freeze.json',freeze)
    contract={'schema':'eatnav_p1_g3_material_diagnostic_v1','Path_publications':0,
        'simulator_source_sha256':sha(target/'source.py'),
        'host_source_sha256':sha(target/'host-runner.py'),
        'outer_source_sha256':sha(target/'outer-runner.py'),
        'material_module_sha256':sha(module),
        'rl_checkpoint_sha256':old_manifest['rl_checkpoint_sha256'],
        'predecessor_manifest_sha256':sha(old/'manifest.json'),
        'predecessor_failure_seal_sha256':sha(reference/'host-seal.json'),
        'input_sha256':{name:sha(target/'input'/name)
                        for name in (*INPUT_NAMES,'freeze.json')},
        'frozen_resource_caps':old_manifest['resource_cap']}
    write_new(target/'input/material-diagnostic-contract.json',contract)
    ledger={str(path.relative_to(target)):sha(path) for path in sorted(target.rglob('*'))
            if path.is_file()}
    write_new(target/'stage-complete.json',{'schema':contract['schema'],
              'contract_sha256':sha(target/'input/material-diagnostic-contract.json'),
              'file_sha256':ledger})
    return {'file_count':len(ledger),'contract_sha256':sha(target/'input/material-diagnostic-contract.json')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old-root',type=Path,default=OLD)
    args=parser.parse_args()
    print(stage(Path(__file__).resolve().parents[1],args.old_root,args.root))


if __name__=='__main__':main()
