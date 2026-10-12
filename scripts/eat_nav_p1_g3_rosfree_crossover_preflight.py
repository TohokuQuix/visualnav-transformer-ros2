"""Candidate-free prelaunch audit of the actual staged ROS-free host path."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_path_contract import (
    validate_host_bind,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_staged_attempt(batch, inspect, *, expected_bind=None):
    batch=Path(batch)
    root=batch.parent
    manifest=json.loads((root/'manifest.json').read_text())
    marker=json.loads((root/'stage-complete.json').read_text())
    if (manifest.get('runtime_authorized') is not True or
            manifest.get('authorized_process_ids') != [batch.name] or
            manifest.get('periods_1_2_authorized') is not False or
            sha(root/'manifest.json') != marker['manifest_sha256'] or
            sha(root/'g3-assignment.json') != marker['assignment_sha256'] or
            sha(batch/'input-sha-ledger.json') != marker['batch_input_ledger_sha256']):
        raise ValueError('runtime authorization or attempt identity mismatch')
    if (batch/'output').exists() or (batch/'host-seal.json').exists() or \
            (root/'host-seal.json').exists():
        raise ValueError('sealed or used batch root cannot be launched')
    repo=Path(__file__).resolve().parents[1]
    if manifest.get('preflight_source_sha256')!=sha(__file__):
        raise ValueError('preflight source SHA mismatch')
    host=batch/'host-runner.py'
    wrapper=host.with_name('eat_nav_p1_owner_wrapper.py')
    for path,expected,repository in (
            (host,manifest['staged_host_runner_sha256'],
             repo/'scripts/eat_nav_p1_g3_rosfree_crossover_host.py'),
            (wrapper,manifest['staged_owner_wrapper_sha256'],
             repo/'scripts/eat_nav_p1_owner_wrapper.py'),
            (batch/'source.py',manifest['g3_simulator_source_sha256'],
             repo/'scripts/eat_nav_p1_g3_rosfree_crossover_sim.py')):
        if not path.is_file() or sha(path)!=expected or sha(repository)!=expected:
            raise ValueError('staged/repository host, wrapper or simulator SHA mismatch: '+str(path))
    if sha(wrapper)!=manifest['owner_wrapper_sha256'] or \
            sha(host)!=manifest['g3_host_source_sha256']:
        raise ValueError('owner wrapper or host manifest SHA mismatch')
    ledger=json.loads((batch/'input-sha-ledger.json').read_text())
    for relative,expected in ledger.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or sha(batch/path)!=expected:
            raise ValueError('staged source/input ledger mismatch: '+relative)
    tree=ast.parse(wrapper.read_text())
    imports=[]
    for node in tree.body:
        if isinstance(node,ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom):
            imports.append(node.module or '')
    allowed={'__future__','argparse','ast','hashlib','json','os','pathlib','sys','time'}
    if not set(imports)<=allowed:
        raise ValueError('owner wrapper has unapproved prelaunch imports')
    spec=importlib.util.spec_from_file_location('staged_rosfree_crossover_host',host)
    staged=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staged)
    if Path(staged.__file__).with_name('eat_nav_p1_owner_wrapper.py')!=wrapper:
        raise ValueError('staged host __file__ wrapper resolution mismatch')
    class Args:
        anchor_id='first-differential'
        batch_index=0
        execution_index=0
    staged.validate_staged(batch,Args())
    remote=Path(manifest['attempt_container_root'])/batch.name
    bind=validate_host_bind(inspect,manifest,batch,remote,
                            expected_bind=expected_bind or
                            (manifest['host_bind_source'],manifest['container_bind_destination']))
    command=staged.build_wrapper_command(remote=str(remote),container_id='preflight',
        token='0'*64,source_sha=sha(batch/'source.py'),
        config_sha=sha(batch/'input/freeze.json'),anchor_id='first-differential',
        batch_index=0,execution_index=0)
    if (str(remote/'owner-wrapper.py') not in command or
            str(remote/'source.py') not in command or
            str(remote/'input/freeze.json') not in command):
        raise ValueError('owner wrapper dependency path mismatch')
    return {'pass':True,'batch':str(batch),'host_execution_path':str(host),
            'wrapper_resolves_from_staged___file__':True,
            'staged_host_sha256':sha(host),'staged_wrapper_sha256':sha(wrapper),
            'wrapper_prelaunch_imports':imports,'wrapper_runtime_dependencies':
            [str(remote/'source.py'),str(remote/'input/freeze.json'),
             str(remote/'owner-wrapper.py')],
            'source_input_file_count':len(ledger),
            'authorized_process_ids':manifest['authorized_process_ids'],
            'bind_mapping':bind}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    inspect=json.loads(subprocess.run(['docker','inspect','isaac-lab-onix'],
                                    capture_output=True,text=True,check=True).stdout)[0]
    if inspect['State']['Status']!='running':raise ValueError('Isaac container not running')
    audit=audit_staged_attempt(args.batch,inspect)
    manifest=json.loads((args.batch.parent/'manifest.json').read_text())
    remote=Path(manifest['attempt_container_root'])/args.batch.name
    for relative in ('host-runner.py','eat_nav_p1_owner_wrapper.py','source.py',
                     'input/freeze.json','input/g3-contract.json'):
        result=subprocess.run(['docker','exec','isaac-lab-onix','sha256sum',
                               str(remote/relative)],capture_output=True,text=True,check=True)
        if result.stdout.split()[0]!=sha(args.batch/relative):
            raise ValueError('host/container staged source SHA mismatch: '+relative)
    audit['container_source_sha_pass']=True
    audit['container_id']=inspect['Id']
    audit['manifest_sha256']=sha(args.batch.parent/'manifest.json')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(audit,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(audit,sort_keys=True))


if __name__=='__main__':main()
