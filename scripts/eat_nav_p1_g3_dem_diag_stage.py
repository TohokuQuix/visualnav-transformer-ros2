"""Prepare and run the single authorized candidate-free G3 DEM diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from eat_nav_p1_g1_lifecycle import verify_seal

BASE=Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1')
ROOT=BASE/'g3-dem-diagnostic-attempt-0'
REMOTE='/workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1/g3-dem-diagnostic-attempt-0/batch-0'
G1_SEAL='c5712eb6bc816f49222f01218bb593f116a38a00f8ce07eb79bd5b637e420609'
G2_SEAL='73ec7b6cac44dfdb717a8f8a8dd2a4cce4f19f9dc8d7b565df6055a0f251f1b0'
G3_SEAL='7bdf0a995fe70d99a8511b4dd6b45896e09256c31defef4070cf445a69cd597f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def verify_originals():
    for name,expected in (('g1-attempt-2',G1_SEAL),('g2-attempt-0',G2_SEAL),
                          ('g3-attempt-0/batch-0',G3_SEAL)):
        root=BASE/name
        if sha(root/'host-seal.json')!=expected:
            raise ValueError('original seal changed:'+name)
        verify_seal(root)


def prepare(repo: Path):
    verify_originals()
    if ROOT.exists():raise FileExistsError('DEM diagnostic root already exists')
    original=BASE/'g3-attempt-0/batch-0'
    source=repo/'scripts/eat_nav_p1_g3_pilot_sim.py'
    host=repo/'scripts/eat_nav_p1_g3_pilot_host.py'
    dem=repo/'src/visualnav_transformer/eat_nav/p1_dem_capture.py'
    protocol=repo/'src/visualnav_transformer/eat_nav/p1_longhorizon_protocol.py'
    for path in (source,host,dem,protocol):
        if not path.is_file():raise FileNotFoundError(path)
    ROOT.mkdir(mode=0o775)
    batch=ROOT/'batch-0';batch.mkdir()
    inputs=batch/'input';inputs.mkdir()
    for name in ('assignment.json','freeze.json','statistical-config.json',
                 'source-contract.json','anchor-fixtures.json','eat_nav.yaml',
                 'g3-assignment.json'):
        shutil.copy2(original/'input'/name,inputs/name)
        if sha(inputs/name)!=sha(original/'input'/name):
            raise ValueError('frozen input copy SHA mismatch:'+name)
    contract=json.loads((original/'input/g3-contract.json').read_text())
    contract['g3_simulator_source_sha256']=sha(source)
    contract['execution_mode']='candidate_free_20_step_dem'
    contract['original_g3_simulator_source_sha256']=sha(original/'source.py')
    contract['diagnostic_module_sha256']=sha(dem)
    contract['diagnostic_host_sha256']=sha(host)
    atomic(inputs/'g3-contract.json',contract)
    shutil.copy2(source,batch/'source.py')
    shutil.copy2(host,batch/'host-runner.py')
    shutil.copytree(BASE/'g2-attempt-0/src',batch/'src',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(protocol,batch/'src/visualnav_transformer/eat_nav/p1_longhorizon_protocol.py')
    shutil.copy2(dem,batch/'src/visualnav_transformer/eat_nav/p1_dem_capture.py')
    manifest={'schema':'p1_g3_candidate_free_dem_diagnostic_v1',
        'anchor_id':'first-differential','batch_index':0,'envs':4,
        'settling_control_steps':20,'capture_steps_zero_based':[4,9,14,19],
        'Path_publications_allowed':0,'retry_allowed':False,
        'source_sha256':sha(batch/'source.py'),'host_sha256':sha(batch/'host-runner.py'),
        'dem_module_sha256':sha(batch/'src/visualnav_transformer/eat_nav/p1_dem_capture.py'),
        'original_g3_seal_sha256':G3_SEAL,'original_g2_seal_sha256':G2_SEAL,
        'original_g1_seal_sha256':G1_SEAL,
        'frozen_inputs_sha256':{name:sha(path) for name,path in
            ((p.name,p) for p in sorted(inputs.iterdir()))}}
    atomic(ROOT/'diagnostic-manifest.json',manifest)
    atomic(ROOT/'status.json',{'stage':'prepared','manifest_sha256':sha(ROOT/'diagnostic-manifest.json')})
    return manifest


def run(repo: Path):
    verify_originals()
    status=json.loads((ROOT/'status.json').read_text())
    if status['stage']!='prepared' or (ROOT/'batch-0/output').exists() or \
            (ROOT/'batch-0/host-seal.json').exists():
        raise ValueError('diagnostic already launched; no retry')
    manifest=json.loads((ROOT/'diagnostic-manifest.json').read_text())
    if sha(ROOT/'diagnostic-manifest.json')!=status['manifest_sha256']:
        raise ValueError('diagnostic manifest SHA drift')
    batch=ROOT/'batch-0'
    for name,key in (('source.py','source_sha256'),('host-runner.py','host_sha256'),
                     ('src/visualnav_transformer/eat_nav/p1_dem_capture.py','dem_module_sha256')):
        if sha(batch/name)!=manifest[key]:raise ValueError('diagnostic source SHA drift:'+name)
    host=repo/'scripts/eat_nav_p1_g3_pilot_host.py'
    if sha(host)!=manifest['host_sha256']:
        raise ValueError('installed diagnostic host SHA drift')
    for name,expected in manifest['frozen_inputs_sha256'].items():
        if sha(batch/'input'/name)!=expected:raise ValueError('diagnostic input SHA drift:'+name)
    atomic(ROOT/'status.json',{'stage':'running','manifest_sha256':status['manifest_sha256']})
    command=[sys.executable,str(host),'--root',str(batch),
             '--remote',REMOTE,'--anchor-id','first-differential','--batch-index','0',
             '--dem-diagnostic']
    with (ROOT/'host.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONPATH':str(repo/'src')+':'+str(repo/'scripts')})
        log.flush();os.fsync(log.fileno())
    seal=verify_seal(batch) if (batch/'host-seal.json').is_file() else None
    host_status=json.loads((batch/'g3-host-status.json').read_text()) if \
        (batch/'g3-host-status.json').is_file() else None
    atomic(ROOT/'status.json',{'stage':'finished','manifest_sha256':status['manifest_sha256'],
        'host_exit':result.returncode,'host_status':host_status,'seal_validation':seal,
        'seal_sha256':sha(batch/'host-seal.json') if seal is not None else None})
    return result.returncode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','run'))
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if args.stage=='prepare':
        print(json.dumps(prepare(repo),sort_keys=True))
    else:
        raise SystemExit(run(repo))


if __name__=='__main__':main()
