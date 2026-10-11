"""Keep host console/exit evidence outside a sealed material-diagnostic batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from eat_nav_p1_g1_lifecycle import verify_seal


def _durable(path: Path, value: dict) -> None:
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def run_outer(command: list[str], batch_root: Path, external_log_dir: Path) -> dict:
    batch_root=batch_root.resolve();external_log_dir=external_log_dir.resolve()
    if external_log_dir.is_relative_to(batch_root) or batch_root.is_relative_to(external_log_dir):
        raise ValueError('outer log directory must be separate from sealed batch')
    external_log_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with (external_log_dir/'host-console.log').open('xb') as console:
        completed=subprocess.run(command,stdout=console,stderr=subprocess.STDOUT,check=False)
        console.flush();os.fsync(console.fileno())
    try:
        verification=verify_seal(batch_root)
        seal_error=None
    except (FileNotFoundError,ValueError,KeyError,TypeError) as exc:
        verification=None;seal_error=type(exc).__name__+':'+str(exc)
    result={'host_exit_code':completed.returncode,'post_launcher_seal_verification':verification,
            'post_launcher_seal_error':seal_error,'elapsed_s':time.monotonic()-started,
            'batch_root':str(batch_root),'console_log':str(external_log_dir/'host-console.log')}
    _durable(external_log_dir/'launcher-result.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--external-log-dir',type=Path,required=True)
    args=parser.parse_args()
    contract=json.loads((args.root/'input/material-diagnostic-contract.json').read_text())
    actual=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if actual!=contract['outer_source_sha256']:
        raise ValueError('outer launcher source SHA drift')
    host=Path(__file__).with_name('eat_nav_p1_g3_material_diag_host.py')
    result=run_outer(['/usr/bin/python3',str(host),'--root',str(args.root),
                      '--remote',args.remote],args.root,args.external_log_dir)
    if result['host_exit_code']!=0 or result['post_launcher_seal_error'] is not None:
        raise RuntimeError('material diagnostic host/launcher failed: '+str(result))
    print('MATERIAL_DIAGNOSTIC_OUTER_PASS',result['batch_root'])


if __name__=='__main__':main()
