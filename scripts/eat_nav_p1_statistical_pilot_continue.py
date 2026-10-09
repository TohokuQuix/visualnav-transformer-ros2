"""Bounded, no-retry continuation of the six prepared limited pilot runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from visualnav_transformer.eat_nav.p1_statistical_freeze import sha


BASE=Path('/tmp/eatnav-p1-statistical-20261009')
SCRIPT=Path(__file__).resolve().parent/'eat_nav_p1_statistical_pilot_host.py'
ORDER=tuple((group,batch) for group in ('second-differential','first-graded','second-graded')
            for batch in (0,1))


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    tmp.replace(path)


def main():
    status=BASE/'pilot-continuation-status.json'
    if status.exists():
        raise FileExistsError('continuation is one-shot')
    started=time.time()
    completed=[]
    atomic(status,{'stage':'running','started_wall':started,'completed':completed,
                   'host_source_sha256':sha(SCRIPT),'schedule':ORDER})
    for group,batch in ORDER:
        root=BASE/f'pilot-{group}-b{batch}-v1'
        freeze=json.loads((root/'input/freeze.json').read_text())
        if (root/'host-seal.json').exists() or (root/'output').exists():
            raise FileExistsError('prepared one-shot root already used: '+str(root))
        if sha(root/'source.py')!=freeze['pilot_simulator_source_sha256']:
            raise ValueError('frozen source drift: '+str(root))
        command=[sys.executable,str(SCRIPT),'--root',str(root),
                 '--anchor-id',group,'--batch-index',str(batch)]
        atomic(status,{'stage':'running','started_wall':started,'completed':completed,
                       'current':str(root),'host_source_sha256':sha(SCRIPT),'schedule':ORDER})
        result=subprocess.run(command,env={**os.environ,'PYTHONPATH':str(SCRIPT.parents[1]/'src')},
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (root/'continuation-command.log').write_text(result.stdout)
        if result.returncode or not (root/'host-seal.json').is_file():
            atomic(status,{'stage':'stopped','started_wall':started,'completed':completed,
                           'failed_root':str(root),'host_exit':result.returncode,
                           'host_log_sha256':sha(root/'continuation-command.log')})
            return 1
        completed.append({'root':str(root),'seal_sha256':sha(root/'host-seal.json')})
        atomic(status,{'stage':'running','started_wall':started,'completed':completed,
                       'host_source_sha256':sha(SCRIPT),'schedule':ORDER})
        print(f'completed {len(completed)}/{len(ORDER)} {root}',flush=True)
    atomic(status,{'stage':'complete','started_wall':started,'finished_wall':time.time(),
                   'completed':completed,'host_source_sha256':sha(SCRIPT),'schedule':ORDER})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
