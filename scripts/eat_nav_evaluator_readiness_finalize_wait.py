"""Wait detached for postprocessing, then generate the small final report."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

ROOT=Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot')
REPO=Path('/tmp/eatnav-vnt-candidate-coverage')

def main():
    while True:
        path=ROOT/'postprocess-status.json'
        if path.exists():
            stage=json.loads(path.read_text())['stage']
            if stage=='COMPLETE':break
            if stage in ('STOPPED','BLOCKED_BY_RAW_GATE'):
                print('READINESS_FINALIZE_BLOCKED',stage,flush=True)
                return
        time.sleep(60)
    with (ROOT/'finalize-runner.log').open('a') as log:
        result=subprocess.run(['/usr/bin/python3','scripts/eat_nav_evaluator_readiness_finalize.py'],
                              cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('finalizer failed; see finalize-runner.log')
    print('READINESS_FINALIZE_COMPLETE',flush=True)

if __name__=='__main__':main()
