"""One non-E1 readiness-002 replay through the E1 cleanup lifecycle monitor."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import eat_nav_fresh_bridge_run as fresh
from eat_nav_e1_monitor import verify_complete


ROOT = Path('/tmp/eatnav-e1-cleanup-fixed-20261004')
SMOKE = ROOT/'smoke'
SOURCE = Path('/tmp/eatnav-vnt-goal-ood-simulator/scripts/eat_nav_goal_subgoal_sim.py')
REMOTE_SOURCE = str(SMOKE/'readiness-002-sim.py')


def main():
    if (SMOKE/'runs/lifecycle-readiness-002').exists():
        raise FileExistsError('one-shot lifecycle smoke already attempted')
    SMOKE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile('/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/r0-freeze.json',
                    SMOKE/'r0-freeze.json')
    subprocess.run(['docker','exec','isaac-lab-onix','mkdir','-p',str(SMOKE)],check=True)
    subprocess.run(['docker','cp',str(SOURCE),f'isaac-lab-onix:{REMOTE_SOURCE}'],check=True)
    if subprocess.check_output(['docker','exec','isaac-lab-onix','sha256sum',REMOTE_SOURCE],text=True).split()[0] != fresh.sha(SOURCE):
        raise ValueError('smoke simulator source SHA mismatch')
    fresh.REPO = Path('/tmp/eatnav-vnt-goal-ood-simulator')
    fresh.ROOT = SMOKE
    fresh.REMOTE_CODE = REMOTE_SOURCE
    fresh.REMOTE_ROOT = str(SMOKE/'runs')
    entry = fresh.read(fresh.DOC/'schedule.json')['runs'][2]
    if entry['run_id'] != 'readiness-002' or entry['control_steps'] != 120:
        raise ValueError('readiness fixture identity changed')
    result = fresh.run_one(entry,'lifecycle-readiness-002',timeout_sec=600,
        lifecycle_monitor=Path(__file__).with_name('eat_nav_e1_monitor.py'),
        lifecycle_config=ROOT/'e1-safety-stops.yaml',lifecycle_sim_source=REMOTE_SOURCE)
    run = SMOKE/'runs/lifecycle-readiness-002'
    digest = verify_complete(run)
    fresh.atomic(SMOKE/'lifecycle-smoke-result.json',{'status':'pass','fixture':'readiness-002',
        'E1_candidate':False,'sim_source_sha256':fresh.sha(SOURCE),
        'run_complete_sha256':digest,'runtime_monitor_sha256':fresh.sha(run/'runtime-monitor-result.json'),
        'finalization_verifier_sha256':fresh.sha(run/'finalization-verifier-result.json'),
        'result':result})
    print(json.dumps({'status':'pass','run_complete_sha256':digest}),flush=True)


if __name__ == '__main__':
    main()
