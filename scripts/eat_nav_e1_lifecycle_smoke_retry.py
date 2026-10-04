"""One-shot E1-source lifecycle smoke on unchanged readiness physical inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import eat_nav_e1_run_one as replay
from eat_nav_e1_monitor import verify_complete
from eat_nav_e1_smoke_fixture import validate_fixture


ROOT = Path('/tmp/eatnav-e1-lifecycle-compatible-20261004')
REPO = Path('/tmp/eatnav-vnt-e1-feasibility')
SOURCE = REPO/'scripts/eat_nav_e1_sim.py'
FROZEN_SOURCE_SHA = '9e2a40f0b283d4581219302c3f2f0782b00ca825e07653bbe3924dd5c47cec07'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(root, entry, production_source_sha):
    folder = validate_fixture(root, entry)
    if sha(SOURCE) != production_source_sha:
        raise ValueError('production simulator source SHA changed')
    if production_source_sha != FROZEN_SOURCE_SHA:
        raise ValueError('production simulator source SHA differs from frozen E1')
    return {'status':'pass','state_dir':str(folder),'production_source_sha256':production_source_sha,
            'physical_readiness_state_unchanged':True}


def docker(*args):
    return subprocess.run(['docker',*args],check=True,capture_output=True,text=True)


def prepare_remote(root):
    root = Path(root)
    remote = str(root)
    original = '/tmp/eatnav-e1-feasibility-20261003/eat_nav_e1_sim.py'
    if docker('exec','isaac-lab-onix','sha256sum',original).stdout.split()[0] != FROZEN_SOURCE_SHA:
        raise ValueError('frozen Docker E1 simulator source SHA differs')
    docker('exec','isaac-lab-onix','mkdir','-p',remote)
    for name in ('eat_nav_e1_sim.py','eat_nav_e1_anchor_gate.py',
                 'eat_nav_g0e_formal_gate.py','eat_nav_goal_subgoal_dem_gate.py'):
        local = REPO/'scripts'/name
        target = remote+'/'+name
        docker('cp',str(local),'isaac-lab-onix:'+target)
        if docker('exec','isaac-lab-onix','sha256sum',target).stdout.split()[0] != sha(local):
            raise ValueError('diagnostic Docker source SHA differs '+name)
    if subprocess.run(['docker','exec','isaac-lab-onix','test','-e',remote+'/inputs'],
                      capture_output=True).returncode == 0:
        raise FileExistsError('one-shot Docker diagnostic inputs already exist')
    docker('cp',str(root/'inputs'),'isaac-lab-onix:'+remote+'/inputs')
    for local in (root/'inputs').rglob('*'):
        if local.is_file():
            target = remote+'/'+str(local.relative_to(root))
            if docker('exec','isaac-lab-onix','sha256sum',target).stdout.split()[0] != sha(local):
                raise ValueError('diagnostic Docker input SHA differs '+str(local))


def main():
    root = ROOT
    smoke = root/'smoke'
    entry = json.loads((root/'diagnostic-entry.json').read_text())
    if (smoke/'runs'/entry['run_id']).exists():
        raise FileExistsError('one-shot E1-source smoke already attempted')
    audit = preflight(root,entry,FROZEN_SOURCE_SHA)
    readiness = json.loads((root/'router-readiness.json').read_text())
    if readiness['gate']['status'] != 'pass':
        raise ValueError('external Zenoh readiness did not pass')
    prepare_remote(root)
    replay.DOC = root
    replay.ROOT = smoke
    replay.INPUT = root/'inputs'
    replay.REMOTE_CODE = str(root/'eat_nav_e1_sim.py')
    replay.REMOTE_ROOT = str(smoke/'runs')
    replay.REMOTE_INPUT_ROOT = root
    replay.frozen = lambda given: validate_fixture(root,given)
    result = replay.run_one(entry,entry['run_id'],timeout_sec=600,timing_telemetry=True,
        timing_monitor_script=REPO/'scripts/eat_nav_e1_monitor.py',
        timing_monitor_config=root/'e1-safety-stops.yaml',
        timing_monitor_extra_args=('--sim-source',replay.REMOTE_CODE),
        monitor_cleanup_protocol=True)
    run = smoke/'runs'/entry['run_id']
    digest = verify_complete(run)
    runtime = json.loads((run/'runtime-monitor-result.json').read_text())
    transitions = [row['to'] for row in runtime['transitions']]
    if transitions != ['RUNNING','FINALIZING','COMPLETE']:
        raise ValueError('lifecycle transitions differ')
    if json.loads((run/'protocol-valid.json').read_text())['status'] != 'valid':
        raise ValueError('smoke protocol-valid marker missing')
    records = [json.loads(line) for line in (run/'timing-progress-timeseries.jsonl').read_text().splitlines()]
    prefix = [row for row in records if row['phase']=='prefix']
    suffix = [row for row in records if row['phase']=='suffix']
    final_progress = json.loads((run/'timing-progress.json').read_text())
    max_gap = max((b['sample_monotonic_ns']-a['sample_monotonic_ns'])/1e9 for a,b in zip(records,records[1:]))
    if (not prefix or not suffix or max_gap >= 90 or
            final_progress['last_completed_step'] != entry['anchor_step']+120):
        raise ValueError('prefix/suffix heartbeat coverage failed')
    marker = {'status':'pass','E1_candidate':False,'original_run_id':'readiness-002',
              'physical_state_unchanged':True,'run_complete_sha256':digest,
              'runtime_monitor_sha256':sha(run/'runtime-monitor-result.json'),
              'finalization_verifier_sha256':sha(run/'finalization-verifier-result.json'),
              'protocol_valid_sha256':sha(run/'protocol-valid.json'),
              'heartbeat_prefix_samples':len(prefix),'heartbeat_suffix_samples':len(suffix),
              'max_sampled_heartbeat_gap_sec':max_gap,'preflight':audit,'result':result}
    replay.atomic(smoke/'lifecycle-smoke-result.json',marker)
    print(json.dumps(marker,sort_keys=True),flush=True)


if __name__=='__main__':main()
