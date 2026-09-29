"""Run only the frozen coverage-003 nominal triplet with clock logging."""
from __future__ import annotations
import json,subprocess,time
from pathlib import Path
from eat_nav_clock_diagnostic_stage0 import OUT,OLD,atomic_json,approved_runs,sha

REPO=Path(__file__).resolve().parents[1]
RUNS=OUT/'clock-diagnostic-traces'
CONTAINER='/tmp/eatnav-candidate-coverage/clock-domain-diagnostic'
SIM=f'{CONTAINER}/clock_sim.py'
ENV='PYTHONPATH=/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'
DEADLINE_SEC=1800


def docker(*args):subprocess.run(['docker',*args],check=True)


def verify_frozen():
    audit=json.loads((OUT/'clock-source-audit.json').read_text())
    freeze=json.loads((OUT/'stage-d0-complete.json').read_text())
    assert sha(OUT/'clock-diagnostic-config.yaml')==freeze['config_sha256']
    assert sha(OUT/'clock-diagnostic-run-identities.json')==freeze['identities_sha256']
    assert sha(OUT/'clock-source-audit.json')==freeze['audit_sha256']
    for entry in audit['source_sha256'].values():assert sha(Path(entry['path']))==entry['sha256']
    for row in audit['original_trace_sha256']:
        assert sha(Path(row['summary']['path']))==row['summary']['sha256']
        assert sha(Path(row['trace']['path']))==row['trace']['sha256']
    fixtures=json.loads((OUT/'diagnostic-fixture-parity.json').read_text())
    assert fixtures['status']=='pass' and len(fixtures['fixtures'])==7
    assert all(x['pass'] for x in fixtures['fixtures'])
    exact=json.loads((OUT/'diagnostic-fixture-exact-parity.json').read_text())
    assert exact['status']=='pass' and len(exact['rows'])==7 and all(x['exact_equal'] for x in exact['rows'])
    identities=json.loads((OUT/'clock-diagnostic-run-identities.json').read_text())
    approved=approved_runs(json.loads((OLD/'long-horizon-run-identities.json').read_text()))
    assert [x['order'] for x in identities['runs']]==[x['order'] for x in approved]==[5,6,7]
    assert all(x['candidate_id']=='bank-nominal' and x['map_path_sha256']=='c15ccddd1be40f576289a0eb1a9a302c1088157503b87e06ad2a9291a6eeed43' for x in approved)
    return approved


def run_one(run,index,deadline):
    folder=RUNS/f'run-{index:03d}';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'run-complete.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        assert saved['frozen_run']==run
        for name,digest in saved['outputs_sha256'].items():assert sha(folder/name)==digest
        print('CLOCK_RESUME_SKIP',index,flush=True);return
    if any(folder.iterdir()):raise RuntimeError(f'partial run requires review; no retry: {folder}')
    idx='state-003';base='/tmp/eatnav-candidate-coverage/pilot-inputs/'+idx
    target=f'{CONTAINER}/runs/run-{index:03d}'
    docker('exec','isaac-lab-onix','mkdir','-p',target)
    command=['docker','exec','-e',ENV,'isaac-lab-onix','/isaac-sim/python.sh',SIM,'--headless',
        '--stage','live','--repeat-id','0','--manifest','/tmp/eatnav-extended/output/manifest.json',
        '--prefix',base+'/prefix-actions.npz',
        '--other-env-actions','/tmp/eatnav-candidate-coverage/command-preview-long-horizon/long-horizon-inputs/state-003/suffix-actions-120.npz',
        '--logged-anchor',base+'/logged-anchor.json',
        '--state','/tmp/eatnav-candidate-coverage/pilot-inputs/selected-states.json',
        '--base-path',base+'/bank-nominal-base-path.json',
        '--fixed-map-path',base+'/bank-nominal-map-path.json',
        '--config','/tmp/eatnav-candidate-coverage/command-preview-long-horizon/long-horizon-inputs/state-003/config-120.yaml',
        '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
        '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
        '--output',target]
    started=time.monotonic()
    log=folder/'runner.log'
    with log.open('x') as stream:
        stream.write(f'CLOCK_RUN_BEGIN order={index} original_order={run["order"]} system_ns={time.time_ns()} monotonic_ns={time.monotonic_ns()}\n');stream.flush()
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()))
        stream.write(f'CLOCK_RUN_END exit={result.returncode} system_ns={time.time_ns()} monotonic_ns={time.monotonic_ns()}\n');stream.flush()
    if result.returncode!=0:raise RuntimeError(f'run-{index:03d} failed; see {log}')
    for name in ('live-0.json','live-0.trace.json.gz','live-0.marker.json'):
        docker('cp',f'isaac-lab-onix:{target}/{name}',str(folder/name))
    summary=json.loads((folder/'live-0.json').read_text())
    assert summary['state']['state_id']=='coverage-003' and summary['suffix_steps']==120
    assert summary['fixed_map_path_sha256']==run['map_path_sha256']
    assert summary['source_sha256']['fixed_map_path']==run['map_path_sha256']
    outputs={name:sha(folder/name) for name in ('live-0.json','live-0.trace.json.gz','live-0.marker.json')}
    atomic_json(marker,{'frozen_run':run,'diagnostic_order':index,'outputs_sha256':outputs,
        'elapsed_sec':time.monotonic()-started,'runner_log_sha256':sha(log),
        'diagnostic_sim_source_sha256':sha(REPO/'scripts/eat_nav_clock_diagnostic_sim.py')})
    print('CLOCK_RUN_COMPLETE',index,round(time.monotonic()-started,1),flush=True)


def main():
    approved=verify_frozen()
    docker('exec','isaac-lab-onix','mkdir','-p',CONTAINER)
    docker('cp',str(REPO/'scripts/eat_nav_clock_diagnostic_sim.py'),f'isaac-lab-onix:{SIM}')
    RUNS.mkdir(parents=True,exist_ok=True)
    budget=OUT/'clock-run-budget.json'
    if budget.exists():b=json.loads(budget.read_text())
    else:
        b={'start_system_ns':time.time_ns(),'start_monotonic_ns':time.monotonic_ns(),
           'deadline_sec':DEADLINE_SEC,'run_identity_sha256':sha(OUT/'clock-diagnostic-run-identities.json'),
           'sim_source_sha256':sha(REPO/'scripts/eat_nav_clock_diagnostic_sim.py'),
           'bridge_source_sha256':sha(REPO/'scripts/eat_nav_clock_bridge.py')}
        atomic_json(budget,b)
    assert b['run_identity_sha256']==sha(OUT/'clock-diagnostic-run-identities.json')
    assert b['sim_source_sha256']==sha(REPO/'scripts/eat_nav_clock_diagnostic_sim.py')
    assert b['bridge_source_sha256']==sha(REPO/'scripts/eat_nav_clock_bridge.py')
    deadline=time.monotonic()+DEADLINE_SEC
    for i,run in enumerate(approved):
        if time.monotonic()>deadline:raise TimeoutError('clock diagnostic deadline before next run')
        atomic_json(OUT/f'stage-d2-run-{i:03d}-started.json',{'frozen_run':run,'sim_source_sha256':b['sim_source_sha256']})
        run_one(run,i,deadline)
        atomic_json(OUT/f'stage-d2-run-{i:03d}-complete.json',{'run_complete_sha256':sha(RUNS/f'run-{i:03d}'/'run-complete.json')})
    atomic_json(OUT/'stage-d2-complete.json',{'status':'complete','run_count':3,
        'run_marker_sha256':[sha(RUNS/f'run-{i:03d}'/'run-complete.json') for i in range(3)]})

if __name__=='__main__':main()
