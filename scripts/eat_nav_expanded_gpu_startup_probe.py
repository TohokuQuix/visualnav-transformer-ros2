"""One candidate-free simulator startup probe for the repaired E1 GPU monitor."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from eat_nav_ros_subprocess_env import build_container_env, docker_pythonpath_arg, env_audit
from eat_nav_e1_monitor import sample, gate_sample, limits
from eat_nav_g0e_formal_gate import runtime_gate
from eat_nav_g0e_gpu_probe import Nvml, atomic, proc_table, sha
from eat_nav_g0e_gpu_run import container_identity
import yaml

ROOT = Path('/tmp/eatnav-expanded-feasibility-gpu-startup-probe-unified-20261005')
SRC = Path('/tmp/eatnav-g0e-gpu-attribution/diag_sim.py')
INPUT = Path('/tmp/eatnav-candidate-bank-g0e/inputs')
OLD_CONFIG = Path('/tmp/eatnav-g0e-gpu-attribution/gpu-attribution-config.yaml')
REMOTE = str(ROOT)
SOURCE = f'{REMOTE}/diag_sim.py'
OUTPUT = f'{REMOTE}/run'
CONTAINER = 'isaac-lab-onix'


def docker(*args, check=True):
    return subprocess.run(['docker', *args], capture_output=True, text=True, check=check)


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def build_probe_plan():
    files = {SRC: Path(SOURCE), INPUT/'selected-states.json':ROOT/'inputs/selected-states.json'}
    for name in ('prefix-actions.npz','logged-anchor.json','anchor-config.yaml'):
        files[INPUT/'state-000'/name] = ROOT/'inputs'/'state-000'/name
    container_env = build_container_env(repo=REPO,extra_paths=(
        ROOT,'/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src',
        '/tmp/eatnav-train','/tmp/eatnav-train/src',
        '/tmp/eatnav-train/upstream-src','/tmp/eatnav-extended/isaac-source'))
    env = docker_pythonpath_arg(container_env)
    command = ['docker','exec','-e',env,CONTAINER,'/isaac-sim/python.sh',SOURCE,
        '--headless','--stage','anchor','--repeat-id','0','--logical-t0-ns','2300000000000',
        '--manifest','/tmp/eatnav-extended/output/manifest.json',
        '--prefix',REMOTE+'/inputs/state-000/prefix-actions.npz',
        '--logged-anchor',REMOTE+'/inputs/state-000/logged-anchor.json',
        '--state',REMOTE+'/inputs/selected-states.json',
        '--config',REMOTE+'/inputs/state-000/anchor-config.yaml',
        '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
        '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
        '--output',OUTPUT,'--timing-telemetry']
    return {'root':ROOT,'output':OUTPUT,'files':files,'env':env,'command':command,
        'container_env_audit':env_audit(container_env,'/isaac-sim/python.sh'),
        'monitor_source_sha256':sha(REPO/'scripts/eat_nav_e1_monitor.py'),
        'nvml_source_sha256':sha(REPO/'scripts/eat_nav_g0e_gpu_probe.py'),
        'candidate_path_published':False,'outcome_collected':False}


def finish_probe_process(process, stop, owned, identity, *, docker_fn=docker,
                         table_fn=proc_table):
    """Allow native shutdown after step one; signal only a stalled owned process."""
    if stop == 'first_control_step_complete':
        try:
            process.wait(timeout=30)
            return {'native_exit':process.returncode == 0,
                    'cleanup_timeout':False,'exit_code':process.returncode}
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        for pid,tick in sorted(owned,reverse=True):
            current=table_fn().get(pid)
            if current and current['start_ticks']==tick and identity['container_id'] in current['cgroup']:
                docker_fn('exec',CONTAINER,'kill','-TERM',str(current['container_pid']),check=False)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    return {'native_exit':False,'cleanup_timeout':stop == 'first_control_step_complete',
            'exit_code':process.returncode}


def main():
    plan = build_probe_plan()
    if ROOT.exists():
        raise FileExistsError('single probe root exists; no retry')
    ROOT.mkdir()
    previous = yaml.safe_load(OLD_CONFIG.read_text())
    safety = yaml.safe_load(Path('/tmp/eatnav-expanded-feasibility-x2-retry-20261005/e1-safety-stops.yaml').read_text())
    for path, digest in previous['container_frozen_sha256'].items():
        if docker('exec', CONTAINER, 'sha256sum', path).stdout.split()[0] != digest:
            raise ValueError('container frozen SHA drift: '+path)
    for path in (SRC, INPUT/'selected-states.json', INPUT/'state-000/prefix-actions.npz',
                 INPUT/'state-000/logged-anchor.json', INPUT/'state-000/anchor-config.yaml'):
        if sha(path) != previous['frozen_sha256'][str(path)]:
            raise ValueError('candidate-free input SHA drift: '+str(path))
    files = plan['files']
    for host, remote in files.items():
        docker('exec',CONTAINER,'mkdir','-p',str(remote.parent))
        docker('cp',str(host),f'{CONTAINER}:{remote}')
        if docker('exec',CONTAINER,'sha256sum',str(remote)).stdout.split()[0] != sha(host):
            raise ValueError('probe input copy SHA drift')
    docker('exec',CONTAINER,'mkdir','-p',OUTPUT)
    atomic(ROOT/'input-ledger.json', {'source':str(SRC), 'source_sha256':sha(SRC),
        'input_sha256':{str(p):sha(p) for p in files}, 'candidate_path_published':False,
        'outcome_collected':False, 'monitor_source_sha256':plan['monitor_source_sha256'],
        'nvml_source_sha256':plan['nvml_source_sha256']})
    identity = container_identity(); nvml = Nvml()
    table = proc_table(); baseline = {(pid,p['start_ticks']) for pid,p in table.items()}
    baseline_oom = 0
    start_baseline = time.monotonic()
    while time.monotonic() - start_baseline < 3:
        snap = sample(nvml,identity,baseline,OUTPUT,SOURCE)
        baseline_oom = snap['cgroup_memory']['events'].get('oom',0)
        append(ROOT/'baseline.jsonl',snap)
        reason = runtime_gate(gate_sample(snap,baseline_oom),limits(safety))
        if reason:raise RuntimeError('prelaunch resource '+reason)
        time.sleep(.4)
    command = plan['command']
    atomic(ROOT/'started.json',{'command':command,'candidate_free':True,'wall_ns':time.time_ns()})
    log = (ROOT/'simulator.log').open('x')
    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    started = time.monotonic(); owned=set(); stop=None; events=[]
    try:
        while time.monotonic()-started < 150:
            snap=sample(nvml,identity,baseline,OUTPUT,SOURCE)
            append(ROOT/'gpu-process-timeseries.jsonl',snap)
            owned.update((p['host_pid'],p['start_ticks']) for p in snap['simulator_tree'])
            reason=runtime_gate(gate_sample(snap,baseline_oom),limits(safety))
            if reason:
                stop=reason;break
            result=docker('exec',CONTAINER,'cat',OUTPUT+'/timing-events.jsonl',check=False)
            if result.returncode==0:
                events=[json.loads(x) for x in result.stdout.splitlines() if x.strip()]
            if any(x['event']=='first_control_step_complete' for x in events):
                stop='first_control_step_complete';break
            if process.poll() is not None:
                stop='simulator_exited_before_first_step';break
            time.sleep(.4)
        else:stop='startup_timeout'
    finally:
        cleanup=finish_probe_process(process,stop,owned,identity)
        log.flush();os.fsync(log.fileno());log.close()
        docker('cp',f'{CONTAINER}:{OUTPUT}/.',str(ROOT/'run'),check=False)
        residual=[]
        for _ in range(25):
            table=proc_table()
            residual=[pid for pid,tick in owned if (p:=table.get(pid)) and p['start_ticks']==tick]
            if not residual:break
            time.sleep(.2)
        atomic(ROOT/'complete.json',{'stop':stop,'simulator_exit_code':process.returncode,
            'owned_residual':residual,'first_control_step_complete':any(
                x['event']=='first_control_step_complete' for x in events),
            'cleanup':cleanup,
            'candidate_path_published':False,'outcome_collected':False,
            'sample_count':len((ROOT/'gpu-process-timeseries.jsonl').read_text().splitlines()),
            'timeseries_sha256':sha(ROOT/'gpu-process-timeseries.jsonl'),
            'simulator_log_sha256':sha(ROOT/'simulator.log')})
    print('PROBE',stop,'exit',process.returncode,'residual',residual,flush=True)
    if stop!='first_control_step_complete' or not cleanup['native_exit'] or residual:
        raise RuntimeError('candidate-free startup probe failed')


if __name__=='__main__':main()
