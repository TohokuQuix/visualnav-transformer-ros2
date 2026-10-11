"""Candidate-free G3 material diagnostic host with G1 GPU ownership gates."""
import argparse
import hashlib
import json
import os
import pathlib
import secrets
import shutil
import signal
import subprocess
import time
from eat_nav_g0e_gpu_probe import Nvml, proc_info, proc_table
from eat_nav_p1_gpu_attribution import capture_live
from eat_nav_p1_longhorizon_gpu import capture_sample
from eat_nav_p1_owner_host import initial_graphics_identities
from eat_nav_p1_g1_lifecycle import (
    G1Lifecycle, diagnose_owner_resolution, residual_identities,
    seal_after_writers, verify_seal,
)

CONTAINER = 'isaac-lab-onix'
ROOT = None
REMOTE = None
BASE = ':'.join(('/tmp/eatnav-train/src', '/tmp/eatnav-extended/isaac-source',
                 '/workspace/isaaclab/source/isaaclab', '/workspace/isaaclab/source/isaaclab_rl',
                 '/workspace/isaaclab/source/isaaclab_tasks',
                 '/workspace/isaaclab_onix/source/onix_isaac_lab'))


def build_wrapper_command(*, remote: str, container_id: str, token: str,
                          source_sha: str, config_sha: str) -> list[str]:
    """Keep the simulator paused until its exact host PID is monitored."""
    return ['docker','exec','-e',f'PYTHONPATH={remote}/src:{BASE}',CONTAINER,
            'timeout','600s','/isaac-sim/python.sh',f'{remote}/owner-wrapper.py',
            '--coord',f'{remote}/coord','--run-token',token,
            '--container-id',container_id,'--source',f'{remote}/source.py',
            '--source-sha',source_sha,'--config',f'{remote}/input/freeze.json',
            '--config-sha',config_sha,'--output',f'{remote}/output','--mode','smoke',
            '--','--headless','--input',f'{remote}/input',
            '--output',f'{remote}/output','--anchor-id','first-differential',
            '--batch-index','0','--gate1-probe']


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


GPU_UUID = 'GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26'
GPU_CONTRACT = {'required_gpu_uuid': GPU_UUID,
                'required_total_memory_mib_min': 16000,
                'gpu_device_total_hard_cap_mib': 7680,
                'bracket_nonatomic_sampling_gap_hard_cap_mib': 256,
                'absolute_attribution_gap_hard_cap_mib': 256,
                'unresolved_gpu_bytes_hard_cap': 0}


def host_available():
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')


def anon_shmem():
    raw = command(['docker', 'exec', CONTAINER, 'cat', '/sys/fs/cgroup/memory.stat'])
    stats = {row.split()[0]: int(row.split()[1]) for row in raw.splitlines()}
    return stats['anon'] + stats['shmem']


def root_bytes():
    return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())


def resource_row(started, gpu):
    # Keep the bracket evidence small enough to stay inside the frozen storage cap.
    audit = {key: gpu[key] for key in ('pass','failures','gpu_uuid','device_before_bytes',
             'device_after_bytes','device_total_bytes','processes','owned_gpu_pids',
             'bracket_gap_bytes','process_sum_bytes','resolved_owned','wall_ns',
             'sample_finished_monotonic_ns')}
    return {'elapsed_s': time.monotonic() - started,
            'gpu_device_total_mib': max(gpu['device_before_bytes'],gpu['device_after_bytes'])/1048576,
            'gpu_attribution': audit,
            'host_available_bytes': host_available(), 'container_anon_plus_shmem_bytes': anon_shmem(),
            'persistent_root_bytes': root_bytes()}


def sample(started, nvml, owned, baseline, container_id):
    gpu = capture_sample(nvml, owned, baseline, GPU_CONTRACT,
                         container_id=container_id, require_owned_gpu=False)
    return gpu, resource_row(started, gpu)


def validate_sample(row):
    if not row['gpu_attribution']['pass']: return 'gpu_attribution:' + ','.join(row['gpu_attribution']['failures'])
    if row['gpu_device_total_mib'] > 7680: return 'gpu_device_total_cap'
    if row['host_available_bytes'] < 10095421850: return 'host_available_floor'
    if row['container_anon_plus_shmem_bytes'] > 6442450944: return 'container_anon_shmem_cap'
    if row['persistent_root_bytes'] > 1000000000: return 'persistent_storage_cap'
    return None


def append_jsonl(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    stream.flush(); os.fsync(stream.fileno())


def exact_owner_alive(owner):
    if owner is None:return False
    row=proc_info(owner['host_pid'])
    return bool(row and row.get('start_ticks')==owner['start_ticks'])


def stop_exact_owner(owner, evidence):
    if not exact_owner_alive(owner):return
    os.kill(owner['host_pid'],signal.SIGTERM)
    evidence.append({'signal':'SIGTERM','host_pid':owner['host_pid'],
                     'start_ticks':owner['start_ticks']})
    for _ in range(50):
        if not exact_owner_alive(owner):return
        time.sleep(.1)
    if exact_owner_alive(owner):
        os.kill(owner['host_pid'],signal.SIGKILL)
        evidence.append({'signal':'SIGKILL','host_pid':owner['host_pid'],
                         'start_ticks':owner['start_ticks']})


def main():
    global ROOT, REMOTE
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--remote', required=True)
    arguments=parser.parse_args()
    ROOT=arguments.root.resolve(); REMOTE=arguments.remote
    if ROOT.name == 'g1-attempt-0':
        raise ValueError('G1 attempt-0 is immutable')
    if (ROOT / 'output').exists() or (ROOT / 'host-seal.json').exists():
        raise FileExistsError('G1 output already exists')
    freeze = json.loads((ROOT / 'input/freeze.json').read_text())
    diagnostic_contract=json.loads((ROOT/'input/material-diagnostic-contract.json').read_text())
    assert digest(ROOT / 'source.py') == freeze['pilot_simulator_source_sha256']
    if diagnostic_contract['schema']!='eatnav_p1_g3_material_diagnostic_v1' or \
            digest(__file__)!=diagnostic_contract['host_source_sha256'] or \
            digest(ROOT/'host-runner.py')!=diagnostic_contract['host_source_sha256'] or \
            digest(ROOT/'source.py')!=diagnostic_contract['simulator_source_sha256'] or \
            digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_material_diagnostic.py')!= \
              diagnostic_contract['material_module_sha256'] or \
            diagnostic_contract['Path_publications']!=0:
        raise ValueError('material diagnostic host/contract drift')
    staged=json.loads((ROOT/'stage-complete.json').read_text())
    if digest(ROOT/'input/material-diagnostic-contract.json')!=staged['contract_sha256'] or \
            any(digest(ROOT/name)!=expected for name,expected in staged['file_sha256'].items()):
        raise ValueError('material diagnostic staged input SHA drift')
    inspect = json.loads(command(['docker', 'inspect', CONTAINER]))[0]
    if inspect['State']['Status'] != 'running':
        raise RuntimeError('container not running')
    if subprocess.run(['docker', 'exec', CONTAINER, 'test', '-d', REMOTE]).returncode != 0:
        raise RuntimeError('host-mounted G1 root not visible')
    host_free = shutil.disk_usage(ROOT).free
    nvml=Nvml()
    initial_devices, initial_rows=nvml.sample()
    initial_table=proc_table()
    baseline_identities=initial_graphics_identities(initial_rows, initial_table)
    # Baseline is sampled before launch, with no expected simulator owner.
    baseline_gpu=capture_live(nvml, {}, baseline_identities, GPU_CONTRACT,
                              container_id=inspect['Id'], expected_simulator_alive=False,
                              device_cap_mib=1024)
    if not baseline_gpu['pass']:
        raise RuntimeError('GPU prelaunch attribution: '+','.join(baseline_gpu['failures']))
    baseline = {'gpu_device_total_mib': max(d['used_bytes'] for d in initial_devices)/1048576,
                'gpu_uuid':initial_devices[0]['uuid'],
                'known_graphics_identities':sorted(baseline_identities)}
    if baseline['gpu_uuid'] != GPU_UUID or not baseline_identities:
        raise RuntimeError('GPU baseline identity invalid')
    if baseline['gpu_device_total_mib'] > 1024:
        raise RuntimeError('GPU baseline cap')
    if host_free < 2000000000:
        raise RuntimeError('host disk free floor')
    atomic(ROOT / 'g1-prelaunch.json', {
        'container_id': inspect['Id'], 'image_id': inspect['Image'],
        'source_sha256': digest(ROOT / 'source.py'),
        'freeze_sha256': digest(ROOT / 'input/freeze.json'),
        'baseline': baseline, 'host_free_disk_bytes': host_free,
        'baseline_attribution': {key:baseline_gpu[key] for key in
                                 ('gpu_uuid','device_before_bytes','device_after_bytes',
                                  'processes','bracket_gap_bytes','owned_gpu_pids')},
        'path_publications': 0,
    })
    # The wrapper writes owner-ready and waits; Python main and physics cannot
    # start until the host has resolved PID/start-tick and accepted an NVML bracket.
    wrapper_source=pathlib.Path(__file__).with_name('eat_nav_p1_owner_wrapper.py')
    wrapper_target=ROOT/'owner-wrapper.py'
    shutil.copy2(wrapper_source,wrapper_target)
    coord=ROOT/'coord';coord.mkdir()
    token=secrets.token_hex(32)
    source_sha=digest(ROOT/'source.py');config_sha=digest(ROOT/'input/freeze.json')
    atomic(coord/'launch-intent.json',{'schema':'eatnav_p1_g1_start_v2',
        'run_token':token,'container_id':inspect['Id'],'output_root':f'{REMOTE}/output',
        'source_sha256':source_sha,'config_sha256':config_sha,
        'wrapper_sha256':digest(wrapper_target)})
    cmd=build_wrapper_command(remote=REMOTE,container_id=inspect['Id'],token=token,
                              source_sha=source_sha,config_sha=config_sha)
    atomic(ROOT/'g1-command.json',{'argv':cmd,'wrapper_sha256':digest(wrapper_target)})
    gate=G1Lifecycle(token=token,container_id=inspect['Id'],source_sha=source_sha,
                     config_sha=config_sha,output_root=f'{REMOTE}/output',gpu_uuid=GPU_UUID)
    started=time.monotonic();failure=None;owner=None;ready=None;client=None
    native_exit=None;brackets=0;cleanup_signals=[];owned_identities={}
    with (ROOT/'runner.log').open('xb') as log, \
         (ROOT/'resource-series.jsonl').open('x') as resources, \
         (ROOT/'owner-polls.jsonl').open('x') as polls:
        try:
            client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+45
            while owner is None:
                if client.poll() is not None:
                    raise RuntimeError('wrapper_exit_before_owner_ready')
                if time.monotonic()>deadline:
                    raise RuntimeError('owner_ready_or_mapping_timeout')
                ready_path=coord/'owner-ready.json'
                if not ready_path.is_file():
                    time.sleep(.05);continue
                ready=json.loads(ready_path.read_text())
                table=proc_table()
                diagnostic=diagnose_owner_resolution(ready,table,inspect['Id'])
                candidates=[{'host_pid':p['host_pid'],'container_pid':p['container_pid'],
                             'start_ticks':p['start_ticks'],'namespace_pid_chain':p['namespace_pid_chain'],
                             'cgroup':p['cgroup'],'command_line':p['command_line']}
                            for p in table.values() if p.get('container_pid')==ready.get('container_pid')
                            or token in p.get('command_line','').split()]
                append_jsonl(polls,{'wall_ns':time.time_ns(),'diagnostic':diagnostic,
                                   'ready':ready,'candidates':candidates})
                try:owner=gate.register_ready(ready,table)
                except ValueError as exc:
                    if str(exc) != 'owner_pid_mapping_missing_or_ambiguous':
                        raise
                    time.sleep(.1)
            atomic(coord/'registered-owner.json',owner)
            gpu=capture_live(nvml,{'simulator':owner},baseline_identities,GPU_CONTRACT,
                             container_id=inspect['Id'],expected_simulator_alive=True)
            row=resource_row(started,gpu);append_jsonl(resources,row);brackets+=1
            if violation:=validate_sample(row):raise RuntimeError('preauthorization_'+violation)
            gate.pre_authorization(gpu)
            authorization=gate.authorize()
            authorization.update({'bracket_monotonic_ns':gpu['monotonic_ns'],
                                  'authorization_wall_ns':time.time_ns()})
            atomic(coord/'owner-authorized.json',authorization)
            while client.poll() is None:
                gpu,row=sample(started,nvml,{'simulator':owner},baseline_identities,inspect['Id'])
                append_jsonl(resources,row);brackets+=1
                if violation:=validate_sample(row):raise RuntimeError(violation)
                gate.observe(gpu)
                for pid in gpu.get('owned_process_tree_host_pids',[]):
                    proc=gpu['process_table_before'].get(pid)
                    if proc:owned_identities[pid]=proc['start_ticks']
                if row['elapsed_s']>600:raise RuntimeError('process_wall_timeout')
                heartbeat=ROOT/'output/heartbeat.json'
                if heartbeat.is_file() and time.time()-heartbeat.stat().st_mtime>90:
                    raise RuntimeError('heartbeat_stale')
                time.sleep(.5)
            native_exit=client.wait(timeout=30)
        except BaseException as exc:
            failure=type(exc).__name__+':'+str(exc)
            if owner is not None:
                stop_exact_owner(owner,cleanup_signals)
            if client is not None and client.poll() is None:
                client.terminate()
            if client is not None:
                try:native_exit=client.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    client.kill();native_exit=client.wait(timeout=10)
        finally:
            for stream in (polls,resources,log):
                stream.flush();os.fsync(stream.fileno())
    # No file is sealed while a simulator, heartbeat, resource or log writer lives.
    final_table=proc_table()
    residual=residual_identities(final_table,owner=owner,known=owned_identities,
                                 token=token,container_id=inspect['Id'])
    writer_state={'simulator':not residual and (client is None or client.poll() is not None),
                  'heartbeat':not residual,'resource':True,'log':True}
    diagnostic_error=ROOT/'output/diagnostic-error.json'
    if diagnostic_error.exists():
        try:
            diagnostic=json.loads(diagnostic_error.read_text())
            failure='simulator_diagnostic_error:'+diagnostic['type']+':'+diagnostic['message']
        except (KeyError,TypeError,ValueError) as exc:
            failure='simulator_diagnostic_error_unreadable:'+str(exc)
    if failure is None:
        try:
            output=ROOT/'output'
            marker=json.loads((output/'material-diagnostic-complete.json').read_text())
            evidence=json.loads((output/'material-diagnostic.json').read_text())
            effective=evidence['robot']['effective']
            if (marker.get('stage')!='candidate_free_material_measurement_complete' or
                    marker.get('Path_publications')!=0 or marker.get('physics_control_steps')!=0 or
                    marker.get('material_diagnostic_sha256')!=digest(output/'material-diagnostic.json') or
                    evidence.get('Path_publications')!=0 or
                    len(evidence['robot']['link_paths_and_shape_counts'])!=4 or
                    effective['tensor_shape'][0]!=4 or
                    not evidence['terrain']['rows'] or
                    not evidence['robot']['collision_prims'] or
                    (output/'path-publication.json').exists()):
                raise ValueError('candidate-free material evidence incomplete')
            host_sha=digest(output/'material-diagnostic.json')
            container_sha=command(['docker','exec',CONTAINER,'sha256sum',
                                   f'{REMOTE}/output/material-diagnostic.json']).split()[0]
            if host_sha!=container_sha:
                raise ValueError('host/container material SHA mismatch')
            gate.complete(native_exit=native_exit,owner_alive=bool(residual),
                          writers_closed=all(writer_state.values()))
        except BaseException as exc:
            failure=type(exc).__name__+':'+str(exc)
    atomic(ROOT/'cleanup-evidence.json',{'signals':cleanup_signals,'owned_residual':residual,
        'writer_state':writer_state,'native_exit':native_exit})
    atomic(ROOT/'material-diagnostic-host-status.json',{'native_exit':native_exit,'failure':failure,
        'elapsed_s':time.monotonic()-started,'gpu_bracket_count':brackets,
        'owned_gpu_bracket_count':gate.live_gpu_samples,'owned_simulator':owner,
        'owned_simulator_residual':bool(residual),'lifecycle_phase':gate.phase})
    if all(writer_state.values()):
        stage=('G3_material_measurement_protocol_valid' if failure is None
               else 'G3_material_measurement_failed_or_inconclusive')
        seal_after_writers(ROOT,writer_state=writer_state,stage=stage,reason=failure)
        time.sleep(.1)
        verify_seal(ROOT)
    else:
        atomic(ROOT/'unsealed-evidence.json',{'reason':'writer_or_owned_residual',
                                             'failure':failure,'writer_state':writer_state})
        raise RuntimeError('material diagnostic cannot seal while owned writer remains')
    if failure is not None:raise RuntimeError('material diagnostic failed or inconclusive: '+failure)
    print('G3_MATERIAL_MEASUREMENT_PASS',str(ROOT/'host-seal.json'))


if __name__ == '__main__':
    main()
