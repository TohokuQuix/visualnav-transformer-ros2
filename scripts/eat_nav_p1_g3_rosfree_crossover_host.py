"""ROS-free G3 host: owner/GPU/resource/finalization without a ROS process."""
from __future__ import annotations
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
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_evidence import SCHEMA, validate_output
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_path_contract import validate_host_bind
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_contract import validate_crossover_prepath
from visualnav_transformer.eat_nav.p1_g3_material_gate_v2 import MATERIAL_SCHEMA

CONTAINER = 'isaac-lab-onix'
ROOT = None
REMOTE = None
BASE = ':'.join(('/tmp/eatnav-train/src', '/tmp/eatnav-extended/isaac-source',
                 '/workspace/isaaclab/source/isaaclab', '/workspace/isaaclab/source/isaaclab_rl',
                 '/workspace/isaaclab/source/isaaclab_tasks',
                 '/workspace/isaaclab_onix/source/onix_isaac_lab'))
GPU_UUID = 'GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26'
GPU_CONTRACT = {'required_gpu_uuid': GPU_UUID,
                'required_total_memory_mib_min': 16000,
                'gpu_device_total_hard_cap_mib': 7680,
                'bracket_nonatomic_sampling_gap_hard_cap_mib': 256,
                'absolute_attribution_gap_hard_cap_mib': 256,
                'unresolved_gpu_bytes_hard_cap': 0}
EXPECTED_EXIT_FLAGS = frozenset({'owned_start_tick_mismatch',
    'simulator_disappeared_in_bracket','owned_process_disappeared_in_bracket'})


def classify_gpu_bracket(sample: dict, completion: dict | None) -> dict:
    """Waive only a proven post-completion PID disappearance, not active failure."""
    start,end=sample.get('sample_started_wall_ns'),sample.get('sample_finished_wall_ns')
    if not isinstance(start,int) or not isinstance(end,int) or start>end:
        raise ValueError('GPU bracket time invalid')
    if not sample.get('device_cap_pass') or not sample.get('attribution_gap_pass'):
        raise ValueError('GPU cap or attribution gap failure')
    failures=set(sample.get('failures',[]))
    if (not completion or not completion.get('sha_valid') or
            not completion.get('owner_identity_valid') or completion.get('native_exit')!=0 or
            not isinstance(completion.get('written_wall_ns'),int)):
        if failures:raise ValueError('GPU completion unverified')
        return {'status':'active_valid'}
    written=completion['written_wall_ns']
    if end<written:
        if failures:raise ValueError('active GPU attribution failure')
        return {'status':'active_valid'}
    if start<written<=end:
        return {'status':'inconclusive_overlap' if failures else 'overlap_valid'}
    unexpected=failures-EXPECTED_EXIT_FLAGS
    if unexpected:raise ValueError('terminal GPU failure:'+','.join(sorted(unexpected)))
    return {'status':'terminal_expected_exit' if failures else 'terminal_valid',
            'waived_exit_flags':sorted(failures)}

def build_wrapper_command(*, remote: str, container_id: str, token: str,
                          source_sha: str, config_sha: str, anchor_id: str,
                          batch_index: int, execution_index: int) -> list[str]:
    """Keep the simulator paused until its exact host PID is monitored."""
    command=['docker','exec','-e',f'PYTHONPATH={remote}/src:{BASE}',CONTAINER,
            'timeout','600s','/isaac-sim/python.sh',f'{remote}/owner-wrapper.py',
            '--coord',f'{remote}/coord','--run-token',token,
            '--container-id',container_id,'--source',f'{remote}/source.py',
            '--source-sha',source_sha,'--config',f'{remote}/input/freeze.json',
            '--config-sha',config_sha,'--output',f'{remote}/output','--mode','smoke',
            '--','--headless','--input',f'{remote}/input',
            '--output',f'{remote}/output','--anchor-id',anchor_id,
            '--batch-index',str(batch_index),'--execution-index',str(execution_index)]
    return command

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

def process_tick(pid):
    try:return int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None

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
            'sample_started_wall_ns': gpu['wall_ns'],
            'sample_finished_wall_ns': gpu['process_table_after_wall_ns'],
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

def simulator_artifact_failure(output: pathlib.Path, native_exit: int | None,
                               *, diagnostic: bool = False) -> str | None:
    """A simulator diagnostic marker is a protocol failure regardless of shell exit."""
    marker=output/'diagnostic-error.json'
    if marker.is_file():
        try:
            error=json.loads(marker.read_text())
            return 'simulator_diagnostic_error:'+str(error['type'])+':'+str(error['message'])
        except (ValueError, KeyError, TypeError) as exc:
            return 'simulator_diagnostic_error_marker_invalid:'+type(exc).__name__
    if native_exit is not None and native_exit != 0:
        return 'simulator_native_exit:'+str(native_exit)
    marker_name = 'gate1-probe-complete.json' if diagnostic else 'complete.json'
    if native_exit is not None and not (output/marker_name).is_file():
        return 'simulator_completion_marker_missing'
    return None

def terminal_completion(output: pathlib.Path, native_exit: int | None,
                        owner: dict | None, root: pathlib.Path) -> dict | None:
    """Validate a pre-close completion marker before interpreting an exit bracket."""
    marker_path=output/'complete.json'
    if not marker_path.is_file() or owner is None:
        return None
    try:
        marker=json.loads(marker_path.read_text())
        registered=json.loads((root/'coord/registered-owner.json').read_text())
        valid=(registered==owner and marker.get('Path_activations')==1 and
               isinstance(marker.get('control_steps'),int) and
               41<=marker['control_steps']<=640 and
               len(marker.get('trace_sha256',[]))==4 and
               all(digest(output/f'trace-slot-{i}.jsonl')==expected for i,expected in
                   enumerate(marker['trace_sha256'])))
    except (OSError, ValueError, KeyError, TypeError):
        valid=False
    return {'written_wall_ns':marker_path.stat().st_mtime_ns,
            'sha_valid':valid,'native_exit':native_exit,
            'owner_identity_valid':registered==owner if 'registered' in locals() else False}

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


def approve_prepath(root: pathlib.Path) -> dict:
    """Period zero establishes the sealed comparison baseline before Path activation."""
    output = root/'output'
    ready = json.loads((output/'prepath-ready.json').read_text())
    snapshot = output/'prepath-crossover-snapshot.json'
    material_path = output/'live-material.json'
    if (ready.get('Path_activations') != 0 or ready.get('physics_steps_after_clone') != 0 or
            (output/'path-activation.json').exists() or
            digest(snapshot) != ready['snapshot_sha256'] or
            digest(material_path) != ready['live_material_sha256']):
        raise ValueError('pre-Path source or SHA changed before approval')
    gate = json.loads((output/'prepath-gate.json').read_text())
    actor = json.loads((output/'direct-actor-gate.json').read_text())
    material = json.loads(material_path.read_text())
    if (not gate['pass'] or actor['max_abs'] > actor['limit'] or
            material.get('schema') != MATERIAL_SCHEMA or not material.get('pass') or
            not material.get('terrain_collision_bindings') or
            not material.get('robot_collision_bindings') or
            not material.get('robot_effective_physx',{}).get('canonical_sha256')):
        raise ValueError('pre-Path physical/Actor/DEM/material gate failed')
    current=json.loads(snapshot.read_text())
    cross = {**validate_crossover_prepath(current,None,0),
             'snapshot_sha256': digest(snapshot),
             'material_sha256': digest(material_path),
             'Path_activations': 0}
    atomic(output/'cross-period-prepath-gate.json', cross)
    approval = {'pass': True, 'snapshot_sha256': digest(snapshot),
                'gate_sha256': digest(output/'cross-period-prepath-gate.json'),
                'material_sha256': digest(material_path),'Path_activations': 0}
    atomic(output/'prepath-approval.json',approval)
    return approval


def validate_staged(root: pathlib.Path, args) -> tuple[dict,dict,list[dict]]:
    manifest = json.loads((root/'input/g3-contract.json').read_text())
    if (manifest.get('schema') != SCHEMA or
            manifest.get('execution_mode') != 'path_crossover_four_env_rosfree' or
            manifest.get('runtime_authorized') is not True or
            manifest.get('authorized_process_ids') != [root.name]):
        raise ValueError('ROS-free runtime is not authorized for this process')
    assignment = json.loads((root/'input/g3-assignment.json').read_text())
    matched = [row for row in assignment['processes'] if
               (row['anchor_id'], row['batch_index'], row['execution_index']) ==
               (args.anchor_id, args.batch_index, args.execution_index)]
    if len(matched) != 1 or len(manifest['processes']) != 3 or \
            (args.anchor_id,args.batch_index,args.execution_index) != ('first-differential',0,0) or \
            manifest.get('periods_1_2_authorized') is not False:
        raise ValueError('period-zero-only crossover assignment missing')
    assigned = sorted(matched[0]['trials'], key=lambda row: row['slot'])
    frozen = sorted(manifest['processes'][0]['trials'],key=lambda row: row['slot'])
    validate_frozen_assignment(frozen, assigned, stage='prelaunch')
    if [row['path_role'] for row in assigned] != ['direct','direct','left_20','right_20']:
        raise ValueError('period-zero frozen Path roles drift')
    for name,key in (
            ('source.py','g3_simulator_source_sha256'),
            ('host-runner.py','g3_host_source_sha256'),
            ('src/visualnav_transformer/eat_nav/p1_follower_core.cpp','cpp_core_source_sha256'),
            ('src/visualnav_transformer/eat_nav/p1_follower_core.py','python_core_source_sha256'),
            ('src/visualnav_transformer/eat_nav/p1_g3_rosfree_control.py','rosfree_control_source_sha256'),
            ('src/visualnav_transformer/eat_nav/p1_g3_rosfree_crossover_evidence.py',
             'rosfree_crossover_evidence_source_sha256')):
        if digest(root/name) != manifest[key]:
            raise ValueError('ROS-free source SHA mismatch: '+name)
    if digest(root/'src/visualnav_transformer/eat_nav/p1_g3_rosfree_crossover_path_contract.py') != \
            manifest['rosfree_path_contract_source_sha256']:
        raise ValueError('ROS-free path contract source SHA mismatch')
    if (digest(root/'host-runner.py') != manifest['staged_host_runner_sha256'] or
            digest(root/'eat_nav_p1_owner_wrapper.py') != manifest['staged_owner_wrapper_sha256'] or
            digest(root/'eat_nav_p1_owner_wrapper.py') != manifest['owner_wrapper_sha256']):
        raise ValueError('staged owner wrapper/host SHA mismatch')
    if digest(root/'src/visualnav_transformer/eat_nav/p1_g3_rosfree_crossover_contract.py') != \
            manifest['rosfree_crossover_contract_source_sha256']:
        raise ValueError('ROS-free crossover contract source SHA mismatch')
    if digest(__file__) != manifest['g3_host_source_sha256']:
        raise ValueError('host source SHA drift')
    ledger = json.loads((root/'input-sha-ledger.json').read_text())
    if any(pathlib.Path(name).is_absolute() or '..' in pathlib.Path(name).parts or
           digest(root/name) != expected for name, expected in ledger.items()):
        raise ValueError('staged source/input SHA ledger mismatch')
    return manifest, matched[0], assigned


def host_container_sha(root: pathlib.Path, remote: str) -> dict:
    names=['complete.json','path-activation.json','prepath-gate.json',
           'prepath-shared-dem-sha-ledger.json','prepath-dem-sha-ledger.json',
           'live-material.json',*[f'trace-slot-{i}.jsonl' for i in range(4)]]
    result=subprocess.run(['docker','exec',CONTAINER,'sha256sum',
        *[f'{remote}/output/{name}' for name in names]],capture_output=True,text=True,check=True)
    container={line.split()[1].rsplit('/',1)[-1]:line.split()[0]
               for line in result.stdout.splitlines()}
    host={name:digest(root/'output'/name) for name in names}
    if host != container:
        raise ValueError('host/container evidence SHA mismatch')
    return host


def main():
    global ROOT, REMOTE
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=pathlib.Path,required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--anchor-id',choices=('first-differential','second-graded'),required=True)
    parser.add_argument('--batch-index',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--execution-index',type=int,choices=(0,1,2),required=True)
    args=parser.parse_args()
    ROOT=args.root.resolve();REMOTE=args.remote
    expected=f'batch-{args.anchor_id}-{args.batch_index}-period-{args.execution_index}'
    if ROOT.name!=expected or (ROOT/'output').exists() or (ROOT/'host-seal.json').exists():
        raise ValueError('fresh dedicated ROS-free batch root required')
    manifest, process, assigned=validate_staged(ROOT,args)
    cfg=json.loads((ROOT/'input/statistical-config.json').read_text())
    frozen=json.loads((ROOT/'input/freeze.json').read_text())
    for name,key in (('assignment.json','assignment_sha256'),
                     ('statistical-config.json','config_sha256'),
                     ('source-contract.json','source_contract_sha256')):
        if digest(ROOT/'input'/name)!=frozen[key]:
            raise ValueError('frozen P1 input SHA mismatch: '+name)
    inspect=json.loads(command(['docker','inspect',CONTAINER]))[0]
    if inspect['State']['Status']!='running':raise RuntimeError('Isaac container not running')
    bind_evidence=validate_host_bind(inspect,manifest,ROOT,REMOTE)
    if subprocess.run(['docker','exec',CONTAINER,'test','-d',REMOTE],capture_output=True).returncode:
        raise RuntimeError('persistent root not visible in container')
    for relative in ('source.py','input/g3-contract.json',
                     'src/visualnav_transformer/eat_nav/p1_g3_rosfree_crossover_path_contract.py'):
        container_sha=command(['docker','exec',CONTAINER,'sha256sum',f'{REMOTE}/{relative}']).split()[0]
        if container_sha!=digest(ROOT/relative):
            raise ValueError('host/container bind file SHA mismatch: '+relative)
    checkpoint=command(['docker','exec',CONTAINER,'sha256sum',cfg['rl_checkpoint']]).split()[0]
    if checkpoint!=cfg['rl_checkpoint_sha256'] or checkpoint!=manifest['rl_checkpoint_sha256']:
        raise ValueError('RL checkpoint SHA mismatch')
    nvml=Nvml();devices,nvml_rows=nvml.sample();table=proc_table()
    baseline_identities=initial_graphics_identities(nvml_rows,table)
    baseline_gpu=capture_live(nvml,{},baseline_identities,GPU_CONTRACT,
        container_id=inspect['Id'],expected_simulator_alive=False,device_cap_mib=1024)
    if (not baseline_gpu['pass'] or not baseline_identities or
            devices[0]['uuid']!=GPU_UUID or devices[0]['used_bytes']>1024*1048576):
        raise RuntimeError('GPU prelaunch identity/attribution invalid')
    if shutil.disk_usage(ROOT).free<2_000_000_000:
        raise RuntimeError('persistent disk free floor')
    atomic(ROOT/'rosfree-prelaunch.json',{'container_id':inspect['Id'],'image_id':inspect['Image'],
        'backend':'ros_free_cpp_core','external_router_used':False,
        'bind_mapping':bind_evidence,
        'source_sha256':digest(ROOT/'source.py'),'host_sha256':digest(__file__),
        'checkpoint_sha256':checkpoint,'gpu_uuid':GPU_UUID})
    wrapper=pathlib.Path(__file__).with_name('eat_nav_p1_owner_wrapper.py')
    if digest(wrapper)!=manifest['owner_wrapper_sha256']:
        raise ValueError('owner wrapper source SHA drift')
    shutil.copy2(wrapper,ROOT/'owner-wrapper.py')
    coord=ROOT/'coord';coord.mkdir()
    token=secrets.token_hex(32)
    source_sha=digest(ROOT/'source.py');config_sha=digest(ROOT/'input/freeze.json')
    atomic(coord/'launch-intent.json',{'schema':SCHEMA,'run_token':token,
        'container_id':inspect['Id'],'output_root':f'{REMOTE}/output',
        'source_sha256':source_sha,'config_sha256':config_sha,
        'wrapper_sha256':digest(ROOT/'owner-wrapper.py')})
    cmd=build_wrapper_command(remote=REMOTE,container_id=inspect['Id'],token=token,
        source_sha=source_sha,config_sha=config_sha,anchor_id=args.anchor_id,
        batch_index=args.batch_index,execution_index=args.execution_index)
    atomic(ROOT/'rosfree-command.json',{'argv':cmd,'wrapper_sha256':digest(ROOT/'owner-wrapper.py')})
    gate=G1Lifecycle(token=token,container_id=inspect['Id'],source_sha=source_sha,
        config_sha=config_sha,output_root=f'{REMOTE}/output',gpu_uuid=GPU_UUID)
    started=time.monotonic();failure=None;owner=None;client=None;native_exit=None
    brackets=0;cleanup_signals=[];owned_identities={};prepath_approved=False
    monitor_failure=None;terminal_gpu_row=None;terminal_gpu_classification=None
    with (ROOT/'runner.log').open('xb') as log, \
         (ROOT/'resource-series.jsonl').open('x') as resources, \
         (ROOT/'owner-polls.jsonl').open('x') as polls:
        try:
            client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+45
            while owner is None:
                if client.poll() is not None:raise RuntimeError('wrapper exited before owner readiness')
                if time.monotonic()>deadline:raise TimeoutError('owner mapping timeout')
                ready_path=coord/'owner-ready.json'
                if not ready_path.is_file():time.sleep(.05);continue
                ready=json.loads(ready_path.read_text());table=proc_table()
                diagnostic=diagnose_owner_resolution(ready,table,inspect['Id'])
                append_jsonl(polls,{'wall_ns':time.time_ns(),'diagnostic':diagnostic,
                                    'ready':ready})
                try:owner=gate.register_ready(ready,table)
                except ValueError as exc:
                    if str(exc)!='owner_pid_mapping_missing_or_ambiguous':raise
                    time.sleep(.1)
            atomic(coord/'registered-owner.json',owner)
            gpu=capture_live(nvml,{'simulator':owner},baseline_identities,GPU_CONTRACT,
                container_id=inspect['Id'],expected_simulator_alive=True)
            row=resource_row(started,gpu);append_jsonl(resources,row);brackets+=1
            if violation:=validate_sample(row):raise RuntimeError('preauthorization_'+violation)
            gate.pre_authorization(gpu)
            auth=gate.authorize();auth.update({'bracket_monotonic_ns':gpu['monotonic_ns'],
                                               'authorization_wall_ns':time.time_ns()})
            atomic(coord/'owner-authorized.json',auth)
            while client.poll() is None:
                gpu,row=sample(started,nvml,{'simulator':owner},baseline_identities,inspect['Id'])
                append_jsonl(resources,row);brackets+=1
                if violation:=validate_sample(row):
                    monitor_failure=violation
                    if not violation.startswith('gpu_attribution:') or \
                            not (ROOT/'output/complete.json').is_file():
                        raise RuntimeError(violation)
                    terminal_gpu_row=row;break
                gate.observe(gpu)
                for pid in gpu.get('owned_process_tree_host_pids',[]):
                    proc=gpu['process_table_before'].get(pid)
                    if proc:owned_identities[pid]=proc['start_ticks']
                if row['elapsed_s']>600:raise TimeoutError('G3 process wall cap')
                if not prepath_approved and (ROOT/'output/prepath-ready.json').is_file():
                    approve_prepath(ROOT);prepath_approved=True
                heartbeat=ROOT/'output/heartbeat.json'
                if heartbeat.is_file() and time.time()-heartbeat.stat().st_mtime>90:
                    raise TimeoutError('G3 heartbeat stale')
                time.sleep(.5)
            native_exit=client.wait(timeout=30)
            if terminal_gpu_row is not None:
                bracket={'sample_started_wall_ns':terminal_gpu_row['sample_started_wall_ns'],
                         'sample_finished_wall_ns':terminal_gpu_row['sample_finished_wall_ns'],
                         'failures':terminal_gpu_row['gpu_attribution']['failures'],
                         'device_cap_pass':terminal_gpu_row['gpu_device_total_mib']<=7680 and
                             terminal_gpu_row['host_available_bytes']>=10095421850 and
                             terminal_gpu_row['container_anon_plus_shmem_bytes']<=6442450944 and
                             terminal_gpu_row['persistent_root_bytes']<=1000000000,
                         'attribution_gap_pass':
                             terminal_gpu_row['gpu_attribution']['bracket_gap_bytes']<=256*1048576}
                terminal_gpu_classification=classify_gpu_bracket(
                    bracket,terminal_completion(ROOT/'output',native_exit,owner,ROOT))
                if terminal_gpu_classification['status'] not in ('terminal_expected_exit','terminal_valid'):
                    raise RuntimeError('terminal GPU boundary inconclusive')
                monitor_failure=None
            if not prepath_approved:
                raise RuntimeError('simulator exited before pre-Path approval')
        except BaseException as exc:
            failure=type(exc).__name__+':'+str(exc)
            if owner is not None:stop_exact_owner(owner,cleanup_signals)
            if client is not None and client.poll() is None:client.terminate()
            if client is not None:
                try:native_exit=client.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    client.kill();native_exit=client.wait(timeout=10)
        finally:
            for stream in (polls,resources,log):
                stream.flush();os.fsync(stream.fileno())
    residual=residual_identities(proc_table(),owner=owner,known=owned_identities,
                                 token=token,container_id=inspect['Id'])
    writer_state={'simulator':not residual and (client is None or client.poll() is not None),
                  'heartbeat':not residual,'resource':True,'log':True}
    artifact_failure=simulator_artifact_failure(ROOT/'output',native_exit)
    if artifact_failure is not None:
        failure=artifact_failure if failure is None else artifact_failure+'; '+failure
    validation=None;container_sha=None
    if failure is None:
        try:
            validation=validate_output(ROOT/'output',manifest,assigned)
            container_sha=host_container_sha(ROOT,REMOTE)
            gate.complete(native_exit=native_exit,owner_alive=bool(residual),
                          writers_closed=all(writer_state.values()))
        except BaseException as exc:failure=type(exc).__name__+':'+str(exc)
    atomic(ROOT/'cleanup-evidence.json',{'signals':cleanup_signals,'owned_residual':residual,
        'writer_state':writer_state,'native_exit':native_exit})
    atomic(ROOT/'rosfree-host-status.json',{'failure':failure,'native_exit':native_exit,
        'monitor_failure':monitor_failure,'monitor_terminal_classification':terminal_gpu_classification,
        'elapsed_s':time.monotonic()-started,'gpu_bracket_count':brackets,
        'owned_gpu_bracket_count':gate.live_gpu_samples,'owned_simulator':owner,
        'owned_simulator_residual':bool(residual),'lifecycle_phase':gate.phase,
        'validation':validation,'container_sha':container_sha})
    if all(writer_state.values()):
        seal_after_writers(ROOT,writer_state=writer_state,
            stage='G3_rosfree_crossover_protocol_valid' if failure is None else 'G3_rosfree_failed_or_inconclusive',
            reason=failure)
        time.sleep(.1);verify_seal(ROOT)
    else:
        atomic(ROOT/'unsealed-evidence.json',{'failure':failure,'writer_state':writer_state})
        raise RuntimeError('owned writer remains; cannot seal')
    if failure is not None:raise RuntimeError('ROS-free G3 failed/inconclusive: '+failure)
    print('ROSFREE_G3_BATCH_PASS',ROOT/'host-seal.json')


if __name__=='__main__':
    main()
