"""One-shot 112-env candidate-free P1 resource-attribution smoke."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from eat_nav_g0e_gpu_probe import Nvml, proc_info, proc_table, safe_signal
from eat_nav_p1_gpu_attribution import capture_live, map_container_pid
from eat_nav_radial_anchor_fixture import docker, remote_json, remote_sha, verify_source


WORKTREE=Path(__file__).resolve().parents[1]
R5=WORKTREE/'docs/eat-nav/local-primitive-p1-r5-20261008-attempt2'
PRIOR=WORKTREE/'docs/eat-nav/eatnav-radial-anchor-contract-20261008-r2'
PRIOR_RAW=Path('/home/kojima/eatnav-backups/2026-10-08-radial-anchor-contract-r2')
ROOT=Path('/tmp/eatnav-p1-resource-rollout-20261008-attempt1')
CONTAINER='isaac-lab-onix'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def choose_shard(entries:list[dict],gpu_peaks:dict[str,int])->dict:
    if not entries or any(e['fixture_subroot'] not in gpu_peaks for e in entries):
        raise ValueError('source_resource_history_incomplete')
    return max(enumerate(entries),key=lambda row:(gpu_peaks[row[1]['fixture_subroot']],-row[0]))[1]


def find_launcher(table:dict,container_id:str,output_path:str)->dict:
    matches=[]
    for row in table.values():
        if container_id not in row.get('cgroup',''):continue
        tokens=row.get('command_line','').split()
        if '--output' in tokens and tokens[tokens.index('--output')+1:tokens.index('--output')+2]==[output_path]:
            matches.append(row)
    if len(matches)!=1 or not isinstance(matches[0].get('start_ticks'),int):
        raise ValueError('launcher_host_pid_mapping_missing_or_ambiguous')
    return {'host_pid':matches[0]['host_pid'],'start_ticks':matches[0]['start_ticks']}


def validate_invocation(cmd:list[str],output_path:str)->None:
    forbidden={'--candidate-path','--follower','--bridge','--path-json','--outcome'}
    if forbidden.intersection(cmd) or 'ros2' in cmd or 'nav_msgs' in cmd:
        raise ValueError('candidate_free_invocation_violation')
    if '--output' not in cmd or cmd[cmd.index('--output')+1:cmd.index('--output')+2]!=[output_path]:
        raise ValueError('output_path_mismatch')
    if '--mode' not in cmd or cmd[cmd.index('--mode')+1]!='full':
        raise ValueError('candidate_free_full_settling_mode_required')


def prepare(root:Path=ROOT)->dict:
    if root.exists():raise RuntimeError('fresh_authoritative_output_root_required')
    prior_decision=read(PRIOR/'final-decision.json')
    if prior_decision['physics_parity']!='pass_4_of_4':
        raise RuntimeError('prior radial parity not passed')
    prior_ledger=read(PRIOR/'artifact-sha-ledger.json')
    for entry in prior_ledger['small_report_files']:
        path=Path(entry['path'])
        if not path.exists() or sha(path)!=entry['sha256']:
            raise RuntimeError('prior parity ledger mismatch: '+str(path))
    entries=read(PRIOR_RAW/'fixture-manifest.json')['shards']
    prior_resources=read(PRIOR_RAW/'resource-lifecycle.json')['shards']
    peaks={entry['subroot']:entry['gpu_peak_mib'] for entry in prior_resources}
    # Durable backup paths differ from the original /tmp subroot strings, so retain exact ledger keys.
    selected=choose_shard(entries,peaks)
    source=Path(selected['source_shard'])
    freeze,prelaunch=verify_source(source)
    target=root/'attribution-smoke'
    cmd=list(prelaunch['command']);cmd[cmd.index('--output')+1]=str(target/'run')
    validate_invocation(cmd,str(target/'run'))
    contract=read(R5/'p1-resource-contract-v2.json')
    if contract['unresolved_gpu_bytes_hard_cap']!=0 or contract['gpu_device_total_hard_cap_mib']!=7680:
        raise RuntimeError('P1 v2 resource contract drift')
    root.mkdir(parents=True)
    manifest={'schema':'eatnav_p1_attribution_smoke_manifest_v1','source_selection_rule':'highest prior candidate-free GPU device peak; tie original shard order',
              'selected_prior_subroot':selected['fixture_subroot'],'selected_prior_peak_mib':peaks[selected['fixture_subroot']],
              'source_shard':str(source),'source_freeze_sha256':sha(source/'input-freeze.json'),
              'source_pose_manifest_sha256':sha(source/'pose-manifest.json'),
              'source_sim_sha256':freeze['sim_source_sha256'],'candidate_free':True,
              'physics_steps':200,'action':[0.]*6,'output_subroot':str(target),
              'command':cmd,'resource_contract_sha256':sha(R5/'p1-resource-contract-v2.json'),
              'prior_parity_decision_sha256':sha(PRIOR/'final-decision.json'),
              'prior_parity_ledger_sha256':sha(PRIOR/'artifact-sha-ledger.json'),
              'candidate_outcome_read':False}
    atomic(root/'attribution-smoke-manifest.json',manifest)
    atomic(root/'prior-parity-reuse-audit.json',{'physics_parity':prior_decision['physics_parity'],
           'anchor_count':prior_decision['anchors_passed'],'prior_decision_sha256':sha(PRIOR/'final-decision.json'),
           'prior_ledger_sha256':sha(PRIOR/'artifact-sha-ledger.json'),
           'three_shard_fixture_rerun':False})
    atomic(root/'source-provenance.json',{'base_commit':'3fa65484873492fef4457922d644033da99f29fc',
           'source_shard':str(source),'source_freeze_sha256':manifest['source_freeze_sha256'],
           'source_pose_manifest_sha256':manifest['source_pose_manifest_sha256'],
           'resource_contract_sha256':manifest['resource_contract_sha256'],
           'collector_source_sha256':sha(WORKTREE/'scripts/eat_nav_p1_gpu_attribution.py'),
           'smoke_runner_source_sha256':sha(WORKTREE/'scripts/eat_nav_p1_attribution_smoke.py'),
           'targeted_test_sha256':sha(WORKTREE/'tests/test_p1_attribution_smoke.py')})
    return manifest


def _gpu_baseline(nvml,contract,container_id):
    devices,rows=nvml.sample();table=proc_table()
    baseline={(row['host_pid'],table[row['host_pid']]['start_ticks']) for row in rows if row['host_pid'] in table}
    sample=capture_live(nvml,{},baseline,contract,container_id=container_id,
                        expected_simulator_alive=False,device_cap_mib=1024)
    return sample,baseline


def _memory(container_id):
    proc=docker('exec',CONTAINER,'cat','/sys/fs/cgroup/memory.stat',check=True)
    d={x.split()[0]:int(x.split()[1]) for x in proc.stdout.splitlines() if len(x.split())==2}
    host=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    return d.get('anon',0)+d.get('shmem',0),host


def _storage(root,target):
    remote=docker('exec',CONTAINER,'du','-sb',target)
    remote_bytes=int(remote.stdout.split()[0]) if remote.returncode==0 else 0
    host_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    return remote_bytes+host_bytes


def run(root:Path=ROOT)->dict:
    manifest=read(root/'attribution-smoke-manifest.json')
    code_freeze=read(root/'source-provenance.json')
    for key,path in [('collector_source_sha256',WORKTREE/'scripts/eat_nav_p1_gpu_attribution.py'),
                     ('smoke_runner_source_sha256',WORKTREE/'scripts/eat_nav_p1_attribution_smoke.py'),
                     ('targeted_test_sha256',WORKTREE/'tests/test_p1_attribution_smoke.py')]:
        if sha(path)!=code_freeze[key]:raise RuntimeError('smoke_code_or_test_SHA_drift:'+key)
    target=Path(manifest['output_subroot'])
    if target.exists():raise RuntimeError('smoke_subroot_already_exists')
    source=Path(manifest['source_shard'])
    verify_source(source)
    if sha(source/'input-freeze.json')!=manifest['source_freeze_sha256'] or sha(source/'pose-manifest.json')!=manifest['source_pose_manifest_sha256']:
        raise RuntimeError('frozen source SHA drift')
    if sha(R5/'p1-resource-contract-v2.json')!=manifest['resource_contract_sha256']:
        raise RuntimeError('resource contract SHA drift')
    cmd=manifest['command'];validate_invocation(cmd,str(target/'run'))
    inspect=docker('inspect','-f','{{.Id}} {{.State.Running}}',CONTAINER,check=True).stdout.split()
    if len(inspect)!=2 or inspect[1]!='true':raise RuntimeError('container_not_running')
    container_id=inspect[0];contract=read(R5/'p1-resource-contract-v2.json')
    nvml=Nvml();prebaseline,baseline=_gpu_baseline(nvml,contract,container_id)
    if not prebaseline['pass']:
        atomic(root/'gpu-baseline-pre-smoke.json',prebaseline)
        atomic(root/'final-decision.json',{'decision':'p1_resource_attribution_fail','first_failing_signal':'pre_smoke_gpu_baseline'})
        return read(root/'final-decision.json')
    atomic(root/'gpu-baseline-pre-smoke.json',prebaseline)
    target.mkdir()
    docker('exec',CONTAINER,'mkdir','-p',target,check=True)
    started=time.monotonic();client=None;owner=None;marker_identity=None;series=[];failure=None
    try:
        with (target/'runner.log').open('wb') as log:
            client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            docker_client=proc_info(client.pid)
            if docker_client is None:raise RuntimeError('docker_exec_host_pid_unreadable')
            docker_identity={'host_pid':client.pid,'start_ticks':docker_client['start_ticks']}
            atomic(target/'docker-exec-identity.json',docker_identity)
            while client.poll() is None:
                owned={'simulator':owner,'docker_exec':docker_identity} if owner else {}
                sample=capture_live(nvml,owned,baseline,contract,container_id=container_id,
                                    expected_simulator_alive=owner is not None,
                                    owner_resolver=(lambda rows:{'simulator':find_launcher(rows,container_id,str(target/'run')),
                                                                 'docker_exec':docker_identity}) if owner is None else None)
                owner=sample['resolved_owned'].get('simulator') or owner
                if owner is None and time.monotonic()-started>20:
                    raise RuntimeError('launcher_host_pid_not_found')
                heartbeat=remote_json(target/'run/heartbeat.json')
                marker=remote_json(target/'run/owned-process.json')
                if marker is not None:
                    mapped=map_container_pid(proc_table(),marker['pid'],container_id)
                    marker_identity=mapped
                    if owner is None or mapped!=owner:
                        raise RuntimeError('launcher_simulator_pid_or_start_tick_mismatch')
                    if marker['proc_start_ticks']!=proc_table()[mapped['host_pid']]['start_ticks']:
                        # The container and host observe the same kernel start tick.
                        raise RuntimeError('container_host_start_tick_mismatch')
                anon,available=_memory(container_id)
                storage=_storage(root,target)
                sample.update({'stage':'running','elapsed_sec':time.monotonic()-started,
                               'heartbeat':heartbeat,'owned':sample['resolved_owned'],'container_marker':marker,
                               'anon_shmem_bytes':anon,'host_available_bytes':available,'storage_bytes':storage})
                series.append(sample)
                with (target/'gpu-resource-series.jsonl').open('a') as f:f.write(json.dumps(sample,sort_keys=True)+'\n')
                atomic(target/'resource-latest.json',sample)
                reasons=list(sample['failures'])
                if anon>contract['container_anon_plus_shmem_cap_bytes']:reasons.append('anon_shmem_cap')
                if available<contract['minimum_host_available_bytes']:reasons.append('host_ram_floor')
                if storage>contract['storage_cap_bytes']:reasons.append('storage_cap')
                if sample['elapsed_sec']>contract['per_process_timeout_sec']:reasons.append('process_timeout')
                if heartbeat and (time.monotonic_ns()-heartbeat['monotonic_ns'])/1e9>contract['heartbeat_stale_sec']:
                    reasons.append('heartbeat_stale')
                if reasons:raise RuntimeError('resource_attribution_gate:'+','.join(sorted(set(reasons))))
                time.sleep(1)
        docker('cp',f'{CONTAINER}:{target}/run/.',target/'run',check=True)
        finished=read(target/'run/sim-finished.json') if (target/'run/sim-finished.json').exists() else None
        if client.returncode!=0 or not finished or finished.get('physics_steps')!=200 or finished.get('physical_termination_pose_ids'):
            raise RuntimeError('smoke_native_exit_or_completion_invalid')
        if finished['sim_source_sha256']!=manifest['source_sim_sha256']:
            raise RuntimeError('smoke_sim_source_sha_mismatch')
        if marker_identity is None:raise RuntimeError('simulator_host_identity_missing')
        if proc_info(marker_identity['host_pid']) is not None and proc_info(marker_identity['host_pid'])['start_ticks']==marker_identity['start_ticks']:
            raise RuntimeError('owned_simulator_residual')
        files=[]
        for path in sorted((target/'run').rglob('*')):
            if path.is_file():
                digest=sha(path)
                if remote_sha(path)!=digest:raise RuntimeError('host_container_artifact_sha_mismatch:'+str(path))
                files.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
        shutdown=capture_live(nvml,{},baseline,contract,container_id=container_id,
                              expected_simulator_alive=False)
        shutdown['stage']='shutdown';series.append(shutdown)
        with (target/'gpu-resource-series.jsonl').open('a') as f:f.write(json.dumps(shutdown,sort_keys=True)+'\n')
        if not shutdown['pass']:raise RuntimeError('shutdown_gpu_attribution:'+','.join(shutdown['failures']))
        steps=[x['heartbeat']['physics_step'] for x in series if x.get('stage')=='running' and x.get('heartbeat')]
        if not any(x.get('stage')=='running' and not x.get('heartbeat') for x in series) or not any(0<s<200 for s in steps):
            raise RuntimeError('startup_or_steady_resource_sampling_missing')
        atomic(target/'raw-sha-ledger.json',{'files':files})
        result={'decision':'p1_resource_attribution_pass','source_shard':str(source),'candidate_free':True,
                'physics_steps':200,'native_exit_code':client.returncode,'completion_marker_sha256':sha(target/'run/sim-finished.json'),
                'host_container_sha_match':True,'owned_residual_count':0,'docker_exec_identity':docker_identity,
                'simulator_identity':marker_identity,'sample_count':len(series),
                'startup_samples':sum(x.get('stage')=='running' and not x.get('heartbeat') for x in series),
                'steady_samples':sum(0<x.get('heartbeat',{}).get('physics_step',0)<200 for x in series if x.get('stage')=='running'),
                'shutdown_sample':True,'gpu_device_peak_bytes':max(max(x['device_before_bytes'],x['device_after_bytes']) for x in series),
                'max_bracket_gap_bytes':max(x['bracket_gap_bytes'] for x in series),
                'unresolved_gpu_bytes_max':max(x['unresolved_gpu_bytes'] for x in series),
                'resource_series_sha256':sha(target/'gpu-resource-series.jsonl'),
                'raw_sha_ledger_sha256':sha(target/'raw-sha-ledger.json')}
        atomic(target/'smoke-seal.json',result);atomic(root/'attribution-smoke-result.json',result)
        atomic(root/'stage-status.json',{'stage':'attribution_smoke_pass','smoke_seal_sha256':sha(target/'smoke-seal.json')})
        return result
    except BaseException as exc:
        failure=repr(exc)
        if client and client.poll() is None:
            identities={(owner['host_pid'],owner['start_ticks'])} if owner else set()
            safe_signal(identities,signal.SIGTERM)
            try:client.wait(timeout=15)
            except subprocess.TimeoutExpired:
                safe_signal(identities,signal.SIGKILL);client.terminate();client.wait(timeout=10)
        docker('cp',f'{CONTAINER}:{target}/run/.',target/'run')
        atomic(target/'smoke-failure.json',{'decision':'p1_resource_attribution_fail','error':failure,
               'native_exit_code':None if client is None else client.poll(),
               'owned_simulator_identity':owner,'container_marker_identity':marker_identity,
               'samples':len(series),'last_sample':series[-1] if series else None})
        atomic(root/'final-decision.json',{'decision':'p1_resource_attribution_fail','first_failing_signal':failure,
               'candidate_paths_published':0,'candidate_processes_started':0})
        return read(root/'final-decision.json')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args()
    if args.prepare==args.run:raise SystemExit('choose exactly one of --prepare / --run')
    result=prepare(args.root) if args.prepare else run(args.root)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
