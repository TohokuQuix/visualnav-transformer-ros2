"""Read-only verification and sealing of completed E1 run artifacts."""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import shutil
import time

from eat_nav_e1_process_ownership import ProcessRegistry,ownership_spec
from eat_nav_e1_monitor import verify_complete,finite

FORMAL_FILES=('started.json','docker-exec-return.json','final-status.json',
    'run-complete.json','bridge-shutdown.json','live-0.marker.json',
    'live-0.trace.json.gz','live-0.clock.json.gz','live-0.json','anchor-gate.json',
    'timing-monitor.done.json','finalization-verifier-result.json','protocol-valid.json',
    'archived-log-sha.json','runtime-monitor-result.json','final-resource-sample.json',
    'gpu-process-timeseries.jsonl','timing-events.jsonl','timing-monitor.log',
    'simulator.log','bridge.log','follower-logs/bridge-events.jsonl')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def verify_recovered_registry(path,expected_sha,run_id):
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError('recovered registry SHA differs')
    registry=ProcessRegistry.load(path)
    if registry.run_id!=run_id:raise ValueError('recovered registry run ID differs')
    registry.require_complete()
    roles=ownership_spec()
    for entry in registry.entries:
        if entry['run_id']!=run_id or entry['run_start_ns']!=registry.run_start_ns:
            raise ValueError('recovered registry run identity differs')
        if (entry['ownership']!=roles[entry['role']]['ownership'] or
                not isinstance(entry['host_pid'],int) or
                not isinstance(entry['start_ticks'],int) or entry['start_ticks']<=0):
            raise ValueError('recovered registry role/PID/start tick differs')
    return {'status':'pass','run_id':run_id,'sha256':expected_sha,
        'role_count':len({x['role'] for x in registry.entries}),
        'entry_count':len(registry.entries),'run_start_ns':registry.run_start_ns}


def tree_ledger(root):
    root=Path(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name!='sealed-tree-manifest.json'}


def seal_tree(root):
    root=Path(root)
    if (root/'sealed-tree-manifest.json').exists():raise FileExistsError('already sealed')
    atomic(root/'sealed-tree-manifest.json',{'status':'sealed','files':tree_ledger(root)})
    return verify_seal(root)


def verify_seal(root):
    root=Path(root)
    manifest=json.loads((root/'sealed-tree-manifest.json').read_text())
    if manifest['status']!='sealed' or manifest['files']!=tree_ledger(root):
        raise ValueError('sealed tree changed')
    return sha(root/'sealed-tree-manifest.json')


def assert_current_run_path(root,run_id,path):
    root=Path(root);path=Path(path)
    if not run_id or run_id!=Path(run_id).name or run_id in ('.','..'):
        raise ValueError('invalid current run ID')
    run=(root/'runs'/run_id)
    if run.is_symlink():raise ValueError('current run cannot be a symlink')
    if path.resolve().parent!=run.resolve() or run.name!=run_id:
        raise ValueError('output path outside current run root')
    return path


def recover_first_run(source,original_registry,recovery,expected_identity):
    """Copy only completed artifacts; inspect values solely for protocol validity."""
    source=Path(source);original_registry=Path(original_registry);recovery=Path(recovery)
    if recovery.exists():raise FileExistsError(recovery)
    run_id=source.name
    verifier=json.loads((source/'finalization-verifier-result.json').read_text())
    expected_registry_sha=verifier['process_registry_sha256']
    registry=verify_recovered_registry(original_registry,expected_registry_sha,run_id)
    corrupted=source/'process-registry.json'
    corrupted_bytes=corrupted.read_bytes();saved_bytes=original_registry.read_bytes()
    if not corrupted_bytes.startswith(saved_bytes):raise ValueError('corrupt registry is not an append')
    suffix=corrupted_bytes[len(saved_bytes):]
    samples=[json.loads(row) for row in suffix.splitlines() if row.strip()]
    if len(samples)!=21:raise ValueError('unexpected appended resource sample count')
    source_sha={name:sha(source/name) for name in FORMAL_FILES}
    original_sha=sha(original_registry);corrupt_sha=sha(corrupted)
    recovery.mkdir(parents=True)
    for name in FORMAL_FILES:
        target=recovery/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
        if sha(target)!=source_sha[name]:raise ValueError('source changed during recovery '+name)
    shutil.copyfile(original_registry,recovery/'process-registry.json')
    if sha(recovery/'process-registry.json')!=original_sha:raise ValueError('registry copy changed')
    if verify_complete(recovery,source_run=source)!=sha(recovery/'run-complete.json'):
        raise ValueError('original finalization verifier failed')
    complete=json.loads((recovery/'run-complete.json').read_text())
    valid=json.loads((recovery/'protocol-valid.json').read_text())
    done=json.loads((recovery/'timing-monitor.done.json').read_text())
    anchor=json.loads((recovery/'anchor-gate.json').read_text())
    live=json.loads((recovery/'live-0.json').read_text())
    final=json.loads((recovery/'final-status.json').read_text())
    if complete['identity']!=expected_identity or valid['identity']!=expected_identity:
        raise ValueError('manifest identity differs')
    if (verifier['status']!='pass' or verifier['run_complete_sha256']!=sha(recovery/'run-complete.json')
            or valid['status']!='valid' or valid['run_complete_sha256']!=sha(recovery/'run-complete.json')
            or valid['monitor_done_sha256']!=sha(recovery/'timing-monitor.done.json')
            or valid['finalization_verifier_sha256']!=sha(recovery/'finalization-verifier-result.json')):
        raise ValueError('completion markers differ')
    if done['status']!='stopped' or done['runtime_monitor']!='pass':
        raise ValueError('monitor stop/join missing')
    if (anchor['status']!='pass' or not anchor['path_publication_allowed'] or
            anchor['dem_history_gate']['status']!='pass' or
            len(anchor['dem_history_gate']['frames'])!=4 or
            not all(f['finite'] and f['identity']==expected_identity['episode_key']
                for f in anchor['dem_history_gate']['frames'])):
        raise ValueError('anchor/DEM/Path gate differs')
    if live.get('anchor_gate')!=anchor or final['stage']!='COMPLETE' or final['residual_processes']:
        raise ValueError('native/anchor/cleanup differs')
    if [t['to'] for t in verifier['lifecycle_transitions']]!=['RUNNING','FINALIZING','COMPLETE']:
        raise ValueError('finalization transition differs')
    if not finite(live):raise ValueError('nonfinite native output')
    with gzip.open(recovery/'live-0.trace.json.gz','rt') as stream:
        trace=json.load(stream)
    if any(step.get('terminal') is not None or
            step['step_identity']['run_id']!=run_id or
            step['step_identity']['control_step']!=index or
            step['step_identity']['state_id']!=expected_identity['state_id']
            for index,step in enumerate(trace)):
        raise ValueError('terminal/reset/protocol step identity differs')
    if json.loads((recovery/'runtime-monitor-result.json').read_text())['status']!='pass':
        raise ValueError('runtime monitor protocol failed')
    manifest={'status':'recovered','run_id':run_id,'source_run':str(source),
        'mapping':{'process-registry.json':str(original_registry)},
        'corrupted_registry':{'path':str(corrupted),'sha256':corrupt_sha},
        'saved_original_registry':{'path':str(original_registry),'sha256':original_sha},
        'verifier_expected_registry_sha256':expected_registry_sha,
        'original_registry_matches_expected':original_sha==expected_registry_sha,
        'appended_resource_sample_count':len(samples),'appended_suffix_sha256':hashlib.sha256(suffix).hexdigest(),
        'recovery_reason':'post_completion_cross_run_append',
        'artifact_recovered':True,'original_completion_valid':True,
        'current_original_tree_intact':False,'recovered_tree_verified':True,
        'registry_check':registry,'source_file_sha256':source_sha,
        'recovery_file_sha256':tree_ledger(recovery),'discovered_wall_ns':time.time_ns()}
    atomic(recovery/'artifact-recovery-manifest.json',manifest)
    seal_sha=seal_tree(recovery)
    if sha(corrupted)!=corrupt_sha or sha(original_registry)!=original_sha:
        raise ValueError('original source changed during recovery')
    return {'status':'pass','run_id':run_id,'sealed_tree_sha256':seal_sha,
        'run_complete_sha256':sha(recovery/'run-complete.json'),
        'process_registry_sha256':original_sha}
