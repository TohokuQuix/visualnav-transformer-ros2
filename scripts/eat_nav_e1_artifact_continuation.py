"""Resume only the frozen 17 E1 rows after verified attempt-3 artifact recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import eat_nav_e1_attempt2 as flow
import eat_nav_e1_schedule as schedule
import eat_nav_e1_run_one as replay
from eat_nav_e1_artifact_recovery import atomic,sha,verify_seal
from eat_nav_e1_recovery_continue import configure_output_root,check_router,install_transport_guard

ATTEMPT3=Path('/tmp/eatnav-e1-attempt3-20261004')
RECOVERY=Path('/tmp/eatnav-e1-artifact-recovery-20261004/artifact-recovery/attempt3-direct-repeat-00')
ROOT=Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004')
FIRST='g0e-train-000-nominal_duplicate-00'


def configure():
    configure_output_root(ROOT)
    replay.REMOTE_CODE=str(ROOT/'eat_nav_e1_sim.py')
    flow.ROOT=ROOT
    schedule.SESSION='eatnav-e1-artifact-recovered'


def prepare():
    configure()
    recovery_seal=verify_seal(RECOVERY)
    if ROOT.exists():raise FileExistsError('continuation root already prepared')
    ROOT.mkdir()
    names=('continuation-manifest.json','original-20-row-manifest.json','e1-gates.yaml',
        'e1-safety-stops.yaml','map-path-identities.jsonl.gz','completed-pending-identity.json',
        'input-freeze.json')
    for name in names:
        shutil.copyfile(ATTEMPT3/name,ROOT/name)
        if sha(ROOT/name)!=sha(ATTEMPT3/name):raise ValueError('frozen input copy SHA differs '+name)
    shutil.copytree(ATTEMPT3/'inputs',ROOT/'inputs')
    for original in (ATTEMPT3/'inputs').rglob('*'):
        if original.is_file() and sha(original)!=sha(ROOT/original.relative_to(ATTEMPT3)):
            raise ValueError('input copy SHA differs '+str(original))
    (ROOT/'runs').mkdir()
    (ROOT/'runs'/FIRST).symlink_to(RECOVERY,target_is_directory=True)
    rows=json.loads((ROOT/'continuation-manifest.json').read_text())['pending_rows']
    if len(rows)!=18 or rows[0]['process_id']!=FIRST or rows[1]['process_id']!='g0e-train-000-nominal_duplicate-01':
        raise ValueError('frozen first/second row identity differs')
    if schedule.completed_rows(rows)!={FIRST}:raise ValueError('recovered run not sole completed row')
    atomic(ROOT/'recovery-adoption.json',{'status':'pass','recovered_protocol_valid':1,
        'partial_second_run_source':str(ATTEMPT3/'runs'/rows[1]['process_id']),
        'partial_second_run_measurement_invalid':True,'historical_invalid_attempts':[0,1,2],
        'recovery_set':str(RECOVERY),'recovery_seal_sha256':recovery_seal,
        'original_attempt3_unchanged':True})
    return rows


def static_preflight():
    configure()
    rows=json.loads((ROOT/'continuation-manifest.json').read_text())['pending_rows']
    if len(rows)!=18 or schedule.completed_rows(rows)!={FIRST}:
        raise ValueError('recovery count or schedule differs')
    frozen=verify_seal(RECOVERY)
    freeze=schedule.freeze_sources()
    if freeze['dirty']:raise ValueError('source worktree is dirty')
    schedule.setup_container()
    table=[]
    for row in rows[1:]:
        entry=schedule.execution_entry(row)
        output=ROOT/'runs'/row['process_id']
        if output.exists():raise FileExistsError('fresh run output already exists '+str(output))
        cmd=replay.simulator_command(entry,str(output),timing_telemetry=True)
        files=flow.audit_cli(cmd,freeze)
        if cmd[cmd.index('--output')+1]!=str(output):raise ValueError('output root differs')
        if entry['local_path_sha256']!=row['path_sha256'] or entry['map_path_sha256']!=row['map_path_sha256']:
            raise ValueError('Path SHA differs')
        table.append({'order':row['order'],'run_id':row['process_id'],
            'state_id':row['state_group_id'],'candidate_id':row['path_id'],
            'seed':row['seed'],'episode_key':entry['episode_key'],
            'logical_t0_ns':entry['logical_t0_ns'],'host_output':str(output),
            'container_output':str(output),'command':cmd,'input_files':files})
    if verify_seal(RECOVERY)!=frozen:raise ValueError('recovery set mutated during preflight')
    result={'status':'pass','recovered_first':FIRST,'remaining_count':17,
        'remaining_order':[x['run_id'] for x in table],
        'manifest_sha256':sha(ROOT/'continuation-manifest.json'),
        'input_freeze_sha256':sha(ROOT/'input-freeze.json'),
        'source_freeze_sha256':sha(ROOT/'source-freeze.json'),
        'recovery_seal_sha256':frozen,'rows':table}
    atomic(ROOT/'artifact-recovered-static-preflight.json',result)
    return result


def continue_runs():
    configure()
    preflight=json.loads((ROOT/'artifact-recovered-static-preflight.json').read_text())
    if preflight['status']!='pass' or len(preflight['rows'])!=17:
        raise ValueError('static preflight missing')
    if verify_seal(RECOVERY)!=preflight['recovery_seal_sha256']:
        raise ValueError('recovered tree changed')
    readiness=json.loads((ROOT/'router-readiness.json').read_text())
    if readiness['gate']['status']!='pass':raise ValueError('router readiness missing')
    check_router(ROOT)
    marker=ROOT/'artifact-recovered-continuation-started.marker.json'
    if marker.exists():raise FileExistsError('continuation clock already started')
    started=time.time_ns()
    atomic(marker,{'start_wall_ns':started,'pending_valid_processes':17,
        'recovered_protocol_valid_processes':1,'historical_invalid_attempts':[0,1,2],
        'partial_attempt3_second_excluded':True})
    atomic(ROOT/'continuation-started.marker.json',{'start_wall_ns':started,
        'pending_processes':17,'artifact_recovered_clock_sha256':sha(marker)})
    install_transport_guard(ROOT)
    try:
        schedule.run()
        if verify_seal(RECOVERY)!=preflight['recovery_seal_sha256']:
            raise ValueError('recovered tree changed after schedule')
        atomic(ROOT/'final-decision.json',{'decision':'artifact_recovered_e1_complete',
            'counts':schedule.process_counts(ROOT),'historical_invalid_attempts':[0,1,2],
            'recovered_protocol_valid_processes':1,'new_protocol_valid_processes':17,
            'partial_measurement_invalid_processes':1,'learning_started':False})
    except BaseException as exc:
        counts=schedule.process_counts(ROOT)
        decision=('artifact_recovered_e1_duplicate_gate_fail' if 'duplicate gate' in str(exc)
            else 'artifact_recovered_e1_runtime_stop')
        atomic(ROOT/'final-decision.json',{'decision':decision,'reason':repr(exc),
            'traceback':traceback.format_exc(),'counts':counts,
            'recovered_protocol_valid_processes':1,
            'new_protocol_valid_processes':max(0,counts['protocol_valid_process_count']-1),
            'partial_measurement_invalid_processes':1,'learning_started':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','preflight','continue'])
    args=parser.parse_args()
    if args.stage=='prepare':print(json.dumps({'rows':len(prepare())}),flush=True)
    elif args.stage=='preflight':print(json.dumps({'status':static_preflight()['status']}),flush=True)
    else:continue_runs()
