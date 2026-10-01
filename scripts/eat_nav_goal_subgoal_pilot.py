"""Resume-safe fresh-process, 44-run goal-conditioned subgoal pilot."""
from __future__ import annotations
import argparse,gzip,hashlib,json,pathlib,subprocess,time,traceback
import numpy as np
from visualnav_transformer.eat_nav.candidate_coverage_pilot import map_paths_from_anchor
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120
from eat_nav_goal_subgoal_run import ROOT,DOC,INPUT,REPO,PROD,FOLL,PARAM,sha,read,atomic,run_one,active_processes

LIMIT_SEC=4*3600
LIMIT_BYTES=800_000_000
SOURCE=REPO/'scripts/eat_nav_goal_subgoal_sim.py'
RUNS=read(DOC/'run-identities.json')['runs']

def check_frozen():
    freeze=read(DOC/'pilot-input-freeze.json')
    assert sha(DOC/'run-identities.json')==freeze['run_identities_sha256']
    assert sha(INPUT/'selected-states.json')==freeze['selected_state_sha256']
    assert sha(REPO/'docs/quality-identifiability-gate/candidate-coverage/goal-ood/simulator-pilot-proposal.json')==freeze['proposal_sha256']
    assert sha(REPO/'docs/quality-identifiability-gate/candidate-coverage/goal-ood/goal-sweep-inputs.json')==freeze['offline_goal_sweep_inputs_sha256']
    assert sha(REPO/'docs/quality-identifiability-gate/candidate-coverage/goal-ood/artifact-sha-ledger.json')==freeze['offline_ledger_sha256']
    assert sha('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json')==freeze['manifest_sha256']
    assert sha('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')==freeze['reference_checkpoint_sha256']
    assert sha(SOURCE)==read(DOC/'parity-smoke.json')['logging_sim_source_sha256']
    assert read(DOC/'parity-smoke.json')['status']=='pass'
    assert len(RUNS)==44 and [r['order'] for r in RUNS]==list(range(44))
    assert [r['kind'] for r in RUNS]==(['duplicate']*3+['candidate']*8)*4
    states=read(INPUT/'selected-states.json')['train'];assert len(states)==4
    metadata=read(DOC/'goal-metadata-sha.json')
    assert metadata['run_identities_sha256']==sha(DOC/'run-identities.json')
    for r in RUNS:
        folder=INPUT/f"state-{r['state_index']:03d}"
        assert sha(folder/f"{r['candidate_id']}-base-path.json")==r['base_path_sha256']
        assert sha(folder/f"{r['run_id']}-goal-metadata.json")==metadata['runs'][r['order']]['metadata_sha256']
        assert r['logical_t0_ns']==2_000_000_000_000+r['state_index']*100_000_000_000
        assert tuple(r['episode_key'])==tuple(states[r['state_index']]['episode_key'])
    assert not active_processes()
    return freeze

def audit():
    freeze=check_frozen();old=read(REPO/'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot/analysis-thresholds.json')
    checkpoint=read(REPO/'docs/quality-identifiability-gate/candidate-coverage/goal-ood/source-provenance.json')
    assert old['planned_path_cluster_rms_m']==old['realized_trajectory_cluster_rms_m']==.05
    assert old['oracle_delta_v_epsilon']==.01
    assert sha(PROD)=='616097e44d199b8d33f6534aedd299f2d20127ed8e100c18777f9ca7a2da389d'
    assert sha(FOLL)=='dda18662f450a79d953d8797be8c7df04754c883ef8bd7a64358dce0ada65201'
    assert sha(PARAM)=='ddedff40d13e20416a43309669ad19e7f3a06e2df8cc8c9d3248369e14d40c90'
    paths={'new_sim_logging_source':SOURCE,'new_host_runner':REPO/'scripts/eat_nav_goal_subgoal_run.py',
        'bridge_source':REPO/'scripts/eat_nav_command_delivery_bridge.py',
        'exact_clock_protocol':REPO/'src/visualnav_transformer/eat_nav/exact_stamp_protocol.py',
        'production_follower_header':pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/include/simple_trajectory_follower/simple_trajectory_follower.hpp'),
        'production_follower_cpp':pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src/simple_trajectory_follower.cpp'),
        'exact_follower_binary':PROD,'pilot_follower_binary':FOLL,'follower_config':PARAM,
        'prior_thresholds':REPO/'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot/analysis-thresholds.json'}
    provenance={'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        'base_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'pilot_input_freeze_sha256':sha(DOC/'pilot-input-freeze.json'),'run_identity_sha256':sha(DOC/'run-identities.json'),
        'offline_proposal_sha256':freeze['proposal_sha256'],'offline_ledger_sha256':freeze['offline_ledger_sha256'],
        'manifest_sha256':freeze['manifest_sha256'],'reference_nomad_checkpoint_sha256':freeze['reference_checkpoint_sha256'],
        'rl_checkpoint_sha256':'b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d',
        'split_hash':freeze['split_hash'],'logical_clock_protocol':'exact-stamp simulation-only /clock, TF, Path and command at 20 Hz',
        'existing_thresholds':old,'goal_tolerance_m':.2,'delta_v_epsilon':.01,
        'source_file_sha256':{name:sha(path) for name,path in paths.items()},
        'parity_smoke_sha256':sha(DOC/'parity-smoke.json'),'selected_state_ids':[s['state_id'] for s in read(INPUT/'selected-states.json')['train']],
        'new_simulator_rollouts_started':False,'heldout_used':False,'nomad_reinference':False}
    atomic(DOC/'source-provenance.json',provenance)
    logging={'status':'parity_pass','step_identity':'state_id, run_id, control_step, logical_stamp_ns',
        'pre_step_fields':['DEM history','mission goal','local goal','planned Path','follower command','pre-noise Actor observation',
        'post-noise pre-delay observation','post-delay Actor observation','latent','action','previous action','root/joint','raw effort',
        'scaled/clipped effort','Critic observation/value','contact','DEM/TF/command stamps'],
        'post_step_fields':['reward','post root state','termination reason'],
        'control_or_model_change':False,'parity_sha256':sha(DOC/'parity-smoke.json')}
    atomic(DOC/'logging-schema-audit.json',logging)
    parity=read(DOC/'parity-smoke.json')
    raw={'source_tensor':'robot.data.applied_torque','capture_phase':'pre env.step, after prior physics and before Actor action',
        'field_order':'robot.joint_names, 52 elements; four flippers then right/left track sums form 6 Actor effort terms',
        'raw_joint_count':parity['raw_joint_effort_dim'],'actor_effort_count':parity['raw_effort_grouped_dim'],
        'observation_processing':'clip raw grouped effort to [-100,100], then scale by 0.01; noise and policy delay follow',
        'raw_to_pre_noise_max_abs':parity['raw_to_prenoise_effort_max_abs'],
        'finite_check':'per step in diagnostic capture','same_control_step':'pre-step raw tensor and captured Actor input share step identity',
        'online_real_robot_equivalent':'not yet verified; motor/driver telemetry must expose ordered joint effort or torque',
        'parity_sha256':sha(DOC/'parity-smoke.json')}
    atomic(DOC/'raw-joint-effort-audit.json',raw)
    atomic(ROOT/'s0-audit.done.json',{'status':'pass','source_provenance_sha256':sha(DOC/'source-provenance.json'),
        'logging_schema_sha256':sha(DOC/'logging-schema-audit.json'),'raw_effort_sha256':sha(DOC/'raw-joint-effort-audit.json')})
    print('S0_AUDIT_PASS',flush=True)

def map_paths(j,anchor_summary,anchor_name):
    folder=INPUT/f'state-{j:03d}'
    candidates=[r for r in RUNS if r['state_index']==j and r['kind']=='candidate']
    base={r['candidate_id']:read(folder/f"{r['candidate_id']}-base-path.json") for r in candidates}
    nominal=read(folder/'logged-seed-1001-base-path.json')
    mapped=map_paths_from_anchor(anchor_summary,{'bank-nominal':nominal,**base})
    existing=folder/'logged-seed-1001-map-path.json'
    subprocess.run(['docker','cp',f'isaac-lab-onix:{ROOT}/inputs/state-{j:03d}/logged-seed-1001-map-path.json',str(existing)],check=True)
    if np.max(abs(np.asarray(read(existing))-np.asarray(mapped['bank-nominal'])))>1e-6:raise ValueError('anchor map Path differs')
    for name,points in mapped.items():
        if name=='bank-nominal':continue
        path=folder/f'{name}-map-path.json'
        if path.exists():
            if np.max(abs(np.asarray(read(path))-np.asarray(points)))>1e-6:raise ValueError('existing map Path differs')
        else:atomic(path,points)
        subprocess.run(['docker','cp',str(path),f'isaac-lab-onix:{ROOT}/inputs/state-{j:03d}/{path.name}'],check=True)
    target=ROOT/'map-path-freeze'/f"{anchor_summary['state']['state_id']}.json"
    atomic(target,{'anchor_marker_sha256':sha(ROOT/'runs'/anchor_name/'run-complete.json'),
        'map_path_sha256':{name:sha(folder/f'{name}-map-path.json') for name in base}})
    return target

def verify_complete(entry):
    folder=ROOT/'runs'/entry['run_id'];marker=folder/'run-complete.json'
    if not marker.exists():
        if folder.exists():raise RuntimeError('partial run; do not retry automatically: '+str(folder))
        return False
    data=read(marker)
    if data['identity']!=entry:raise ValueError('run identity changed')
    for name,digest in data['output_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('run output SHA mismatch '+entry['run_id'])
    return True

def trace(entry):
    folder=ROOT/'runs'/entry['run_id']
    with gzip.open(folder/'live-0.trace.json.gz','rt') as f:return json.load(f)

def duplicate_gate(j,entries):
    summaries=[read(ROOT/'runs'/e['run_id']/'live-0.json') for e in entries]
    traces=[trace(e) for e in entries]
    result=duplicate_120(summaries,traces,.0105)
    # Critical exact sequence checks supplement the frozen 0.0105 m XY limit.
    signals=('follower_command','command','actor_input','actor_latent','actor_action','critic_observation','critic_value',
             'reward','contact_force','dem','logical_sim_time_ns','tf_stamp_ns','follower_command_stamp_ns')
    first=None
    for k in range(120):
        for signal in signals:
            if any(t[k][signal]!=traces[0][k][signal] for t in traces[1:]):first={'step':k,'signal':signal};break
        if first:break
    result.update({'exact_signals':list(signals),'first_difference':first,
       'anchor_root_joint_exact':all(s['anchor']['root_state']==summaries[0]['anchor']['root_state'] and
           s['anchor']['joint_position']==summaries[0]['anchor']['joint_position'] and
           s['anchor']['joint_velocity']==summaries[0]['anchor']['joint_velocity'] for s in summaries[1:]),
       'logical_stamp_equal':len({e['logical_t0_ns'] for e in entries})==1,
       'complete_marker_sha256':[sha(ROOT/'runs'/e['run_id']/'run-complete.json') for e in entries]})
    result['pass']=bool(result['pass'] and result['dem_sha_equal'] and first is None and result['anchor_root_joint_exact'] and result['logical_stamp_equal'])
    target=DOC/'duplicate-gates.json';prior=read(target) if target.exists() else {'states':{}}
    prior['states'][entries[0]['state_id']]=result
    atomic(target,prior)
    if not result['pass']:raise RuntimeError('duplicate gate failed '+entries[0]['state_id']+' '+str(first))
    print('DUPLICATE_PASS',entries[0]['state_id'],result['xy_rms_m'],flush=True)

def storage_bytes():
    return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and 'parity/' not in str(p))

def status(stage,started,completed,**extra):
    elapsed=time.monotonic()-started
    atomic(ROOT/'status.json',{'stage':stage,'completed_runs':completed,'candidate_runs':sum(verify_complete(r) for r in RUNS if r['kind']=='candidate'),
        'elapsed_sec':elapsed,'remaining_wall_sec':max(0,LIMIT_SEC-elapsed),'storage_bytes':storage_bytes(),**extra})

def execute(resume=True):
    check_frozen()
    if not (ROOT/'s0-audit.done.json').exists():raise ValueError('S0 audit missing')
    start_marker=ROOT/'pilot-start.json'
    if start_marker.exists():
        original=read(start_marker)
        if original['identity_sha256']!=sha(DOC/'run-identities.json') or original['limit_sec']!=LIMIT_SEC:
            raise ValueError('pilot start marker drift')
        if original['source_sha256']!=sha(SOURCE):
            revision=read(ROOT/'anchor-diagnostic-revision.json')
            if revision['candidate_runs_before_revision']!=0:raise ValueError('source changed after candidate')
            atomic(ROOT/'source-revision.json',{'original_source_sha256':original['source_sha256'],
                'revised_source_sha256':sha(SOURCE),'reason':revision['failure'],
                'candidate_runs_before_revision':0})
    else:
        atomic(start_marker,{'started_unix':time.time(),'limit_sec':LIMIT_SEC,'run_count':44,
            'identity_sha256':sha(DOC/'run-identities.json'),'source_sha256':sha(SOURCE)})
        original=read(start_marker)
    started=time.monotonic()-(time.time()-original['started_unix']);done=0;durations=[]
    for j in range(4):
        group=RUNS[j*11:(j+1)*11];state=group[0]['state_id'];anchor_name=f'{state}-anchor-v2' if j==0 else f'{state}-anchor'
        anchor_entry={**group[0],'kind':'anchor','run_id':anchor_name,'order':-1}
        anchor_dir=ROOT/'runs'/anchor_name
        if (anchor_dir/'run-complete.json').exists():
            marker=read(anchor_dir/'run-complete.json')
            if marker['identity']!=anchor_entry:raise ValueError('anchor identity changed')
            summary=read(anchor_dir/'anchor-0.json')
        else:
            if anchor_dir.exists():raise RuntimeError('partial anchor; stop')
            status('S2_ANCHOR_RUNNING',started,done,state_id=state)
            run_one(anchor_entry,anchor_name,timeout_sec=900)
            summary=read(anchor_dir/'anchor-0.json')
        if not summary['anchor_error_vs_logged']['dem_history_sha_match']:raise RuntimeError('anchor DEM history SHA mismatch')
        map_paths(j,summary,anchor_name)
        for entry in group[:3]:
            if verify_complete(entry):done+=1;continue
            if time.monotonic()-started>=LIMIT_SEC:raise TimeoutError('4h wall cap before duplicate')
            status('S2_DUPLICATE_RUNNING',started,done,state_id=state,run_id=entry['run_id'])
            t=time.monotonic();run_one(entry,entry['run_id'],timeout_sec=min(900,LIMIT_SEC-(time.monotonic()-started)))
            durations.append(time.monotonic()-t);done+=1
            if storage_bytes()>LIMIT_BYTES:raise RuntimeError('0.8 GB storage cap')
        duplicate_gate(j,group[:3])
        status('S2_DUPLICATE_PASS',started,done,state_id=state)
        for entry in group[3:]:
            if verify_complete(entry):done+=1;continue
            elapsed=time.monotonic()-started;remaining=44-done
            mean=np.median(durations) if durations else 185.
            predicted=elapsed+remaining*mean
            if elapsed>=3*3600:
                atomic(ROOT/'three-hour-forecast.json',{'elapsed_sec':elapsed,'completed_runs':done,
                    'median_run_sec':float(mean),'predicted_total_sec':float(predicted),'limit_sec':LIMIT_SEC})
            if elapsed>=LIMIT_SEC or predicted>LIMIT_SEC:raise TimeoutError('4h wall forecast; no new candidate')
            if storage_bytes()>LIMIT_BYTES:raise RuntimeError('0.8 GB storage cap')
            status('S3_CANDIDATE_RUNNING',started,done,state_id=state,run_id=entry['run_id'],predicted_total_sec=float(predicted))
            t=time.monotonic();run_one(entry,entry['run_id'],timeout_sec=min(900,LIMIT_SEC-elapsed))
            durations.append(time.monotonic()-t);done+=1
            if storage_bytes()>LIMIT_BYTES:raise RuntimeError('0.8 GB storage cap')
            status('S3_CANDIDATE_COMPLETE',started,done,state_id=state,run_id=entry['run_id'])
    if done!=44:raise RuntimeError('incomplete live run count')
    status('RAW_COMPLETE',started,done)
    atomic(ROOT/'pilot-runs.done.json',{'status':'pass','run_count':44,'elapsed_sec':time.monotonic()-started,
        'run_identity_sha256':sha(DOC/'run-identities.json')})
    print('PILOT_44_RAW_COMPLETE',time.monotonic()-started,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('audit','run'),required=True);args=p.parse_args()
    try:
        if args.stage=='audit':audit()
        else:execute()
    except BaseException as exc:
        atomic(ROOT/'status.json',{'stage':'STOPPED','error':repr(exc),'traceback':traceback.format_exc(),
            'candidate_runs':sum((ROOT/'runs'/r['run_id']/'run-complete.json').exists() for r in RUNS if r['kind']=='candidate')})
        raise
