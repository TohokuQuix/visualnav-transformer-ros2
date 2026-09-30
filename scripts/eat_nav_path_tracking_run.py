"""CPU-only, SHA-gated tracking audit of frozen readiness rollouts."""
from __future__ import annotations
import argparse,gzip,hashlib,json,pathlib,subprocess,time
from visualnav_transformer.eat_nav.path_tracking_audit import audit_run,select_live_entries,validate_source_hashes,_sha

REPO=pathlib.Path('/tmp/eatnav-vnt-candidate-coverage')
PRIOR=REPO/'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot'
DOC=REPO/'docs/quality-identifiability-gate/candidate-coverage/path-tracking-audit'
ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/path-tracking-audit')
UNIFIED=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified')
INPUT=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')


def read(path):return json.loads(pathlib.Path(path).read_text())

def atomic(path,data,*,once=False):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists() and once:
        if path.read_text()!=content:raise FileExistsError('frozen output changed '+str(path))
        return
    temp=path.with_name(path.name+'.tmp');temp.write_text(content);temp.replace(path)


def identity_table(rows):
    target=DOC/'rollout-identity-table.jsonl.gz';target.parent.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in rows).encode()
    if target.exists():
        with gzip.open(target,'rb') as stream:
            if stream.read()!=content:raise FileExistsError('identity table changed')
        return _sha(target)
    temp=target.with_suffix('.tmp')
    with gzip.GzipFile(filename='',mode='wb',fileobj=temp.open('wb'),mtime=0) as stream:stream.write(content)
    temp.replace(target)
    return _sha(target)


def stage_a0():
    started=time.monotonic();ROOT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    if read(UNIFIED/'status.json')['stage']!='RAW_COMPLETE':raise ValueError('frozen 119-process pilot incomplete')
    schedule=read(PRIOR/'schedule.json')['runs'];live,anchors=select_live_entries(schedule)
    for sid in sorted({e['state_id'] for e in live}):
        nominal=[e for e in live if e['state_id']==sid and e['kind']=='nominal']
        if len(nominal)!=3 or len({e['base_path_sha256'] for e in nominal})!=1:
            raise ValueError('nominal repeat Path identity mismatch '+sid)
    ledger=read(PRIOR/'artifact-sha-ledger.json');by_run={x['run_id']:x for x in ledger['runs']}
    if len(by_run)!=119 or ledger['raw_status_sha256']!=_sha(UNIFIED/'status.json'):
        raise ValueError('prior artifact ledger mismatch')
    gate=read(PRIOR/'fresh-bridge-gate-ledger.json')
    if gate['sha256']['unified_source_ledger']!=_sha(UNIFIED/'source-ledger.json'):
        raise ValueError('fresh bridge source ledger mismatch')
    freeze=read(INPUT/'input-freeze.json')
    if freeze['selection_sha256']!=_sha(PRIOR/'selection.json'):
        raise ValueError('selection SHA mismatch')
    identities=[];coordinates=[];origins={};source_hashes={}
    for entry in live:
        result=audit_run(entry,UNIFIED/'runs'/entry['run_id'],INPUT/f"state-{entry['state_index']:03d}",by_run[entry['run_id']])
        identities.append(result['identity']);coordinates.append(result['coordinate'])
        sid=entry['state_id'];origin=result['coordinate']['env_origin_world_xy']
        if sid in origins and max(abs(a-b) for a,b in zip(origins[sid],origin))>1e-6:
            raise ValueError('env origin not stable for '+sid)
        origins[sid]=origin
        for name,digest in result['summary']['source_sha256'].items():source_hashes.setdefault(name,set()).add(digest)
    for entry in anchors:
        path=UNIFIED/'runs'/entry['run_id'];marker=read(path/'run-complete.json')
        if marker['identity']!=entry or _sha(path/'run-complete.json')!=by_run[entry['run_id']]['marker_sha256']:
            raise ValueError('anchor identity/SHA mismatch')
        summary=read(path/'anchor-0.json')
        identities.append({'run_id':entry['run_id'],'state_id':entry['state_id'],'terrain':entry['terrain_type'],
            'episode_id':entry['episode_key'],'generator':'anchor','candidate_id':entry['candidate_id'],
            'diffusion_seed':None,'base_path_sha256':entry['base_path_sha256'],
            'map_path_sha256':None,'trace_sha256':None,'complete_marker_sha256':_sha(path/'run-complete.json'),
            'anchor_pose_world':summary['anchor']['root_state'][:7],
            'path_point_count':None,'future_point_count':None,'path_sampling_interval_sec':.25,
            'path_horizon_sec':2.,'run_complete':True,'included':False,'exclusion_reason':'anchor_no_live_outcome'})
    if len(identities)!=119 or sum(x['included'] for x in identities)!=112:
        raise ValueError('identity table support mismatch')
    identities.sort(key=lambda x:x['run_id'])
    table_sha=identity_table(identities)
    validate_source_hashes(source_hashes)
    provenance={'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        'base_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'schedule_sha256':_sha(PRIOR/'schedule.json'),'artifact_sha_ledger_sha256':_sha(PRIOR/'artifact-sha-ledger.json'),
        'fresh_bridge_gate_ledger_sha256':_sha(PRIOR/'fresh-bridge-gate-ledger.json'),
        'final_decision_sha256':_sha(PRIOR/'final-decision.json'),
        'unified_status_sha256':_sha(UNIFIED/'status.json'),'unified_source_ledger_sha256':_sha(UNIFIED/'source-ledger.json'),
        'input_freeze_sha256':_sha(INPUT/'input-freeze.json'),
        'nomad_checkpoint_sha256':freeze['reference_checkpoint_sha256'],
        'rl_checkpoint_sha256':next(iter(source_hashes['rl_checkpoint'])),
        'follower_diagnostic_binary_sha256':_sha('/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery/follower-build/simple_trajectory_follower_node'),
        'follower_production_binary_sha256':_sha('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node'),
        'follower_parameter_sha256':_sha('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'),
        'follower_source_sha256':{name:_sha(pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower')/name) for name in
            ('include/simple_trajectory_follower/simple_trajectory_follower.hpp','src/simple_trajectory_follower.cpp')},
        'recorded_source_sha256':{name:sorted(values) for name,values in source_hashes.items()},
        'excluded_non_schedule_sources':['protocol equivalence replay','fault injection','diagnostic replay','old failed attempt'],
        'statistical_unit':'state','validation_used':False,'heldout_used':False,'new_rollouts':False,
        'identity_table_sha256':table_sha}
    atomic(DOC/'source-provenance.json',provenance,once=True)
    time_audit={'included_live_runs':112,'excluded_anchors':7,
        'coordinate_rule':'map_path is env-local; root_state is world; world XY = env-local XY + verified env origin',
        'step_rule':'trace[k] Actor and command use pre-step root[k]; post_root_state[k] equals root_state[k+1]',
        'env_origins_world_xy':origins,
        'max_map_base_roundtrip_error_m':max(x['map_base_roundtrip_max_abs_m'] for x in coordinates),
        'max_increment_roundtrip_error_m':max(x['increment_roundtrip_max_abs_m'] for x in coordinates),
        'max_anchor_root_first_step_error':max(x['anchor_root_first_step_max_abs'] for x in coordinates),
        'max_command_recording_error':max(x['command_max_abs'] for x in coordinates),
        'stamp_mismatch_count':sum(x['stamp_mismatch_count'] for x in coordinates),
        'control_period_sec':.05,'path_sampling_interval_sec':.25,'path_horizon_sec':2.,
        'path_coordinate_frame':'simulation env-local map','root_coordinate_frame':'simulation world',
        'runs':coordinates}
    atomic(DOC/'coordinate-time-audit.json',time_audit,once=True)
    atomic(ROOT/'a0.done.json',{'stage':'A0_PASS','identity_table_sha256':table_sha,
        'source_provenance_sha256':_sha(DOC/'source-provenance.json'),
        'coordinate_time_sha256':_sha(DOC/'coordinate-time-audit.json'),
        'elapsed_sec':time.monotonic()-started})
    atomic(ROOT/'status.json',{'stage':'A0_PASS','included_live_runs':112,'excluded_anchors':7,
        'elapsed_sec':time.monotonic()-started,'next':'A1_A2'})
    return time_audit

def stage_dispatch(stage):
    from eat_nav_path_tracking_stages import stage_a2,stage_a3,stage_a4,stage_a5,stage_a6,stage_candidate_features,stage_endpoint_timing
    stages={'a0':stage_a0,'a2':stage_a2,'a3':stage_a3,'a4':stage_a4,'a5':stage_a5,'a6':stage_a6,
            'features':stage_candidate_features,'timing':stage_endpoint_timing}
    if stage in stages:return stages[stage]()
    if stage=='figures':
        from eat_nav_path_tracking_figures import main
        return main()
    if stage=='final':
        from eat_nav_path_tracking_finalize import main
        return main()
    raise ValueError(stage)

def verify_existing_stage(stage):
    marker_path=ROOT/f'{stage}.done.json'
    if not marker_path.exists():return False
    marker=read(marker_path)
    mapping={'a0':{'source_provenance_sha256':DOC/'source-provenance.json','coordinate_time_sha256':DOC/'coordinate-time-audit.json'},
        'a2':{'metrics_sha256':DOC/'path-tracking-metrics.json','series_sha256':ROOT/'tracking-series.jsonl.gz','endpoint_sha256':DOC/'endpoint-reach-summary.json'},
        'a3':{'distance_sha256':DOC/'path-command-realized-distances.json','transition_sha256':DOC/'mode-transition-table.json','classification_sha256':DOC/'collapse-classification.json'},
        'a4':{'outcome_sha256':DOC/'tracking-outcome-analysis.json'},'a5':{'strata_sha256':DOC/'evaluator-tracking-strata.json'},
        'a6':{'schema_sha256':DOC/'evaluator-input-schema-audit.json'}}
    for key,path in mapping.get(stage,{}).items():
        if not path.exists() or marker.get(key)!=_sha(path):raise ValueError('resume marker SHA drift '+stage+' '+key)
    if stage=='a0' and read(PRIOR/'artifact-sha-ledger.json')['raw_status_sha256']!=_sha(UNIFIED/'status.json'):
        raise ValueError('resume input status SHA drift')
    return True

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('a0','a2','a3','a4','a5','a6','features','timing','figures','final','all'),required=True)
    parser.add_argument('--resume',action='store_true',help='verify completed stage markers and skip unchanged work')
    args=parser.parse_args()
    order=('a0','a2','a3','a4','a5','a6','features','timing','figures','final')
    try:
        for stage in (order if args.stage=='all' else (args.stage,)):
            if args.resume and stage in ('a0','a2','a3','a4','a5','a6') and verify_existing_stage(stage):
                print('RESUME_SKIP',stage,flush=True);continue
            stage_dispatch(stage);print('STAGE_COMPLETE',stage,flush=True)
    except BaseException as exc:
        atomic(ROOT/'status.json',{'stage':'STOPPED','stop_reason':repr(exc),'stage_failed':stage})
        raise
