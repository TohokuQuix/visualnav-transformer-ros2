"""Read-only source and candidate identity gate for feasible transfer analysis."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import yaml
from visualnav_transformer.eat_nav.feasible_transfer import validate_candidate_identity
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest

ROOT=Path('/tmp/eatnav-candidate-coverage')
OUT=ROOT/'feasible-transfer'
WORKTREE=Path('/tmp/eatnav-vnt-candidate-coverage')
CONFIG=WORKTREE/'docs/quality-identifiability-gate/candidate-coverage/feasible-transfer/analysis-config.yaml'
CONTAINER_FILES={
    'nomad_reference_best_ema':'/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt',
    'evaluator_GT_11':'/tmp/eatnav-quality-identifiability-gate/forward-evaluator/checkpoints/GT-11/best.pt',
    'evaluator_GT_29':'/tmp/eatnav-quality-identifiability-gate/forward-evaluator/checkpoints/GT-29/best.pt',
    'evaluator_GT_47':'/tmp/eatnav-quality-identifiability-gate/forward-evaluator/checkpoints/GT-47/best.pt',
    'evaluator_train_scalers':'/tmp/eatnav-quality-identifiability-gate/relative-quality/output/relative-quality-extraction.json',
    'phase_b_heldout_results':'/tmp/eatnav-quality-identifiability-gate/forward-evaluator/heldout-results.json',
    'phase_b_test_split':'/tmp/eatnav-quality-identifiability-gate/relative-quality/splits/test.npz',
    'phase_b_config':'/tmp/eatnav-train/forward-evaluator-config.yaml',
}


def container_hashes():
    process=subprocess.run(['docker','exec','isaac-lab-onix','sha256sum',*CONTAINER_FILES.values()],
                           capture_output=True,text=True,check=True)
    by_path={line.split()[1]:line.split()[0] for line in process.stdout.splitlines()}
    if set(by_path)!=set(CONTAINER_FILES.values()):raise ValueError('container source list mismatch')
    return {name:{'path':path,'sha256':by_path[path]} for name,path in CONTAINER_FILES.items()}


def read_json(path):return json.loads(path.read_text())


def run():
    started=time.time()
    spec=yaml.safe_load(CONFIG.read_text())
    selection_path=ROOT/'stage0/frozen/pilot-selection.json'
    schedule_path=ROOT/'stage0/frozen/pilot-schedule.json'
    outcome_path=ROOT/'analysis/oracle-outcomes.json'
    summary_path=ROOT/'analysis/oracle-summary.json'
    post_path=ROOT/'analysis/postprocess-complete.json'
    selection=read_json(selection_path)['selected']
    schedule=read_json(schedule_path)['runs']
    outcomes=read_json(outcome_path)['rows']
    post=read_json(post_path)
    if len(selection)!=7 or len(schedule)!=119 or len(outcomes)!=91 or spec['candidate_count']!=91:
        raise ValueError('frozen source count mismatch')
    for name,sha in post['outputs_sha256'].items():
        if digest(ROOT/'analysis'/name)!=sha:raise ValueError(f'postprocess SHA mismatch: {name}')
    offline_files={state['state_id']:ROOT/'stage1'/f"{state['state_id']}.json" for state in selection}
    offline={state_id:read_json(path) for state_id,path in offline_files.items()}
    offline_by_state={state_id:{c['candidate_id']:c for c in data['candidates']}
                      for state_id,data in offline.items()}
    validate_candidate_identity(outcomes,schedule,offline_by_state)
    hashes=container_hashes()
    if hashes['nomad_reference_best_ema']['sha256']!='191d6a03266bca2ce8d43afac44b04828ae0d2571632be4f47a5f0dc3564b4e0':
        raise ValueError('NoMaD checkpoint mismatch')
    prior=read_json(ROOT/'analysis/evaluator-ranking.json')['evaluator_checkpoint_sha256']
    for seed in (11,29,47):
        if hashes[f'evaluator_GT_{seed}']['sha256']!=prior[str(seed)]:
            raise ValueError(f'Evaluator checkpoint seed {seed} mismatch')
    if hashes['evaluator_train_scalers']['sha256']!=read_json(ROOT/'analysis/evaluator-ranking.json')['evaluator_scaler_sha256']:
        raise ValueError('Evaluator normalization mismatch')
    expected_seeds=spec['nomad_seeds_k32']
    for state_id,data in offline.items():
        candidates={c['candidate_id']:c for c in data['candidates']}
        if len(candidates)!=37 or len(data['candidates'])!=37:
            raise ValueError(f'K32 candidate count mismatch: {state_id}')
        for seed in expected_seeds:
            cid=f'nomad-seed-{seed}'
            if cid not in candidates or candidates[cid]['seed']!=seed:
                raise ValueError(f'K32 seed identity mismatch: {state_id} {seed}')
        if data['state']['state_id']!=state_id:raise ValueError('offline state identity mismatch')
    table=[]
    for row in outcomes:
        state_id=row['state_id'];cid=row['candidate_id'];run=row['run_id']
        index=int(state_id.rsplit('-',1)[1]);folder=ROOT/'pilot-inputs'/f'state-{index:03d}'
        map_path=folder/f'{cid}-map-path.json';base_path=folder/f'{cid}-base-path.json'
        trace=ROOT/'pilot/runs'/run/'live-0.trace.json.gz'
        summary=ROOT/'pilot/runs'/run/'live-0.json'
        marker=ROOT/'pilot/runs'/run/'live-0.marker.json'
        for key,path,expected in (('map Path',map_path,row['map_path_sha256']),
                                  ('trace',trace,row['source_sha256']['trace_sha256']),
                                  ('summary',summary,row['source_sha256']['summary_sha256']),
                                  ('marker',marker,row['source_sha256']['marker_sha256'])):
            if digest(path)!=expected:raise ValueError(f'{key} SHA mismatch: {run}')
        if len(read_json(base_path))!=9 or len(read_json(map_path))!=9:
            raise ValueError(f'Path point count mismatch: {run}')
        proposal=offline_by_state[state_id][cid]
        if row['proposal_ood']['constraint_valid']!=proposal['constraints']['valid'] or row['proposal_ood']['ood_over_train_p95']!=proposal['ood_over_train_p95']:
            raise ValueError(f'proposal validity mismatch: {run}')
        table.append({'state_id':state_id,'terrain':row['terrain_type'],'generator':row['kind'],
            'candidate_id':cid,'diffusion_seed':row['seed'],'run_id':run,
            'map_path_sha256':digest(map_path),'base_path_sha256':digest(base_path),
            'trace_sha256':digest(trace),'summary_sha256':digest(summary),'marker_sha256':digest(marker),
            'constraint_valid':proposal['constraints']['valid'],'ood_over_train_p95':proposal['ood_over_train_p95'],
            'delta_v':row['delta_v'],'advantage_h':row['advantage_h'],'drawdown':row['drawdown'],
            'recovery':row['recovery'],'progress_m':row['progress_m']})
    OUT.mkdir(parents=True,exist_ok=True)
    table_path=OUT/'candidate-identity-table.jsonl.gz'
    if table_path.exists():raise FileExistsError('identity table already exists; resume only after SHA audit')
    with table_path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
            for row in table:zipped.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode())
    provenance={'stage':'source_audit_complete','unix_sec':time.time(),
        'git_at_start':{'branch':'exp/eatnav-candidate-coverage','head':'43c05ad','porcelain':''},
        'git_at_audit':{'branch':subprocess.check_output(['git','branch','--show-current'],cwd=WORKTREE,text=True).strip(),
                        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip(),
                        'porcelain':subprocess.check_output(['git','status','--porcelain'],cwd=WORKTREE,text=True)},
        'config_sha256':digest(CONFIG),'selection_sha256':digest(selection_path),'schedule_sha256':digest(schedule_path),
        'pilot_outcome_sha256':digest(outcome_path),'pilot_summary_sha256':digest(summary_path),
        'pilot_postprocess_sha256':digest(post_path),'pilot_status_sha256':digest(ROOT/'pilot/status.json'),
        'offline_candidate_sha256':{state_id:digest(path) for state_id,path in offline_files.items()},
        'follower_command_sha256':{state_id:digest(ROOT/'stage1'/f'{state_id}-follower.json') for state_id in offline_files},
        'action_normalization_sha256':digest(WORKTREE/'docs/quality-identifiability-gate/phase-c-remediation/candidate-thresholds.json'),
        'trajectory_normalization_source_sha256':hashes['nomad_reference_best_ema']['sha256'],
        'train_constraint_limits_sha256':digest(ROOT/'stage1/train-constraint-limits.json'),
        'container':hashes,'candidate_identity_table_sha256':digest(table_path),
        'candidate_count':len(table),'k32_seed_count_per_state':len(expected_seeds),'elapsed_sec':time.time()-started,
        'new_rollouts_started':False,'retraining_started':False,'formal_validation_started':False}
    atomic_once(OUT/'source-provenance.json',provenance)
    atomic_once(OUT/'stage-0-complete.json',{'stage':0,'status':'complete','input_sha256':{
        'config':digest(CONFIG),'outcomes':digest(outcome_path),'selection':digest(selection_path),
        'schedule':digest(schedule_path)},'output_sha256':{'source_provenance':digest(OUT/'source-provenance.json'),
        'identity_table':digest(table_path)},'unix_sec':time.time()})
    print('FEASIBLE_TRANSFER_STAGE0_COMPLETE',len(table),'rows',round(time.time()-started,1),'sec',flush=True)

if __name__=='__main__':run()
