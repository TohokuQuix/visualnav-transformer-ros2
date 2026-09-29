"""Apply frozen Phase B GT Evaluator only after the full oracle pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.candidate_coverage_evaluator import ranking_diagnostics
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest
from eat_nav_phase_c_coverage_evaluator import FrozenEvaluator,SEEDS
from eat_nav_forward_evaluator import HEADS

ROOT=Path('/tmp/eatnav-candidate-coverage')


def run(output):
    status=json.loads((ROOT/'pilot/status.json').read_text())
    if status['state']!='raw_runs_complete' or status['completed_runs']!=119:
        raise ValueError('Evaluator ranking requires all 119 completed oracle runs')
    oracle_path=ROOT/'analysis/oracle-outcomes.json'
    summary_path=ROOT/'analysis/oracle-summary.json'
    oracle=json.loads(oracle_path.read_text())
    summary=json.loads(summary_path.read_text())
    if summary['complete_state_count']!=7 or oracle['candidate_count']!=91:
        raise ValueError('oracle candidate pool incomplete')
    selection_path=ROOT/'stage0/frozen/pilot-selection.json'
    selection=json.loads(selection_path.read_text())['selected']
    epsilon_path=ROOT/'pilot/oracle-epsilon.json';epsilon=json.loads(epsilon_path.read_text())['epsilon']
    manifest=Path('/tmp/eatnav-extended/output/manifest.json')
    reference_config=Path('/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
    cfg=load_eat_config(reference_config)
    view=CompleteOnlyView(EATDataset(manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(view)!=49598:raise ValueError('complete train count changed')
    evaluator=FrozenEvaluator('cuda')
    by_state={state['state_id']:[] for state in selection}
    for row in oracle['rows']:by_state[row['state_id']].append(row)
    states=[]
    with torch.inference_mode():
        for state in selection:
            w=view.dataset.windows[view.indices[state['view_index']]]
            dem=np.asarray(w['dem'],dtype=np.float32)
            if hashlib.sha256(dem.tobytes()).hexdigest()!=state['dem_sha256'] or list(w['episode_key'])!=state['episode_key']:
                raise ValueError('Evaluator input state changed')
            offline_path=ROOT/'stage1'/f"{state['state_id']}.json"
            offline=json.loads(offline_path.read_text())
            ids=[f'nomad-seed-{seed}' for seed in range(1001,1009)]
            ids += [f'bank-{name}' for name in ('nominal','slight_left','slight_right','strong_left','strong_right')]
            candidates={r['candidate_id']:r for r in offline['candidates']}
            paths=np.stack([np.asarray(candidates[cid]['points'],dtype=np.float32) for cid in ids])
            predictions=evaluator.predict(dem,np.asarray(w['goal'],dtype=np.float32),paths)
            if predictions.shape!=(3,13,4):raise ValueError('Evaluator prediction shape mismatch')
            outcomes={r['candidate_id']:r for r in by_state[state['state_id']]}
            if set(outcomes)!=set(ids):raise ValueError('oracle/evaluator candidate IDs differ')
            heads={}
            for head_index,head in enumerate(HEADS):
                values=[outcomes[cid][head] for cid in ids]
                measure=ranking_diagnostics(predictions[:,:,head_index:head_index+1],values,epsilon=epsilon[head])
                top=ids[measure['predicted_order'][0]] if 'predicted_order' in measure else None
                measure['predicted_top1_candidate_id']=top
                measure['top1_proposal_valid']=None if top is None else bool(candidates[top]['constraints']['valid']
                    and not candidates[top].get('ood_over_train_p95',False))
                heads[head]=measure
            states.append({'state_id':state['state_id'],'terrain_type':state['terrain_type'],
                'candidate_ids':ids,'prediction_by_seed_head':predictions.tolist(),
                'heads':heads,'offline_sha256':digest(offline_path)})
    aggregate={}
    for head in HEADS:
        rows=[state['heads'][head] for state in states]
        support=[r for r in rows if r['top1_recall'] is not None]
        aggregate[head]={'state_support_count':len(support),
            'top1_recall':sum(r['top1_recall'] for r in support)/len(support) if support else None,
            'top3_recall':sum(r['top3_recall'] for r in support)/len(support) if support else None,
            'pairwise_accuracy':sum(r['pairwise_accuracy']*r['pairwise_support'] for r in rows if r['pairwise_accuracy'] is not None)/sum(r['pairwise_support'] for r in rows) if sum(r['pairwise_support'] for r in rows) else None,
            'pairwise_support':sum(r['pairwise_support'] for r in rows),
            'calibration_rmse_mean':float(np.mean([r['calibration_rmse'] for r in support])) if support else None,
            'ensemble_spread_mean':float(np.mean([r['ensemble_spread_mean'] for r in rows])),
            'top1_invalid_proposal_count':sum(r['top1_proposal_valid'] is False for r in rows)}
    result={'status':'secondary_evaluator_after_complete_oracle','candidate_selection_used':False,
        'reference_no_nomad_retraining':True,'validation_used':False,'heldout_used':False,
        'evaluator_seeds':list(SEEDS),'evaluator_checkpoint_sha256':evaluator.checkpoint_sha256,
        'evaluator_scaler_sha256':evaluator.scaler_sha256,
        'source_sha256':{'selection':digest(selection_path),'oracle_outcomes':digest(oracle_path),
                         'oracle_summary':digest(summary_path),'epsilon':digest(epsilon_path),
                         'manifest':digest(manifest),'reference_config':digest(reference_config)},
        'heads':aggregate,'states':states}
    output.mkdir(parents=True,exist_ok=True)
    atomic_once(output/'evaluator-ranking.json',result)
    lines=['# Secondary Phase B GT Evaluator ranking', '',
        'The Evaluator was run after all 119 simulator oracle runs. It did not select or filter rollout candidates.',
        '', '| Head | top-1 recall | top-3 recall | pairwise accuracy | state support | invalid top-1 proposals |',
        '|---|---:|---:|---:|---:|---:|']
    fmt=lambda v:'NA' if v is None else f'{v:.3f}'
    for name,row in aggregate.items():
        lines.append(f"| {name} | {fmt(row['top1_recall'])} | {fmt(row['top3_recall'])} | {fmt(row['pairwise_accuracy'])} | {row['state_support_count']} | {row['top1_invalid_proposal_count']} |")
    report=output/'evaluator-report.md';text='\n'.join(lines)+'\n'
    if report.exists():
        if report.read_text()!=text:raise FileExistsError('Evaluator report differs')
    else:
        temp=report.with_name(report.name+'.tmp');temp.write_text(text);temp.replace(report)
    print('EVALUATOR_SECONDARY_COMPLETE',aggregate['delta_v'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'analysis')
    run(p.parse_args().output)

if __name__=='__main__':main()
