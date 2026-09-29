"""Offline frozen Phase B score spread for a coverage-passing proposal set."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import yaml
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from eat_nav_phase_c_coverage import atomic,digest,write_marker
from eat_nav_phase_c_coverage_evaluator import FrozenEvaluator,score_spread
from eat_nav_phase_c_coverage_structured import proposals


def primary_standardized(pred,scalers):
    heads=('delta_v','advantage_h','drawdown')
    p=np.asarray(pred,dtype=float)
    return np.mean(np.stack([(p[...,i]-scalers[h]['mean'])/scalers[h]['std'] for i,h in enumerate(heads)]),axis=0)


def main():
    parser=argparse.ArgumentParser()
    for name in ('manifest','reference-config','coverage-config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.output
    sampling=json.loads((root/'sampling-coverage-decision.json').read_text())
    structured=json.loads((root/'structured-coverage-decision.json').read_text())
    choice=('sampling',sampling['selected']) if sampling['decision']=='pass' else ('structured','global_train_pca') if structured['decision']=='pass' else None
    if choice is None:
        atomic(root/'evaluator-score-spread.json',{'status':'not_run','reason':'offline_coverage_failed'})
        return
    spec=yaml.safe_load(args.coverage_config.read_text())
    ids=json.loads((root/'diagnostic-state-ids.json').read_text())
    states=[('train',x) for x in ids['train']]+[('val',x) for x in ids['val']]
    cfg=load_eat_config(args.reference_config)
    views={s:CompleteOnlyView(EATDataset(args.manifest,s,cfg,split_seed=123),cfg.len_traj_pred) for s in ('train','val')}
    evaluator=FrozenEvaluator('cuda')
    basis=scale=None
    if choice[0]=='structured':
        with np.load(root/'residual-basis.npz') as data:
            if str(data['split'])!='train' or int(data['train_count'])!=49598:raise ValueError('nontrain basis')
            basis=data['basis'];scale=data['scale']
    rows=[]
    for index,(split,state) in enumerate(states):
        window=views[split].dataset.windows[views[split].indices[state['view_index']]]
        with np.load(root/'trace-states'/f'{index:03d}.npz') as data:raw=data['paths']
        paths=raw[:8] if choice==('sampling','baseline_k8') else raw if choice==('sampling','baseline_k32') else proposals(raw[:8],basis,scale)
        pred=evaluator.predict(window['dem'],window['goal'],paths)
        score=primary_standardized(pred,evaluator.scalers)
        mean=score.mean(axis=0)
        uncertainty=score.std(axis=0)
        spread=score_spread(pred)
        candidate=json.loads((root/'sampling-states'/f'{index:03d}.json').read_text())['conditions'][choice[1]] if choice[0]=='sampling' else json.loads((root/'structured-states'/f'{index:03d}.json').read_text())
        commands=np.asarray(candidate['follower_commands'])
        relative=np.abs(commands[:,None,2]-commands[None,:,2])>=spec['clusters']['command_angular_radps']
        rank=spearmanr(np.arange(len(mean)),mean).statistic if np.std(mean)>0 else None
        rows.append({'state_index':index,'split':split,'terrain_type':state['terrain_type'],
           'score':mean.tolist(),'score_uncertainty':uncertainty.tolist(),
           'primary_composite_spread':float(np.ptp(mean)),
           'candidate_pairs_over_ensemble_uncertainty_fraction':float(np.mean(np.abs(mean[:,None]-mean[None,:])>uncertainty[:,None]+uncertainty[None,:])),
           'angular_distinct_pairs':int(relative[np.triu_indices(len(mean),1)].sum()),
           'score_index_rank_diagnostic_only':None if rank is None else float(rank),
           'head_spread':spread})
    atomic(root/'evaluator-score-spread.json',{'status':'complete','selected_proposal':choice,
        'evaluator_checkpoint_sha256':evaluator.checkpoint_sha256,'scaler_sha256':evaluator.scaler_sha256,
        'primary_composite':'mean_train_standardized_delta_v_advantage_h_drawdown',
        'all_same_score_fraction':sum(r['head_spread']['all_same_score'] for r in rows)/len(rows),
        'rows':rows,'used_for_proposal_selection':False})
    write_marker(root/'evaluator.done.json',[root/'evaluator-score-spread.json'],digest(args.coverage_config))

if __name__=='__main__':main()
