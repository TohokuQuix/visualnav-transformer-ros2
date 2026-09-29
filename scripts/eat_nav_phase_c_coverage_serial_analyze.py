"""Descriptive eight-state serial smoke analysis; never a formal gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from eat_nav_phase_c_coverage import atomic,digest,write_marker
from eat_nav_phase_c_coverage_evaluator import FrozenEvaluator
from eat_nav_phase_c_coverage_score import primary_standardized

GAMMA=.99

def outcome(data,branch=0):
    value=np.asarray(data['critic_value'],dtype=float)[:,branch].reshape(-1)
    reward=np.asarray(data['reward'],dtype=float)[:,branch].reshape(-1)
    end=float(np.asarray(data['critic_value_end'],dtype=float)[branch].reshape(-1)[0])
    initial=float(value[0]);minimum=float(min(np.min(value),end));n=len(reward)
    discounted=float(np.dot(np.power(GAMMA,np.arange(n)),reward))
    return {'delta_v':end-initial,'advantage_h':discounted+GAMMA**n*end-initial,
            'drawdown':minimum-initial,'recovery':end-minimum,'v0':initial,'v_end':end}


def analysis(args):
    root=args.output
    decision=json.loads((root/'sampling-coverage-decision.json').read_text())
    structured=json.loads((root/'structured-coverage-decision.json').read_text())
    if decision['decision']!='pass' and structured['decision']!='pass':
        atomic(root/'serial-smoke-results.json',{'status':'not_run','reason':'offline_coverage_failed'})
        return
    evaluator=FrozenEvaluator('cuda')
    frozen=json.loads((root/'diagnostic-state-ids.json').read_text())
    rows=[]
    for state_i,state in enumerate(frozen['train'][:8]):
        folder=root/'serial'/f'state-{state_i:02d}'
        baseline=json.loads((folder/'serial_duplicate-0.json').read_text())
        anchor=baseline['serial_anchor']
        if anchor['history_frame_count']!=4:raise ValueError('not natural history')
        paths=np.asarray(baseline['candidate_base_paths'],dtype=float)
        pred=evaluator.predict(anchor['dem_history'],anchor['relative_goal'],paths)
        predicted=primary_standardized(pred,evaluator.scalers).mean(axis=0)
        duplicates=np.load(folder/'serial_duplicate-0.npz')
        dup_xy=np.asarray(duplicates['xy'],dtype=float)
        dup_rms=float(np.sqrt(np.mean((dup_xy-dup_xy[:,:1])**2)))
        true=[];xy=[];commands=[];hashes=[]
        for candidate in range(8):
            summary=json.loads((folder/f'serial_candidate-{candidate}.json').read_text())
            if summary['serial_anchor']['dem_history_sha256']!=anchor['dem_history_sha256']:
                raise ValueError('serial history mismatch')
            if not np.allclose(summary['serial_anchor']['actor_input'],anchor['actor_input'],atol=1e-5):
                raise ValueError('serial initial Actor observation mismatch')
            trace_path=folder/f'serial_candidate-{candidate}.npz'
            with np.load(trace_path) as data:
                true.append(outcome(data));xy.append(np.asarray(data['xy'])[:,0]);commands.append(np.asarray(data['command'])[:,0])
            hashes.append(digest(trace_path))
        xy=np.asarray(xy);commands=np.asarray(commands)
        path_effect=float(np.sqrt(np.mean((xy-xy[:1])**2)))
        rho=dup_rms/path_effect if path_effect>0 else None
        heads=('delta_v','advantage_h','drawdown')
        true_score=np.mean(np.column_stack([np.asarray([(t[h]-evaluator.scalers[h]['mean'])/evaluator.scalers[h]['std'] for t in true]) for h in heads]),axis=1)
        oracle=int(np.argmax(true_score));selected=int(np.argmax(predicted));canonical=0
        denominator=float(true_score[oracle]-true_score[canonical])
        recovery=float((true_score[selected]-true_score[canonical])/denominator) if denominator>1e-8 else None
        rank=spearmanr(true_score,predicted).statistic if np.std(true_score)>0 and np.std(predicted)>0 else None
        rows.append({'state_index':state_i,'terrain_type':state['terrain_type'],'terrain_seed':state['terrain_seed'],
             'duplicate_xy_rms_m':dup_rms,'candidate_xy_effect_rms_m':path_effect,'rho':rho,
             'rho_below_0_25':bool(rho is not None and rho<.25),
             'initial_command_spread_linear_mps':float(np.ptp(commands[:,0,0])),
             'initial_command_spread_angular_radps':float(np.ptp(commands[:,0,2])),
             'true_outcomes':true,'predicted_outcomes':pred.mean(axis=0).tolist(),
             'predicted_score':predicted.tolist(),'true_score':true_score.tolist(),
             'oracle_index':oracle,'predicted_index':selected,'canonical_index':canonical,
             'oracle_recovery_ratio_descriptive':recovery,
             'spearman_descriptive':None if rank is None else float(rank),
             'candidate_trace_sha256':hashes,'duplicate_trace_sha256':digest(folder/'serial_duplicate-0.npz')})
    atomic(root/'serial-smoke-results.json',{'status':'complete','state_count':len(rows),
        'rho_below_0_25_fraction':sum(r['rho_below_0_25'] for r in rows)/len(rows),
        'interpretation':'small-state descriptive smoke only, no formal pass/fail or terrain generalization',
        'rows':rows})
    write_marker(root/'serial.done.json',[root/'serial-smoke-results.json'],digest(args.coverage_config))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coverage-config',type=Path,required=True)
    analysis(p.parse_args())
