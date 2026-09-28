"""One held-out Gate A evaluation after aggregate-validation model selection."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.quality_gate_config import load_gate_settings, decide_gate_a
from visualnav_transformer.eat_nav.quality_gate_probe import tensorize_windows, predict_probe
from visualnav_transformer.eat_nav.quality_gate_evaluation import episode_cluster_ci, pair_cluster_ci, terrain_support, trajectory_errors
from visualnav_transformer.eat_nav.quality_gate_data import trajectory_shape
from visualnav_transformer.eat_nav.quality_followup import is_complete_window


def pairs_and_ci(control,windows,settings):
    result={}
    for label,key in (('cross_q','cross_q_pairs'),('same_q','same_q_pairs')):
        rows=control[key]
        values={m:[] for m in ('trajectory_rms_m','endpoint_m','forward_progress_m','path_length_m','total_turning_rad')}
        episodes=[]
        for row in rows:
            a=windows[row['low_index']];b=windows[row['high_index']]
            xa,fa,la,ta=trajectory_shape(a['action'])
            xb,fb,lb,tb=trajectory_shape(b['action'])
            values['trajectory_rms_m'].append(float(np.sqrt(np.square(xa-xb).sum(axis=1).mean())))
            values['endpoint_m'].append(float(np.linalg.norm(xa[-1]-xb[-1])))
            values['forward_progress_m'].append(abs(fa-fb))
            values['path_length_m'].append(abs(la-lb))
            values['total_turning_rad'].append(abs(ta-tb))
            episodes.append((tuple(a['episode_key']),tuple(b['episode_key'])))
        result[label]={'count':len(rows),'means':{k:float(np.mean(v)) for k,v in values.items()},
                       'medians':{k:float(np.median(v)) for k,v in values.items()},
                       'values':values,'episodes':episodes}
    comparison={name:pair_cluster_ci(result['cross_q']['values'][name],result['cross_q']['episodes'],
                                      result['same_q']['values'][name],result['same_q']['episodes'],
                                      seed=settings.bootstrap_seed,repeats=settings.bootstrap_replicates)
                for name in result['cross_q']['values']}
    for value in result.values():
        value.pop('values');value.pop('episodes')
    return {'cross_q':result['cross_q'],'same_q':result['same_q'],'cross_minus_same':comparison}


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','quality-config','gate-config','runs-summary','pair-controls','audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--complete-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    config=load_eat_config(a.quality_config);settings=load_gate_settings(a.gate_config)
    runs=json.loads(a.runs_summary.read_text())['runs']
    audit=json.loads(a.audit.read_text())
    controls=json.loads(a.pair_controls.read_text())
    train=EATDataset(a.manifest,'train',config)
    val=EATDataset(a.manifest,'val',config)
    test=EATDataset(a.manifest,'test',config)
    if train.split_map!=val.split_map or train.split_map!=test.split_map:
        raise ValueError('split mismatch')
    if train.ecdf.values!=val.ecdf.values or train.ecdf.values!=test.ecdf.values:
        raise ValueError('ECDF mismatch')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def rows(dataset):
        result=[dict(dataset[i],window_index=i) for i in range(len(dataset))]
        return [row for row in result if is_complete_window(row,config.len_traj_pred)] if a.complete_only else result
    train_rows=rows(train)
    val_rows=rows(val)
    test_rows=rows(test)
    if a.complete_only and (len(train_rows),len(val_rows),len(test_rows))!=(49598,6171,6208):
        raise ValueError('complete-only evaluation window count mismatch')
    train_data=tensorize_windows(train_rows,train.stats,device)
    val_data=tensorize_windows(val_rows,train.stats,device)
    test_data=tensorize_windows(test_rows,train.stats,device)
    mean=(train_data['action']*train_data['mask'].unsqueeze(-1)).sum(dim=0)/train_data['mask'].sum(dim=0).clamp_min(1).unsqueeze(-1)
    def mean_error(data):
        y=mean.unsqueeze(0).expand(len(data['q']),-1,-1)
        return float((((y-data['action']).square())*data['mask'].unsqueeze(-1)).sum()/(2*data['mask'].sum()))
    mean_val=mean_error(val_data);mean_test=mean_error(test_data)
    baseline_target=test_data['action']
    baseline_mask=test_data['mask']
    mean_test_window=float(((((mean.unsqueeze(0)-baseline_target).square())*baseline_mask.unsqueeze(-1)).sum(dim=(1,2))/(2*baseline_mask.sum(dim=1).clamp_min(1))).mean())
    target=test_data['action'].cpu().numpy()
    mask=test_data['mask'].cpu().numpy()
    physical_target=target*train.stats.std+train.stats.mean
    run_results=[];by_seed={}
    for run in runs:
        kind=run['kind'];seed=run['training_seed']
        qkind='probe_0' if kind=='probe_0' else 'probe_q'
        pred=predict_probe(run['checkpoint'],test_data,qkind,settings.probe_batch_size,
                           settings.fixed_probe_q).cpu().numpy()
        norm_mse=(np.square(pred-target)*mask[:,:,None]).sum(axis=(1,2))/(2*np.maximum(mask.sum(axis=1),1))
        physical_pred=pred*train.stats.std+train.stats.mean
        metrics=trajectory_errors(physical_pred,physical_target,mask)
        entry={'kind':kind,'training_seed':seed,'shuffle_seed':run['shuffle_seed'],
               'heldout_mse':float(np.sum(np.square(pred-target)*mask[:,:,None])/(2*mask.sum())),
               'heldout_window_mean_mse':float(np.mean(norm_mse)),
               'heldout_rms_m':float(np.mean(metrics['rms_m'])),
               'endpoint_error_m':float(np.mean(metrics['endpoint_m'])),
               'forward_progress_error_m':float(np.mean(metrics['forward_progress_error_m'])),
               'path_length_error_m':float(np.mean(metrics['path_length_error_m'])),
               'turning_error_rad':float(np.mean(metrics['total_turning_error_rad'])),
               'per_point_error_m':[float(x) for x in (metrics['point_error_m']*mask).sum(axis=0)/np.maximum(mask.sum(axis=0),1)],
               'best_validation_loss':run['best_validation_loss'],'best_epoch':run['best_epoch'],
               'validation_improving_at_cap':run['validation_improving_at_cap']}
        run_results.append(entry)
        if kind in ('probe_0','probe_q'):
            by_seed.setdefault(seed,{})[kind]={'mse':norm_mse,'rms':metrics['rms_m'],
                                                 'pred':physical_pred,'summary':entry}
        print('HELDOUT_PROBE',kind,seed,run['shuffle_seed'],round(entry['heldout_mse'],6),flush=True)
    paired=np.stack([by_seed[seed]['probe_0']['mse']-by_seed[seed]['probe_q']['mse']
                     for seed in settings.training_seeds]).mean(axis=0)
    ci,_=episode_cluster_ci(paired,test_data['episode'],seed=settings.bootstrap_seed,
                            repeats=settings.bootstrap_replicates)
    p0_mse=np.mean([by_seed[s]['probe_0']['summary']['heldout_window_mean_mse'] for s in settings.training_seeds])
    pq_mse=np.mean([by_seed[s]['probe_q']['summary']['heldout_window_mean_mse'] for s in settings.training_seeds])
    p0_rms=np.mean([by_seed[s]['probe_0']['summary']['heldout_rms_m'] for s in settings.training_seeds])
    pq_rms=np.mean([by_seed[s]['probe_q']['summary']['heldout_rms_m'] for s in settings.training_seeds])
    shuffle_effect=[]
    for entry in run_results:
        if entry['kind']=='probe_q_shuffled':
            shuffle_effect.append(float(by_seed[entry['training_seed']]['probe_0']['summary']['heldout_window_mean_mse']-entry['heldout_window_mean_mse']))
    terrain=np.asarray(test_data['terrain'])
    by_terrain={}
    for typ in sorted(set(terrain)):
        indices=np.flatnonzero(terrain==typ)
        by_terrain[typ]={'windows':len(indices),'mse_improvement':float(paired[indices].mean()),
                         'status':'supported'}
    support_input={split:{terrain:{int(k):v for k,v in spec['bins'].items()}
                          for terrain,spec in value['q_by_terrain'].items()}
                   for split,value in audit['splits'].items()}
    support=terrain_support(support_input,settings.min_terrain_q_bin_count)
    for typ,entry in by_terrain.items():
        if support['by_split']['test'][typ]['status']!='supported':
            entry['status']='insufficient support';entry['mse_improvement']=None
    supported=[t for t,v in by_terrain.items() if v['status']=='supported']
    robust=(sum(by_terrain[t]['mse_improvement']>0 for t in supported)>=2
            and all(float(paired[terrain!=t].mean())>0 for t in supported))
    support['q_supported_all_splits']=[t for t in support['common_terrain_types']
            if all(support['by_split'][split][t]['status']=='supported' for split in ('train','val','test'))]
    pair_directions={name:{'actual':[],'predicted':[]} for name in ('forward_progress','path_length','total_turning')}
    index_to_pos={int(i):pos for pos,i in enumerate(test_data['index'])}
    for row in controls['cross_q_pairs']:
        lo=row['low_index'];hi=row['high_index']
        if lo not in index_to_pos or hi not in index_to_pos:continue
        la=test[lo]['action'];ha=test[hi]['action']
        _,lf,ll,lt=trajectory_shape(la);_,hf,hl,ht=trajectory_shape(ha)
        for name,lo_value,hi_value in (('forward_progress',lf,hf),('path_length',ll,hl),('total_turning',lt,ht)):
            pair_directions[name]['actual'].append(hi_value-lo_value)
        lp=index_to_pos[lo];hp=index_to_pos[hi]
        pred_deltas=np.mean([by_seed[s]['probe_q']['pred'][[lp,hp]] for s in settings.training_seeds],axis=0)
        _,lf,ll,lt=trajectory_shape(pred_deltas[0]);_,hf,hl,ht=trajectory_shape(pred_deltas[1])
        for name,lo_value,hi_value in (('forward_progress',lf,hf),('path_length',ll,hl),('total_turning',lt,ht)):
            pair_directions[name]['predicted'].append(hi_value-lo_value)
    directions={name:{'actual_median':float(np.median(v['actual'])),
                      'predicted_median':float(np.median(v['predicted'])),
                      'sign_agreement':bool(np.sign(np.median(v['actual']))==np.sign(np.median(v['predicted'])))}
                for name,v in pair_directions.items()}
    near=(p0_mse-pq_mse>0 and ((p0_rms-pq_rms)/p0_rms>=settings.min_relative_rms_gain-settings.near_relative_rms_margin
           or (p0_mse-pq_mse)/mean_test_window>=settings.min_delta_r2-settings.near_delta_r2_margin))
    decision_metrics={'improvement':float(p0_mse-pq_mse),'ci_lower':float(ci[0]),
                      'shuffled':shuffle_effect,'relative_rms_improvement':float((p0_rms-pq_rms)/p0_rms),
                      'delta_r2':float((p0_mse-pq_mse)/mean_test_window),
                      'terrain_robust':bool(robust),
                      'direction_agreement':any(v['sign_agreement'] for v in directions.values()),
                      'probe_0_mse':float(np.mean([by_seed[s]['probe_0']['summary']['best_validation_loss'] for s in settings.training_seeds])),
                      'probe_q_mse':float(np.mean([by_seed[s]['probe_q']['summary']['best_validation_loss'] for s in settings.training_seeds])),
                      'mean_mse':mean_val,
                      'validation_improving_at_cap':any(r['validation_improving_at_cap'] for r in run_results if r['kind'] in ('probe_0','probe_q')),
                      'near_boundary':bool(near)}
    status=decide_gate_a(decision_metrics,settings)
    result={'status':status,'decision_metrics':decision_metrics,'bootstrap_ci_95':ci,
            'permutation_p_value':(1+sum(x>=decision_metrics['improvement'] for x in shuffle_effect))/(1+len(shuffle_effect)),
            'mean_trajectory_validation_mse':mean_val,'mean_trajectory_test_mse':mean_test,'mean_trajectory_test_window_mse':mean_test_window,
            'heldout_window_count':len(test_data['q']),'run_results':run_results,
            'by_terrain':by_terrain,'support':support,'data_pair_comparison':pairs_and_ci(controls,test.windows,settings),
            'direction_comparison':directions,'heldout_only_for_final_evaluation':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('GATE_A_DONE',json.dumps({'status':status,'improvement':decision_metrics['improvement'],
                                    'ci':ci,'relative_rms':decision_metrics['relative_rms_improvement'],
                                    'shuffle_count':len(shuffle_effect)}),flush=True)

if __name__=='__main__':main()
