"""Score R6 forecasts only against the candidate actually executed in each2swindow.
Stop/goal-hold overrides censor candidate supervision. Equivalent reused outputs
are not counted again. Test outputs are diagnostic only and never recalibrate.
"""
import argparse,collections,json
from pathlib import Path
import numpy as np
import torch
from eat_nav_r6_run import read,write,sha
from eat_nav_r6_storage import read_trace
from visualnav_transformer.eat_nav.r6_model import load_checkpoint,predict
from visualnav_transformer.eat_nav.r6_targets import future_targets
from visualnav_transformer.eat_nav.r6_metrics import event_metrics,trajectory_metrics


def evaluate(queue_root,checkpoint,output):
 root=Path(queue_root);output=Path(output)
 if output.exists():raise FileExistsError('fresh rollout metric output required')
 summary=read(root/'summary.json');binding=read(root/'frozen-binding.json');ledger=read(root/'progress.json')
 if not summary['all_jobs_completed']:raise ValueError('completed queue required; no partial result masquerading as final')
 if sha(checkpoint)!=binding['model_sha256']:raise ValueError('evaluation model mismatch')
 model,payload=load_checkpoint(checkpoint);observations=[];ep=[];et=[];em=[];tp=[];tt=[];tm=[];skipped=collections.Counter();groups=collections.defaultdict(list)
 for job_id,record in ledger.items():
  if record.get('state')!='completed' or record.get('execution')!='simulated':continue
  job=read(root/job_id/'assignment.json');actual=root/job_id/'sim-output'
  for folder in sorted(actual.glob('group-*')):
   group=read(folder/'group.json')
   if not group['initial_valid']:skipped['initialization_invalid']+=1;continue
   rows=read_trace(folder/'trace.jsonl')
   for decision_file in sorted(folder.glob('decision-*.json')):
    decision=read(decision_file);slot=decision['selected_index'];step=decision['decision_step']
    if slot is None:skipped['STOP_has_no_executed_bank_candidate']+=1;continue
    window=[]
    for offset,row in enumerate(rows[step:step+40]):
     copy=dict(row,source_control_step=row['control_step'],control_step=offset,candidate_cycle=0,candidate_replan=False)
     if row.get('goal_hold') or row.get('protective_stop'):copy['manual_stop_before_step']=True
     window.append(copy)
    if not window:continue
    target=future_targets(window)
    with np.load(folder/f'input-{step:04d}.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    if int(arrays['decision_stamp_ns'])!=window[0]['logical_sim_time_ns'] or np.any(arrays['dem_stamps_ns']>arrays['decision_stamp_ns']):raise ValueError('decision input/actual window timing mismatch')
    prediction=predict(model,payload['normalization'],arrays,[0]);index=len(observations)
    ep.append(prediction['events'][0,slot]);et.append(target['event_targets']);em.append(target['event_mask']);tp.append(prediction['trajectory'][0,slot]);tt.append(target['trajectory']);tm.append(target['trajectory_mask'])
    observations.append(dict(group_id=group['group_id'],source_actual=str(folder),condition=job['condition']['id'],field=job['field'],level=job['level'],decision_step=step,candidate=slot,known_event_heads=int(target['event_mask'].sum()),valid_future_points=int(target['trajectory_mask'].sum()),scope='one actually executed candidate, not all14 counterfactual branches'))
    groups[(job['condition']['id'],job['field'],job['level'])].append(index)
 if not observations:raise ValueError('no actual nonSTOP candidate outcomes available')
 ep,et,em,tp,tt,tm=map(np.asarray,(ep,et,em,tp,tt,tm))
 def metrics(ids):return dict(windows=len(ids),events=event_metrics(ep[ids],et[ids],em[ids]),trajectory=trajectory_metrics(tp[ids],tt[ids],tm[ids]))
 per_cell={str(key):metrics(ids) for key,ids in groups.items()};per_condition={}
 for name in {r['condition'] for r in observations}:per_condition[name]=metrics([i for i,r in enumerate(observations) if r['condition']==name])
 result=dict(schema='eatnav_r6_actual_candidate_forecast_metrics_v1',split=binding['role'],model_sha256=sha(checkpoint),scope='unique actual simulated outputs only; policy-equivalent reuse not duplicated',selected_windows=len(observations),skipped=dict(skipped),overall=metrics(list(range(len(observations)))),by_condition=per_condition,by_condition_field_level=per_cell,
  limitations=['Selected-action support is policy-dependent; no unexecutedcandidate labels are invented','Goal hold and protective-stop overrides censor candidate outcomes','Multiple windows/candidate conditions share sourcecohorts; pooled samples are correlated','Test metrics are never used for fitting thresholds or model selection'])
 output.mkdir(parents=True);write(output/'report.json',result);write(output/'selected-window-index.json',observations)
 np.savez_compressed(output/'selected-window-predictions.npz',event_probabilities=ep,event_targets=et,event_mask=em,trajectory_predictions=tp,trajectory_targets=tt,trajectory_mask=tm)
 return result


def main():
 torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__)
 for key in ('queue-root','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();r=evaluate(a.queue_root,a.checkpoint,a.output);print(json.dumps(dict(split=r['split'],selected_windows=r['selected_windows'],skipped=r['skipped']),indent=2))
if __name__=='__main__':main()
