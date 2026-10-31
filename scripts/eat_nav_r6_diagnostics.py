"""Validation-only DEM/state dependence diagnostics; no physical counterfactual claim."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.r6_model import load_checkpoint,predict
from visualnav_transformer.eat_nav.r6_metrics import event_metrics,trajectory_metrics
from eat_nav_r6_run import read,write,sha


def run(dataset,checkpoint,output):
 output=Path(output)
 if output.exists():raise FileExistsError('fresh diagnostic output required')
 with np.load(Path(dataset)/'samples.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
 selected=read(Path(dataset)/'split.json')['roles']['validation'];model,payload=load_checkpoint(checkpoint)
 n=len(selected);rng=np.random.default_rng(20261031);permutation=rng.permutation(n)
 def inference(a):
  pieces=[predict(model,payload['normalization'],a,selected[i:i+32]) for i in range(0,n,32)]
  return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
 baseline=inference(arrays);results={}
 members=read(Path(dataset)/'sample-index.json');cells={}
 for local,original in enumerate(selected):
  m=members[original];cells.setdefault(str((m['field'],m.get('level'),m.get('scene','field'))),[]).append(local)
 per_cell={}
 for key,positions in cells.items():
  originals=np.asarray(selected)[positions]
  per_cell[key]=dict(groups=len(positions),events=event_metrics(baseline['events'][positions],arrays['event_targets'][originals],arrays['event_mask'][originals]),trajectory=trajectory_metrics(baseline['trajectory'][positions],arrays['trajectory'][originals],arrays['trajectory_mask'][originals]))
 for name,keys in [('fixed_DEM_permuted_state',('actor_state','body_state')),('fixed_state_permuted_DEM',('dem',))]:
  changed=dict(arrays)
  for k in keys:
   changed[k]=arrays[k].copy();changed[k][selected]=arrays[k][np.asarray(selected)[permutation]]
  predicted=inference(changed)
  results[name]=dict(mean_abs_event_probability_change=float(np.abs(predicted['events']-baseline['events']).mean()),mean_future_position_change_m=float(np.linalg.norm(predicted['trajectory'][...,:3]-baseline['trajectory'][...,:3],axis=-1).mean()),
   events=event_metrics(predicted['events'],arrays['event_targets'][selected],arrays['event_mask'][selected]),trajectory=trajectory_metrics(predicted['trajectory'],arrays['trajectory'][selected],arrays['trajectory_mask'][selected]),interpretation='network-input sensitivity using observed validation values; state/DEM combinations may be physically out of distribution, not simulator outcomes')
 # Native deployment batch=1 timings, including normalization and host transfer.
 lat=[]
 for _ in range(3):predict(model,payload['normalization'],arrays,[selected[0]])
 for j in range(min(100,n)):
  start=time.perf_counter();predict(model,payload['normalization'],arrays,[selected[j]]);lat.append(time.perf_counter()-start)
 result=dict(schema='eatnav_r6_validation_diagnostics_v1',source='validation_only',checkpoint_sha256=sha(checkpoint),sample_count=n,field_level_scene_metrics=per_cell,permutation_seed=20261031,diagnostics=results,cpu_inference_batch1_seconds=dict(count=len(lat),mean=float(np.mean(lat)),p50=float(np.quantile(lat,.5)),p95=float(np.quantile(lat,.95)),maximum=max(lat)),baseline=dict(events=event_metrics(baseline['events'],arrays['event_targets'][selected],arrays['event_mask'][selected]),trajectory=trajectory_metrics(baseline['trajectory'],arrays['trajectory'][selected],arrays['trajectory_mask'][selected])))
 output.mkdir(parents=True);write(output/'diagnostics.json',result);return result


def main():
 torch.set_num_threads(1)
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('dataset','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();print(json.dumps(run(a.dataset,a.checkpoint,a.output),indent=2))
if __name__=='__main__':main()
