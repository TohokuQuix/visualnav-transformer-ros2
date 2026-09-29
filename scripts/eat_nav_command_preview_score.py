"""Frozen three-seed Phase B GT scoring of deployment-time follower previews."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np,torch
from scipy.stats import spearmanr
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from eat_nav_phase_c_coverage_evaluator import FrozenEvaluator
ROOT=Path('/tmp/eatnav-candidate-coverage');OUT=ROOT/'command-preview-long-horizon';PRIOR=ROOT/'feasible-transfer';BASE=ROOT/'baseline-worktree-8e8f4fd'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_once(p,obj):
 data=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n';p=Path(p)
 if p.exists():
  if p.read_text()!=data:raise ValueError(f'output differs: {p}')
 else:
  temp=p.with_suffix('.tmp');temp.write_text(data);temp.replace(p)

def metrics(y,p,ids,eps=.01):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);order=np.argsort(-p,kind='stable');best=np.max(y)
 pairs=[(i,j) for i,j in combinations(range(len(y)),2) if abs(y[i]-y[j])>eps]
 correct=sum((p[i]-p[j])*(y[i]-y[j])>0 for i,j in pairs)
 rho=spearmanr(y,p).statistic if len(y)>2 and np.ptp(y)>1e-8 and np.ptp(p)>1e-8 else None
 return {'top1_candidate_id':ids[int(order[0])],'top1_recall':bool(y[order[0]]>=best-eps),
         'top3_recall':bool(any(y[i]>=best-eps for i in order[:3])),
         'pairwise_accuracy':correct/len(pairs) if pairs else None,'supported_pairs':len(pairs),
         'oracle_regret':float(best-y[order[0]]),'spearman':float(rho) if rho is not None and np.isfinite(rho) else None,
         'predicted_score_spread':float(np.ptp(p)),'true_delta_v_spread':float(np.ptp(y))}

def main():
 start=time.monotonic();marker=read(OUT/'stage-a-preview-complete.json');assert marker['candidate_count']==91
 provenance=read(OUT/'source-provenance.json');old=read(PRIOR/'evaluator-counterfactual-results.json')
 view=CompleteOnlyView(EATDataset(Path('/tmp/eatnav-extended/output/manifest.json'),'train',load_eat_config(BASE/'docs/quality-identifiability-gate/phase-c/reference-eat.yaml'),split_seed=123),8)
 assert len(view)==49598
 frozen=FrozenEvaluator('cuda' if torch.cuda.is_available() else 'cpu')
 assert frozen.checkpoint_sha256=={str(s):provenance['prior_checkpoint_sha256'][f'evaluator_GT_{s}']['sha256'] for s in (11,29,47)}
 assert frozen.scaler_sha256==provenance['prior_checkpoint_sha256']['evaluator_train_scalers']['sha256']
 selected=read(ROOT/'stage0/frozen/pilot-selection.json')['selected']
 detail=read(PRIOR/'feasibility-by-candidate.json')['rows'];by=defaultdict(list)
 for r in detail:by[r['state_id']].append(r)
 states=[];candidate=[]
 for state in selected:
  sid=state['state_id'];rows=by[sid];assert len(rows)==13
  w=view.dataset.windows[view.indices[state['view_index']]];dem=np.asarray(w['dem'],dtype=np.float32);goal=np.asarray(w['goal'],dtype=np.float32)
  assert hashlib.sha256(dem.tobytes()).hexdigest()==state['dem_sha256']
  ids=[r['candidate_id'] for r in rows];paths=[]
  for cid in ids:
   p=OUT/'preview-candidates'/f'{sid}--{cid}.json';m=p.with_suffix('.marker.json');assert sha(p)==read(m)['output_sha256']
   x=read(p);assert x['candidate_id']==cid and x['state_id']==sid and x['preview_inputs_only']
   paths.append(x['closed_trajectory'])
  paths=np.asarray(paths,dtype=np.float32);assert paths.shape==(13,8,2)
  predictions=frozen.predict(dem,goal,paths);mean=predictions[:,:,0].mean(0)
  y=[r['delta_v'] for r in rows]
  group={'state_id':sid,'terrain':state['terrain_type'],'candidate_ids':ids,'generators':[r['generator'] for r in rows],
         'seeds':[r['seed'] for r in rows],'true_delta_v':y,'gt_c_hat_by_seed_candidate_head':predictions.tolist(),
         'gt_c_hat':metrics(y,mean,ids),'old_gt':next(s['gt'] for s in old['states'] if s['state_id']==sid),
         'ensemble_disagreement':float(np.std(predictions[:,:,0],axis=0).mean()),
         'generator_bias':{g:float(np.mean([mean[i]-y[i] for i,r in enumerate(rows) if r['generator']==g])) for g in ('nomad','bank')}}
  strata={}
  for name,indices in {'constraint_valid':[i for i,r in enumerate(rows) if r['condition']['constraint_valid']],
                       'nomad_only':[i for i,r in enumerate(rows) if r['generator']=='nomad'],
                       'bank_only':[i for i,r in enumerate(rows) if r['generator']=='bank']}.items():
   strata[name]=metrics([y[i] for i in indices],[mean[i] for i in indices],[ids[i] for i in indices]) if len(indices)>=2 else None
  group['strata']=strata;states.append(group)
  for i,r in enumerate(rows):candidate.append({'state_id':sid,'terrain':state['terrain_type'],'candidate_id':ids[i],
    'generator':r['generator'],'seed':r['seed'],'delta_v':y[i],'gt_c_hat_by_seed':predictions[:,i,0].tolist(),
    'gt_c_hat_mean':float(mean[i]),'constraint_valid':r['condition']['constraint_valid']})
 vals=[s['gt_c_hat'] for s in states]
 aggregate={'top1_recall_count':sum(s['top1_recall'] for s in vals),'top3_recall_count':sum(s['top3_recall'] for s in vals),
  'state_macro_pairwise_accuracy':float(np.mean([s['pairwise_accuracy'] for s in vals])),
  'state_macro_oracle_regret':float(np.mean([s['oracle_regret'] for s in vals])),
  'state_macro_spearman':float(np.mean([s['spearman'] for s in vals if s['spearman'] is not None])),
  'constraint_valid_pairwise_state_macro':float(np.mean([s['strata']['constraint_valid']['pairwise_accuracy'] for s in states if s['strata']['constraint_valid'] and s['strata']['constraint_valid']['pairwise_accuracy'] is not None]))}
 result={'states':states,'candidate_rows':candidate,'aggregate':aggregate,'prior_aggregate':old['aggregate'],
  'checkpoint_sha256':frozen.checkpoint_sha256,'scaler_sha256':frozen.scaler_sha256,'state_count':7,'candidate_count':91,
  'statistical_unit':'state','privileged_recorded_C_R_only_for_comparison':True,'heldout_used_for_selection':False,
  'elapsed_sec':time.monotonic()-start,'gpu_peak_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
 save_once(OUT/'preview-evaluator-results.json',result)
 save_once(OUT/'stage-a-score-complete.json',{'status':'complete','preview_sha256':sha(OUT/'stage-a-preview-complete.json'),
  'results_sha256':sha(OUT/'preview-evaluator-results.json'),'source_gt_sha256':sha(PRIOR/'evaluator-counterfactual-results.json')})
 print('PREVIEW_SCORE_COMPLETE',aggregate,flush=True)
if __name__=='__main__':main()
