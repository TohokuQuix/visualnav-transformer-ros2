"""Mini-batch supervised GPU fitting with strict R6 horizons and validation selection."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,pathlib,random,time
import numpy as np
import torch
from visualnav_transformer.eat_nav.r6_model import (CONTRACT,PoseRiskEvaluator,fit_normalization,prepare_inputs,training_support,objective,save_checkpoint,load_checkpoint,predict)
from visualnav_transformer.eat_nav.r6_metrics import event_metrics,trajectory_metrics

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def verify_roles(index,roles):
 seen={};allids=[]
 for role,indices in roles.items():
  if role not in ('train','validation'):raise ValueError('test must not enter training')
  for i in indices:
   if index[i]['role']!=role:raise ValueError('role mismatch')
   g=index[i].get('episode_group',str((index[i].get('field'),index[i].get('terrain_seed',index[i].get('seed')))))
   if g in seen and seen[g]!=role:raise ValueError('episode/terrain seed leakage')
   seen[g]=role;allids.append(i)
 if len(set(allids))!=len(allids):raise ValueError('sample index leakage')
 if not roles.get('train') or not roles.get('validation'):raise ValueError('train and validation required')

def metrics(model,stats,arrays,selected,batch_size=32):
 collected={k:[] for k in ('events','trajectory')};timings=[]
 for start in range(0,len(selected),batch_size):
  t=time.perf_counter();p=predict(model,stats,arrays,selected[start:start+batch_size]);timings.append(time.perf_counter()-t)
  for k,v in p.items():collected[k].append(v)
 pred={k:np.concatenate(v) for k,v in collected.items()};cm=arrays['candidate_mask'][selected]
 return dict(events=event_metrics(pred['events'],arrays['event_targets'][selected],arrays['event_mask'][selected]&cm[:,:,None]),trajectory=trajectory_metrics(pred['trajectory'],arrays['trajectory'][selected],arrays['trajectory_mask'][selected]&cm[:,:,None]),inference_batch_seconds=timings,inference_including_host_transfer_sec_per_group=sum(timings)/len(selected)),pred

def fit(arrays,index,roles,output,*,epochs=300,batch_size=16,device='cuda',seed=123,use_dem=True,on_epoch=None):
 output=pathlib.Path(output)
 if output.exists():raise FileExistsError('fresh training output required')
 verify_roles(index,roles)
 if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; do not silently train CPU')
 torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 stats=fit_normalization(arrays,roles['train']);support=training_support(arrays,roles['train']);model=PoseRiskEvaluator(use_dem).to(device)
 optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5);rng=np.random.default_rng(seed)
 output.mkdir(parents=True);history=[];best=float('inf');best_state=None;best_epoch=None
 val=roles['validation'];train=np.asarray(roles['train']);start=time.perf_counter()
 config=dict(epochs=epochs,batch_size=batch_size,learning_rate=.001,weight_decay=1e-5,optimizer='Adam',gradient_clip_norm=5.,seed=seed,device=device,use_dem=use_dem,event_horizon_sec=2,selection='validation_total_masked_event_plus_position_plus_.25orientation_plus_.01unit',normalization_source='train_only')
 write(output/'config.json',config)
 for epoch in range(1,epochs+1):
  model.train();sums={};steps=0
  for batch in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/batch_size)))):
   optimizer.zero_grad(set_to_none=True);o=model(**prepare_inputs(arrays,batch,stats,device));loss=objective(o,arrays,batch,stats,support)
   if not all(torch.isfinite(v) for v in loss.values()):raise ValueError('nonfinite train loss')
   loss['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
   for k,v in loss.items():sums[k]=sums.get(k,0.)+float(v.detach())*len(batch)
   steps+=len(batch)
  model.eval();vsum=0.;vcount=0
  with torch.inference_mode():
   for vs in range(0,len(val),batch_size):
    batch=val[vs:vs+batch_size];loss=objective(model(**prepare_inputs(arrays,batch,stats,device)),arrays,batch,stats,support);vsum+=float(loss['total'])*len(batch);vcount+=len(batch)
  score=vsum/vcount
  if not np.isfinite(score):raise ValueError('nonfinite validation objective')
  if score<best:best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  row=dict(epoch=epoch,train={k:v/steps for k,v in sums.items()},validation_objective=score,elapsed_sec=time.perf_counter()-start);history.append(row)
  write(output/'progress.json',dict(state='training',done=epoch,planned=epochs,best_epoch=best_epoch,best_validation_objective=best,elapsed_sec=row['elapsed_sec']))
  if on_epoch:on_epoch(row)
 model.load_state_dict(best_state);save_checkpoint(output/'checkpoint.pt',model,stats,support,dict(config=config,roles=roles,best_epoch=best_epoch,training_groups=[index[i].get('episode_group') for i in train],contract=CONTRACT))
 restored,payload=load_checkpoint(output/'checkpoint.pt',device=device);outmetrics={}
 for role,selected in roles.items():
  outmetrics[role],pred=metrics(restored,stats,arrays,selected,batch_size)
  np.savez_compressed(output/(role+'-predictions.npz'),**pred)
 # Exact model/loader parity, finite outputs and shape are mandatory even for partial experiments.
 check=predict(model,stats,arrays,[val[0]]);other=predict(restored,stats,arrays,[val[0]])
 for k in check:assert np.array_equal(check[k],other[k])
 result=dict(state='completed',epochs_completed=epochs,best_epoch=best_epoch,best_validation_objective=best,device=device,support=support,metrics=outmetrics,elapsed_sec=time.perf_counter()-start,checkpoint_sha256=sha(output/'checkpoint.pt'),checkpoint_load_finite_shape_parity_verified=True,contract=CONTRACT)
 write(output/'training.json',dict(result,history=history));write(output/'metrics.json',outmetrics);write(output/'progress.json',result)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--epochs',type=int,default=300);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=123);p.add_argument('--device',default='cuda');p.add_argument('--without-dem',action='store_true');a=p.parse_args()
 with np.load(a.dataset/'samples.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
 index=json.loads((a.dataset/'sample-index.json').read_text());roles=json.loads((a.dataset/'split.json').read_text())['roles']
 result=fit(arrays,index,roles,a.output,epochs=a.epochs,batch_size=a.batch_size,device=a.device,seed=a.seed,use_dem=not a.without_dem,on_epoch=lambda row:print(json.dumps(row),flush=True))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
