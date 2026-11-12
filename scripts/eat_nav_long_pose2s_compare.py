"""Matched M10/M30 accuracy and signed40cm occurrence; executed candidates only."""
from pathlib import Path
import argparse,csv,hashlib,json,time
from collections import defaultdict
import numpy as np
import torch
from visualnav_transformer.eat_nav.single_pose2s_model import SinglePose2s,prepare
from visualnav_transformer.eat_nav.single_pose2s_contract import MODEL_SCHEMA,PROFILE_SHA
from visualnav_transformer.eat_nav.moving_pose2s_geometry import sixd_to_matrix
from visualnav_transformer.eat_nav.single_pose2s_metrics import evaluate_records,save_evaluation,verify_dataset_identity
from visualnav_transformer.eat_nav.long_pose2s import horizontal_overprediction,rate_summary
from eat_nav_single_pose2s_evaluate import make_rows
OLD_REPO=Path('/home/ubuntu/.codex/worktrees/eatnav-model2000-pose2s/worktree')
OLD_CP=OLD_REPO/'docs/eat-nav/model2000-pose2s-20261110/models/seed123/checkpoint.pt'
OLD_SHA='78474a26145137ab12c7486c9b8dde72f2078e6fd61d861d6db58a07ccbb6125'

def format_rate(rate):
 return 'NA' if rate is None else f'{100*rate:.2f}%'

def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def infer(cp,a):
 if cp['schema']!=MODEL_SCHEMA or cp['profile_sha256']!=PROFILE_SHA:raise ValueError('model/profile mismatch')
 model=SinglePose2s();model.load_state_dict(cp['state_dict'],strict=True);model.eval();norm=cp['normalization'];out=[];n=len(a['dem'])
 for i in range(0,n,128):
  b={k:v[i:i+128] for k,v in a.items()};inputs=prepare(b,norm,'cpu')
  with torch.inference_mode():y=model(*inputs)
  out.append({k:v[:,0].numpy() for k,v in y.items()})
 pos=np.concatenate([y['position_z'] for y in out])*np.asarray(norm['position_std'])+np.asarray(norm['position_mean'])
 R,valid=sixd_to_matrix(np.concatenate([y['rotation_6d'] for y in out]));valid&=np.isfinite(pos).all(1)
 return pos,R,valid

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;prod=root/'production';dataset=prod/'dataset';out=root/'results'
 if (out/'complete.json').exists():return
 if out.exists():raise ValueError('partial comparison retained; inspect before retry')
 out.mkdir();torch.set_num_threads(4)
 if sha(OLD_CP)!=OLD_SHA:raise ValueError('M10 identity mismatch')
 with np.load(dataset/'final_test-samples.npz',allow_pickle=False) as f:x={k:f[k] for k in f.files}
 records=json.loads((dataset/'final_test-records.json').read_text());n=len(records)
 if n!=7350:raise ValueError('unexpected final_test windows')
 if any(r['role']!='final_test' for r in records):raise ValueError('mixed evaluation role')
 if np.any(x['dem_stamp_ns'][x['input_valid']]>x['t0_ns'][x['input_valid']]):raise ValueError('future input')
 models={'M10':OLD_CP,'M30':prod/'model-seed123/checkpoint.pt'};predictions={}
 for name,path in models.items():
  cp=torch.load(path,map_location='cpu',weights_only=True)
  if name=='M30':verify_dataset_identity(cp,dataset)
  else:verify_dataset_identity(cp,Path('/home/ubuntu/eatnav-model2000-pose2s-data-20261110/production/dataset'))
  predictions[name]=infer(cp,x)
 common=np.logical_and.reduce([x['pose_mask'],x['input_valid'],predictions['M10'][2],predictions['M30'][2]])
 model_reports={};rate_rows={}
 for name,path in models.items():
  pos,R,valid=predictions[name]
  # Matched population and each own train normalization, deliberately cross-data M10 inference.
  rows=make_rows(x,records,pos,R,common)
  rep=evaluate_records(rows);rep['provenance']=dict(checkpoint_sha256=sha(path),evaluation_dataset_manifest_sha256=sha(dataset/'manifest.json'),matched_valid_windows=int(common.sum()),training_dataset='old10s' if name=='M10' else 'new30s',no_finetuning=True)
  save_evaluation(out/name,rows,rep)
  prefix=[r for r in rows if r['stage']<5];save_evaluation(out/name/'prefix10s',prefix,evaluate_records(prefix))
  model_reports[name]=dict(full30=rep['overall'] if 'overall' in rep else rep.get('summary'),checkpoint_sha256=sha(path))
  rates=[]
  for i,r in enumerate(records):
   good=bool(x['pose_mask'][i] and x['input_valid'][i] and valid[i]);excess=horizontal_overprediction(pos[i],x['position'][i],x['root0'][i,3:7]) if good else None
   rates.append(dict(parent_id=r['parent_id'],field=r['field'],level=r['level'],candidate_id=r['candidate_id'],stage=r['stage'],start_motion=r.get('start_motion'),valid=good,excess_m=excess,positive=bool(good and excess>=.4),reason=r.get('reason')))
  rate_rows[name]=rates;csvwrite(out/(name+'-horizontal-excess.csv'),rates)
  np.savez_compressed(out/(name+'-predictions.npz'),position=pos,rotation=R,prediction_valid=valid,matched_valid=common,truth_position=x['position'],truth_rotation=x['rotation'],root0=x['root0'])
 # Main duration analysis uses only the same frozen M30 predictions.
 rows=rate_rows['M30'];summaries=[]
 for time_name,lo,hi in [('first10',0,5),('full30',0,15),('seconds10to20',5,10),('seconds20to30',10,15)]:
  subset=[r for r in rows if lo<=r['stage']<hi]
  summaries.append(dict(scope='all',time=time_name,**rate_summary(subset)))
  for key in ['field','level','candidate_id','start_motion']:
   for val in sorted({r[key] for r in subset},key=str):summaries.append(dict(scope=key,value=val,time=time_name,**rate_summary([r for r in subset if r[key]==val])))
  for field in sorted({r['field'] for r in subset}):
   for level in range(5):
    for cid in range(14):summaries.append(dict(scope='candidate_field_level',field=field,level=level,candidate_id=cid,time=time_name,**rate_summary([r for r in subset if (r['field'],r['level'],r['candidate_id'])==(field,level,cid)])))
 csvwrite(out/'M30-duration-rates.csv',summaries)
 # Paired parent bootstrap: prefix nested in full30, never independent samples.
 groups=defaultdict(list)
 for r in rows:groups[r['parent_id']].append(r)
 counts=[]
 for v in groups.values():
  short=rate_summary([r for r in v if r['stage']<5]);full=rate_summary(v)
  counts.append([short['positive_windows'],short['valid_windows'],full['positive_windows'],full['valid_windows']])
 counts=np.asarray(counts);rng=np.random.default_rng(123);delta=[]
 for _ in range(2000):
  c=counts[rng.integers(0,len(counts),len(counts))].sum(0)
  if c[1] and c[3]:delta.append(c[2]/c[3]-c[0]/c[1])
 ci=np.percentile(delta,[2.5,97.5]).tolist() if delta else None
 audit=dict(actual_test_windows=n,independent_parents=len(groups),matched_model_valid_windows=int(common.sum()),input_future_dem=0,main_duration_model='M30',numerator_definition='norm(worldXY predicted net displacement)-norm(worldXY actual net displacement)>=0.40m',denominator='valid executed-candidate complete2s windows',paired_parent_bootstrap_rate_difference_CI95=ci,known_terrain_independent_runs=True,old_data_used_for_new_training=False,history_input=False,unexecuted_candidates_scored=False,closedloop_run=False,push_done=False)
 write(out/'audit.json',audit);write(out/'summary.json',dict(models=model_reports,main_rates=[r for r in summaries if r['scope']=='all'],audit=audit))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4));values=[r for r in summaries if r['scope']=='all'];ax.bar([r['time'] for r in values],[100*r['rate'] if r['rate'] is not None else float('nan') for r in values]);ax.set_ylabel('Signed horizontal overprediction >=40cm (%)');ax.set_title('Same frozen M30 / executed candidates / new final_test');fig.tight_layout();fig.savefig(out/'duration-rates.png',dpi=180);plt.close(fig)
 lines=['# 30秒・履歴なしNN 比較結果','', '主比較は同じM30の予測による新記録の先頭10秒と全30秒。入れ子の区間を独立群として扱わない。','', '| 区間 | 該当数 | 有効区間数 | 割合 | 独立試行数 |','|---|---:|---:|---:|---:|']
 for row in values:lines.append(f"|{row['time']}|{row['positive_windows']}|{row['valid_windows']}|{format_rate(row['rate'])}|{row['independent_trials']}|")
 lines+=['','後半の早期終了・欠測による生存者偏りがある。40cm条件は予測誤差の指標で、物理的な阻害原因を確定しない。','M10/M30精度は各ディレクトリのreport.json・490組CSVを参照。正規化は各モデル自身のtrain統計。','Scene1・実機走行・経路選択・GitHub pushは実施していない。']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 write(out/'complete.json',dict(state='analysis_complete_pending_independent_audit',time=time.time(),sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))

if __name__=='__main__':main()
