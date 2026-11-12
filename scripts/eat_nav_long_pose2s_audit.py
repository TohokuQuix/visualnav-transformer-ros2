from pathlib import Path
import json,hashlib,csv,sys,numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
repo=Path('/home/ubuntu/.codex/worktrees/eatnav-30s-nohistory/worktree');sys.path[:0]=[str(repo/'src')]
from visualnav_transformer.eat_nav.bidirectional_bank import bank
root=Path('/home/ubuntu/eatnav-model2000-30s-nohistory-20261112');ds=root/'production/dataset';out=root/'independent-audit';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):(out/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def mat(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
manifest=json.loads((ds/'manifest.json').read_text());sources=json.loads((ds/'source-index.json').read_text());audit=json.loads((ds/'audit.json').read_text())
assert len(manifest['parents'])==1330==len(sources)
expected={p['parent_id']:p for p in manifest['parents']};parentroles={};raw_hashes={};initial_hashes={};allrows=[];teacher_max=0.;rotation_max=0.;odom_max=0.;verified_valid=0;driftrows=[];metaindex={}
for role,nrows in [('train',5250),('validation',7350),('final_test',7350)]:
 with np.load(ds/(role+'-samples.npz')) as f:arrays={k:f[k] for k in f.files}
 records=json.loads((ds/(role+'-records.json')).read_text());assert len(records)==nrows
 groups={}
 for i,r in enumerate(records):groups.setdefault(r['parent_id'],[]).append((i,r))
 for source in [s for s in sources if expected[s['parent_id']]['role']==role]:
  pid=source['parent_id'];meta=json.loads(Path(source['metadata_path']).read_text());metaindex[pid]=meta
  assert meta['parent']==expected[pid] and meta['trial_seconds']==30
  assert sha(Path(source['raw_path']))==source['raw_sha256']==meta['raw_sha256']
  assert sha(Path(source['odometry_npz']))==source['odometry_sha256']
  raw_hashes.setdefault(source['raw_sha256'],set()).add(role);parentroles.setdefault(pid,set()).add(role)
  with np.load(source['raw_path']) as f:
   roots=f['root'];alive=f['alive'];cmd=f['command'];stamp=f['stamps_ns'];jpos=f['joint_position'];jvel=f['joint_velocity']
  assert roots.shape==(601,13) and cmd.shape==(600,3) and np.array_equal(stamp,np.arange(601)*50000000)
  initial=roots[0].astype(float).copy();initial[:3]-=meta['env_origin_w'];ih=hashlib.sha256(np.concatenate([initial,jpos[0],jvel[0]]).astype('<f8').tobytes()).hexdigest();initial_hashes.setdefault(ih,set()).add(role)
  with np.load(source['odometry_npz']) as o:ot=o['stamps_ns'];orecv=o['received_ns'];op=o['poses']
  for i,r in groups[pid]:
   j=r['stage'];s=j*40;e=s+40;complete=bool(alive[s:e].all());good=bool(arrays['pose_mask'][i]&arrays['input_valid'][i])
   assert r['candidate_id']==expected[pid]['candidates'][j]
   if good:
    assert complete and alive[s];assert arrays['dem_stamp_ns'][i]<=s*50000000 and arrays['dem_received_ns'][i]<=s*50000000
    R0=mat(roots[s,3:7]);R2=mat(roots[e,3:7]);target=R0.T@(roots[e,:3]-roots[s,:3]);six=np.r_[ (R0.T@R2)[:,0],(R0.T@R2)[:,1] ]
    teacher_max=max(teacher_max,float(np.max(np.abs(target-arrays['position'][i]))));rotation_max=max(rotation_max,float(np.max(np.abs(six-arrays['rotation'][i]))))
    ii=r['odom_pose_indices'];assert len(ii)==2
    assert all(ot[k]<=s*50000000 and orecv[k]<=s*50000000 for k in ii)
    delta=(ot[ii[1]]-ot[ii[0]])*1e-9;v=mat(op[ii[1],3:7]).T@(op[ii[1],:3]-op[ii[0],:3])/delta
    odom_max=max(odom_max,float(np.max(np.abs(v-arrays['state'][i,18:21]))));verified_valid+=1
    truthv=R0.T@roots[s,7:10];driftrows.append(dict(role=role,parent_id=pid,stage=j,field=r['field'],level=r['level'],speed_lio=float(np.linalg.norm(v)),speed_true=float(np.linalg.norm(truthv)),velocity_error=float(np.linalg.norm(v-truthv))))
   if complete:
    net=float(np.linalg.norm(roots[e,:2]-roots[s,:2]));requested=float(np.abs(cmd[s:e,0]).mean())
    allrows.append(dict(role=role,parent_id=pid,field=r['field'],level=r['level'],stage=j,proxy_stall=requested>=.1 and net<.05))
 print('audited',role,len(groups),'parents',flush=True)
assert teacher_max<1e-6 and rotation_max<1e-6 and odom_max<1e-3
assert all(len(x)==1 for groups in [parentroles,raw_hashes,initial_hashes] for x in groups.values())
def proxy(rs):
 n=len(rs);k=sum(r['proxy_stall'] for r in rs)
 return dict(valid_complete_raw_windows=n,proxy_stall_windows=k,rate=k/n if n else None,independent_parents=len({r['parent_id'] for r in rs}))
proxy_summary={role:{name:proxy([r for r in allrows if (role=='all' or r['role']==role) and lo<=r['stage']<hi]) for name,lo,hi in [('first10',0,5),('seconds10to20',5,10),('seconds20to30',10,15),('full30',0,15)]} for role in ['all','train','validation','final_test']}
save('raw-stall-proxy.json',dict(definition='requested mean abs body vx>=.1m/s AND actual net worldXY displacement<.05m over complete2s; descriptive only, not40cm rule',summary=proxy_summary))
quality={}
for role in ['train','validation','final_test']:
 rs=[r for r in driftrows if r['role']==role];v=np.array([r['speed_lio'] for r in rs]);tv=np.array([r['speed_true'] for r in rs]);er=np.array([r['velocity_error'] for r in rs]);quality[role]=dict(valid_windows=len(rs),lio_speed_median=float(np.median(v)),lio_speed_p95=float(np.percentile(v,95)),lio_speed_max=float(v.max()),true_speed_max=float(tv.max()),lio_speed_gt2mps=int(sum(v>2)),lio_speed_gt10mps=int(sum(v>10)),velocity_error_mean=float(er.mean()),velocity_error_p95=float(np.percentile(er,95)))
save('fastlio-input-quality.json',dict(summary=quality,largest_errors=sorted(driftrows,key=lambda r:r['velocity_error'],reverse=True)[:30],note='Diagnostic thresholds only; no samples excluded or normalization changed. Actual FAST-LIO can drift; not NN motion truth.'))
# Independent metrics from saved predictions: scipy rotations + numerical path projection.
with np.load(ds/'final_test-samples.npz') as f:x={k:f[k] for k in f.files}
records=json.loads((ds/'final_test-records.json').read_text());summary=json.loads((root/'results/summary.json').read_text());metric_audit={}
for model in ['M10','M30']:
 y=np.load(root/'results'/f'{model}-predictions.npz');ids=np.flatnonzero(y['matched_valid']);values={k:[] for k in ['longitudinal_cm','lateral_cm','height_cm','distance_3d_cm','roll_deg','pitch_deg','yaw_deg','rotation_deg']};rate=[]
 for i in ids:
  r=records[i];R0=mat(x['root0'][i,3:7]);yaw=np.arctan2(R0[1,0],R0[0,0]);cy,sy=np.cos(yaw),np.sin(yaw);Ryaw=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
  pw=R0@y['position'][i];gw=R0@x['position'][i];pred=Ryaw.T@pw;truth=Ryaw.T@gw;res=pred-truth;can=bank()[r['candidate_id']];theta=np.deg2rad(can.arc_angle_deg);d=can.direction
  if abs(theta)<1e-12:s=np.clip(d*truth[0],0,.75)
  else:
   def point(s):return d*.75/theta*np.array([np.sin(theta*s/.75),1-np.cos(theta*s/.75)])
   def obj(s):return np.sum((point(s)-truth[:2])**2)
   sol=minimize_scalar(obj,bounds=(0,.75),method='bounded',options={'xatol':1e-14});s=min([0,.75,sol.x],key=obj)
  tangent=d*np.array([np.cos(theta*s/.75),np.sin(theta*s/.75)]);normal=np.array([-tangent[1],tangent[0]])
  for k,v in [('longitudinal_cm',tangent@res[:2]*100),('lateral_cm',normal@res[:2]*100),('height_cm',res[2]*100),('distance_3d_cm',np.linalg.norm(res)*100)]:values[k].append(abs(float(v)))
  six=x['rotation'][i];u=six[:3]/np.linalg.norm(six[:3]);v=six[3:]-np.dot(u,six[3:])*u;v/=np.linalg.norm(v);Rt=np.stack([u,v,np.cross(u,v)],1)
  Rp=y['rotation'][i];ar=Rotation.from_matrix(R0@Rp).as_euler('xyz');br=Rotation.from_matrix(R0@Rt).as_euler('xyz');angles=np.abs(np.rad2deg((ar-br+np.pi)%(2*np.pi)-np.pi))
  for k,v in zip(['roll_deg','pitch_deg','yaw_deg'],angles):values[k].append(float(v))
  values['rotation_deg'].append(float(np.rad2deg(Rotation.from_matrix(Rp.T@Rt).magnitude())))
  rate.append(dict(parent_id=r['parent_id'],stage=r['stage'],positive=bool(np.linalg.norm(pw[:2])-np.linalg.norm(gw[:2])>=.4)))
 calc={k:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v))) for k,v in values.items()};reported=summary['models'][model]['full30'];deltas={k+'_'+s:abs(v-reported[k+'_'+s]) for k,vals in calc.items() for s,v in vals.items()}
 assert max(deltas.values())<1e-4,(model,max(deltas.values()))
 rates={name:dict(positive_windows=sum(r['positive'] for r in rate if lo<=r['stage']<hi),valid_windows=sum(lo<=r['stage']<hi for r in rate)) for name,lo,hi in [('first10',0,5),('seconds10to20',5,10),('seconds20to30',10,15),('full30',0,15)]}
 if model=='M30':
  for rr in summary['main_rates']:assert rates[rr['time']]['positive_windows']==rr['positive_windows'] and rates[rr['time']]['valid_windows']==rr['valid_windows']
 metric_audit[model]=dict(valid_windows=len(ids),max_metric_difference=max(deltas.values()),metrics=calc,rates=rates,position_error_gt1m=sum(v>100 for v in values['distance_3d_cm']),position_error_gt5m=sum(v>500 for v in values['distance_3d_cm']))
 print('independent metrics passed',model,flush=True)
lock=json.loads((root/'baseline-lock.json').read_text())
for name,digest in lock['files_sha256'].items():assert sha(Path(lock['source_repo'])/name)==digest
cp=root/'production/model-seed123/checkpoint.pt';assert sha(cp)=='520a0af98184e26a74d01f6f75b4fec6c410eb22b8c2fe91f0dafce50e8b1f1e'
finish=dict(state='independent_audit_passed_with_reported_FASTLIO_quality_limitations',parents=1330,planned_windows=19950,valid_teacher_windows=verified_valid,missing_teacher_windows=19950-verified_valid,teacher_max_abs_difference_m=teacher_max,rotation6d_max_abs_difference=rotation_max,causal_velocity_max_abs_difference=odom_max,parent_raw_initial_cross_role_overlap=0,known_terrain_overlap_hashes=31,baseline_files_unchanged=True,checkpoint_sha256=sha(cp),metrics=metric_audit)
save('audit.json',finish)
print('PROXY',json.dumps(proxy_summary));print('QUALITY',json.dumps(quality))
