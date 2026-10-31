"""Audit real R6 arrays and derive validation branch statistics, never test outcomes.
Two-second event-free outcomes below are not proof of climbing/reaching a2mgoal.
"""
import argparse, collections, json, itertools,math
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.r6_policy import longitudinal_geometry,wilson
from visualnav_transformer.eat_nav.bidirectional_bank import bank
from eat_nav_r6_run import write,sha
from visualnav_transformer.eat_nav.fall_stall_targets import post_state,quaternion,tilt,termination_reasons


def audit_arrays(a,index,strict_bank=False):
 errors=[];groups={};seeds={};roles=collections.Counter()
 if len(index)!=len(a['dem']):errors.append('index_length')
 for m in index:
  role=m['role'];roles[role]+=1
  if role not in ('train','validation'):errors.append('forbidden_role')
  for key,table in ((m.get('episode_group',m['sample_id']),groups),(m.get('terrain_seed',m.get('seed')),seeds)):
   if key in table and table[key]!=role:errors.append('group_or_seed_split_leakage')
   table[key]=role
 for k,v in a.items():
  if not np.isfinite(v).all():errors.append('nonfinite_'+k)
 for k in ('candidate_mask','event_mask','trajectory_mask'):
  if not np.isin(a[k],[0,1]).all():errors.append('nonbinary_'+k)
 if np.any(a['event_mask']&~a['candidate_mask'][...,None]):errors.append('event_on_absent_candidate')
 if np.any(a['trajectory_mask']&~a['candidate_mask'][...,None]):errors.append('pose_on_absent_candidate')
 mask=a['trajectory_mask'];pair=a['trajectory'][...,3:].reshape(*mask.shape,3,2)
 if mask.any() and not np.allclose(np.linalg.norm(pair[mask],axis=-1),1,atol=1e-4):errors.append('orientation_pairs')
 if strict_bank:
  candidates=bank();paths=np.stack([c.waypoints for c in candidates]).astype(np.float32);directions=np.asarray([c.direction for c in candidates],np.float32).reshape(14,1)
  bank_mask=a['candidate_mask'].astype(bool)
  if not np.array_equal(a['paths'][bank_mask],np.broadcast_to(paths,a['paths'].shape)[bank_mask]) or not np.array_equal(a['directions'][bank_mask],np.broadcast_to(directions,a['directions'].shape)[bank_mask]):errors.append('candidate_bank_geometry_or_direction_order')
 result=dict(passed=not errors,errors=sorted(set(errors)),group_count=len(index),roles=dict(roles),actual_candidate_count=int(a['candidate_mask'].sum()),valid_future_points=int(mask.sum()),event_horizon_sec=2,test_included=False,event_support={})
 for role in ('train','validation'):
  ids=[i for i,m in enumerate(index) if m['role']==role];known=a['event_mask'][ids]&a['candidate_mask'][ids,:,None]
  result['event_support'][role]=dict(positive=(a['event_targets'][ids]*known).sum(axis=(0,1)).astype(int).tolist(),negative=((1-a['event_targets'][ids])*known).sum(axis=(0,1)).astype(int).tolist(),unknown=int((~known).sum()))
 return result


def raw_observations(member,slot):
 if member.get('source'):
  path=Path(member['source'])/f'candidate-{slot:02d}'/'trace.jsonl'
 elif member.get('sources'):
  order=[0,1,2,7,3,4,5,6,8,9,10,11,12,13]
  path=Path(member['sources'][order.index(slot)])/'trace.jsonl'
 else:return None
 rows=[]
 with path.open() as stream:
  for line in itertools.islice(stream,40):
   row=json.loads(line)
   if row.get('boundary_censor') or row.get('reset') or row.get('manual_stop_before_step'):break
   rows.append(row)
   if row.get('terminal') or row.get('manual_stop') or row.get('external_intervention'):break
 return rows


def observed_angles(rows):
 roll=[];pitch=[];indices=[];onsets={};high=0;distance=0.;initial=np.asarray(rows[0]['root_state'][:3]);last=initial;onset_distance={}
 for j,row in enumerate(rows):
  state=post_state(row);q=quaternion(state)
  if q is None:continue
  w,x,y,z=q;roll.append(math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))));pitch.append(math.degrees(math.asin(np.clip(2*(w*y-z*x),-1,1))));indices.append(j)
  angle=tilt(state);high=high+1 if angle is not None and angle>=60 else 0
  after=np.asarray(state[:3]);distance+=float(np.linalg.norm(after-last));last=after
  for name,yes in (('tip_over',(angle is not None and angle>=85) or 'tilt_over' in termination_reasons(row)),('near_fall',high>=5)):
   if yes and name not in onsets:onsets[name]=(j+1)*.05;onset_distance[name]=distance
 return np.asarray(roll),np.asarray(pitch),np.asarray(indices),dict(first_confirmation_sec=onsets,path_distance_to_confirmation_m=onset_distance,sample_period_sec=.05,initial_tilt_deg=tilt(rows[0]['root_state']))


def angle_rate(angles,indices,dt=.1):
 if len(angles)<2:return 0.
 delta=np.diff(np.radians(angles));wrapped=np.degrees(np.arctan2(np.sin(delta),np.cos(delta)))
 return float(np.max(np.abs(wrapped)/(np.diff(indices)*dt)))


def branch_records(a,index):
 result=[]
 for i,m in enumerate(index):
  if m['role']!='validation':continue
  for k in range(14):
   if not a['candidate_mask'][i,k]:continue
   t=a['trajectory'][i,k];valid=a['trajectory_mask'][i,k].astype(bool);evknown=a['event_mask'][i,k].astype(bool)
   raw=raw_observations(m,k)
   if raw:
    roll,pitch,angle_indices,onsets=observed_angles(raw);sample_dt=.05
   elif valid.any():
    roll=np.degrees(np.arctan2(t[valid,5],t[valid,6]));pitch=np.degrees(np.arctan2(t[valid,7],t[valid,8]));angle_indices=np.flatnonzero(valid);sample_dt=.1;onsets={}
   else:continue
   if not len(roll):continue
   path=np.vstack([np.zeros(2),a['paths'][i,k]])
   points=[]
   for start,end in zip(path[:-1],path[1:]):
    points.extend(start+(end-start)*s for s in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(end-start)/.05))),endpoint=False))
   points=np.vstack([points,path[-1]])
   cell=np.floor((points+1)/.1).astype(int);inside=((cell>=0)&(cell<20)).all(axis=1);cell=cell[inside]
   if len(cell):cell=cell[np.r_[True,np.any(np.diff(cell,axis=0),axis=1)]]
   h=a['dem'][i,-1,0];known=a['dem'][i,-1,1];geo=dict(kind='unknown',max_step_m=0.,tread_depths_m=[],stable_tread_count=0)
   geom_known=bool(len(cell)>=2 and np.all(known[cell[:,1],cell[:,0]]==1))
   if geom_known:
    dist=np.cumsum(np.r_[0,np.linalg.norm(np.diff(cell,axis=0),axis=1)*.1]);geo=longitudinal_geometry(dist,h[cell[:,1],cell[:,0]])
   observed_failure=bool(np.any(a['event_targets'][i,k][evknown]>0))
   r=dict(split='validation',group_id=m.get('episode_group',m['sample_id']),sample_id=m['sample_id'],candidate=k,field=m.get('field'),level=m.get('level'),seed=m.get('terrain_seed',m.get('seed')),scene=m.get('scene','field'),source_observation_hz=1/sample_dt,observed_onsets=onsets,
    complete=bool(geom_known and ((valid.all() and evknown.all()) or observed_failure)),failed=observed_failure,full_pose_horizon=bool(valid.all()),observed_failure_before_censor=bool(observed_failure and not valid.all()),
    outcome_definition='complete_two_second_event_free; not a two_metre_climb_success',
    max_abs_pitch_deg=float(np.abs(pitch).max()),max_abs_roll_deg=float(np.abs(roll).max()),pitch_min_deg=float(pitch.min()),pitch_max_deg=float(pitch.max()),roll_min_deg=float(roll.min()),roll_max_deg=float(roll.max()),
    pitch_positive_duration_sec=float((pitch>0).sum()*sample_dt),pitch_negative_duration_sec=float((pitch<0).sum()*sample_dt),roll_positive_duration_sec=float((roll>0).sum()*sample_dt),roll_negative_duration_sec=float((roll<0).sum()*sample_dt),
    max_pitch_rate_degps=angle_rate(pitch,angle_indices,sample_dt),max_roll_rate_degps=angle_rate(roll,angle_indices,sample_dt),
    step_cm=float(geo['max_step_m']*100),geometry_kind=geo['kind'],geometry=geo,geometry_source='initial causal DEM candidate centreline; not measured footprint support',
    contact_metadata=m.get('contact_metadata',{}).get(str(k),{}),pre_anchor_contact=m.get('observed_contact_before_anchor'),pre_anchor_contact_scope=m.get('pre_anchor_contact_observation_scope','prefix_only_or_unknown'),prefix_contact_complete=m.get('prefix_contact_observation_complete'),
    tilt_duration_by_threshold_sec={str(threshold):dict(roll=float((np.abs(roll)>=threshold).sum()*sample_dt),pitch=float((np.abs(pitch)>=threshold).sum()*sample_dt)) for threshold in (15,30,45,60,85)},
    initial_body_state=a['body_state'][i].tolist(),initial_flipper_features=a['actor_state'][i].tolist(),event_targets=a['event_targets'][i,k].tolist(),event_mask=evknown.tolist(),valid_pose_points=int(valid.sum()),future_displacement_m=float(np.linalg.norm(t[np.flatnonzero(valid)[-1],:2])) if valid.any() else None)
   result.append(r)
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.output.exists():raise FileExistsError('fresh audit output required')
 with np.load(args.dataset/'samples.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
 index=json.loads((args.dataset/'sample-index.json').read_text());report=audit_arrays(a,index,strict_bank=True);report['dataset_sha256']=sha(args.dataset/'samples.npz')
 records=branch_records(a,index);cells={}
 for r in records:
  key=str((r['field'],r['level'],r['scene'],r['geometry_kind']));cells.setdefault(key,[]).append(r)
 summaries={}
 for key,rows in cells.items():
  complete=[r for r in rows if r['complete']];failed=sum(r['failed'] for r in complete)
  summaries[key]=dict(total=len(rows),complete=len(complete),failed=failed,failure_wilson95=wilson(failed,len(complete)),quantiles={name:np.quantile([r[name] for r in rows],[0,.25,.5,.75,.95,1]).tolist() for name in ('max_abs_pitch_deg','max_abs_roll_deg','step_cm','max_pitch_rate_degps','max_roll_rate_degps')})
 args.output.mkdir(parents=True);write(args.output/'audit.json',report);write(args.output/'validation-records.json',records);write(args.output/'field-level-statistics.json',summaries)
 print(json.dumps(report,indent=2))
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
