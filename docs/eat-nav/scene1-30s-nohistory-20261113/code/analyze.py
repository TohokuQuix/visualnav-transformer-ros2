"""Audit real traces and produce ten-trial results/failure map, no new simulation."""
from pathlib import Path
import json,csv,hashlib,sys,collections
import numpy as np
from scipy.spatial.transform import Rotation
from nav_contract import outcome,inside_course,scene_location,choose_best,diagnose_motion,needs_walls,initialization_ok,audit_decision
ROOT=Path(__file__).resolve().parents[1]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def analyze(cohort,index):
 protocol=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'results'/cohort;out.mkdir(parents=True,exist_ok=True);rows=[];audit=[];details=[]
 if len(index)!=10 or len(set(index))!=10:raise ValueError('ten unique formal trialIDs required')
 for tid,folder in sorted(index.items()):
  d=Path(folder);r=json.loads((d/'run.json').read_text());z=np.load(d/'trace.npz');t=z['time_s'];p=z['root'];cmd=z['command'];n=len(t)
  if r['parent']['trial_id']!=tid:raise ValueError('trial identity mismatch')
  assert initialization_ok(r['initialization']['xy_deviation_m'],np.deg2rad(r['initialization']['yaw_deviation_deg']),r['initialization']['warm_termination'],protocol['initialization_tolerance_xy_m'],protocol['initialization_tolerance_yaw_deg'])
  assert r['protocol_sha256']==hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
  assert n==len(cmd)+1 and np.allclose(t,np.arange(n)*.05) and np.isfinite(p).all() and np.isfinite(z['state']).all()
  assert np.allclose(p[:,:3]+r['origin'],z['root_world'][:,:3]);R=Rotation.from_quat(p[:,[4,5,6,3]]).as_matrix();tilt=np.rad2deg(np.arccos(np.clip(R[:,2,2],-1,1)));distance=np.linalg.norm(p[:,:2]-protocol['goal_xy'],axis=1);inside=inside_course(p[:,:2]);bypass=bool((~inside).any());valid_prefix=np.logical_and.accumulate(inside)
  re=outcome(float(t[-1]),p[-1,:3],tilt[-1],bypass,goal=protocol['goal_xy'],limit=300.)
  assert r['result']==re and r['result']!='technical_failure',(tid,r['result'],re)
  assert bypass==r['bypass_observed'];prefix_ids=np.flatnonzero(valid_prefix)
  best=int(prefix_ids[np.argmin(distance[prefix_ids])]) if len(prefix_ids) else 0
  assert abs(float(distance[best])-r['minimum_course_valid_goal_distance_m'])<2e-5
  diag=diagnose_motion(t,p,cmd);details.extend([dict(trial_id=tid,**x) for x in diag]);traveled=float(np.linalg.norm(np.diff(p[:,:3],axis=0),axis=1).sum())
  row=dict(trial_id=tid,result=r['result'],duration_s=float(t[-1]),bypass_observed=bypass,goal_counted=r['result']=='goal_reached' and not bypass,minimum_course_valid_goal_distance_m=float(distance[best]),best_time_s=float(t[best]),final_goal_distance_m=float(distance[-1]),final_x_m=float(p[-1,0]),final_y_m=float(p[-1,1]),final_z_m=float(p[-1,2]),final_region=scene_location(p[-1,:2])['region'],travel_distance_m=traveled,max_tilt_deg=float(tilt.max()),minimum_height_m=float(p[:,2].min()),below_terrain_observed=bool((p[:,2]<-1).any()),initial_x_m=float(p[0,0]),initial_y_m=float(p[0,1]),initial_yaw_deg=float(Rotation.from_matrix(R[0]).as_euler('xyz',degrees=True)[2]),linear_blocked_seconds=sum(x['duration_s'] for x in diag if x['kind']=='linear_motion_blocked'),turn_blocked_seconds=sum(x['duration_s'] for x in diag if x['kind']=='turn_motion_blocked'),command_stop_seconds=sum(x['duration_s'] for x in diag if x['kind']=='commanded_stop'),trace=str(d/'trace.npz'),run_json=str(d/'run.json'),trace_sha256=hashlib.sha256((d/'trace.npz').read_bytes()).hexdigest())
  rows.append(row);audit.append(dict(trial_id=tid,finite=True,terminal_recomputed=True,bypass_recomputed=True,minimum_distance_recomputed=True,root_count=n))
 # Check recorded candidate boundaries from each batch (once), independent of success outcome.
 batches={Path(folder).parent for folder in index.values()};checked=0
 for batch in batches:
  for d in json.loads((batch/'decisions.json').read_text()):
   if d['trial_id'] not in index:continue
   trace=np.load(Path(index[d['trial_id']])/'trace.npz');s=int(d['step']);r=json.loads((Path(index[d['trial_id']])/'run.json').read_text());audit_decision(d,trace['root_world'][s],trace['state'][s],np.asarray(r['origin']),np.asarray(r['goal_world']))
   if d['candidate_id'] is not None:
    cid=d['candidate_id'];assert d['allowed'][cid] and d['course_allowed'][cid] and d['original_tilt_allowed'][cid];checked+=1
   assert d['dem_stamp_ns']<=d['t0_ns'] and d['dem_received_ns']<=d['t0_ns']
 best=choose_best(rows);summary=dict(cohort=cohort,completed_trials=10,goal_count=sum(x['goal_counted'] for x in rows),outcomes=dict(collections.Counter(x['result'] for x in rows)),bypass_trials=sum(x['bypass_observed'] for x in rows),needs_walls=needs_walls(rows),best_trial=best,protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),trial_rows=rows)
 write(out/'summary.json',summary);write(out/'motion-diagnostics.json',details);write(out/'audit.json',dict(state='passed',trials=audit,decisions_checked=checked,executed_candidates_only=True,not_training=True))
 for filename,data in [('trials.csv',rows),('motion-diagnostics.csv',details)]:
  keys=list(dict.fromkeys(k for x in data for k in x)) or ['trial_id','kind','t0_s','t1_s','duration_s']
  with (out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
 plot(cohort,index,rows,details,out,protocol)
 lines=['# Scene1 model2000 10-trial navigation',f"Goal count: {summary['goal_count']}/10. Outcomes: {summary['outcomes']}. Outside-course trials: {summary['bypass_trials']}.",'Start requested(.6,1.7),yaw-y; goal(4.3,.5), radius.2m, heading ignored. Maximum300 simulated seconds. 30s nohistory NN+same Actor, B60deg with explicit course boundary exclusion.','|trial|result|seconds|best distance m|final XY m|region|blocked linear s|blocked turn s|','|---|---|---:|---:|---|---|---:|---:|']
 for x in rows:lines.append(f"|{x['trial_id']}|{x['result']}|{x['duration_s']:.2f}|{x['minimum_course_valid_goal_distance_m']:.3f}|({x['final_x_m']:.3f},{x['final_y_m']:.3f})|{x['final_region']}|{x['linear_blocked_seconds']:.1f}|{x['turn_blocked_seconds']:.1f}|")
 lines+=['','Blocked diagnostics are post-hoc2s observations, not control rules. Linear: meanabs requestedvx>=.1m/s with total3Dmotion<.04m/2s. Turn: meanabs requestedwz>=.2rad/s with actualyawtravel<.1rad/2s. Commanded STOP is separate and does not guarantee physical stillness. Exact locations/times in motion-diagnostics.csv and trajectories.npz.','Successes involving outside-course detours are not counted. Best video must render recorded root/joints; it does not constitute a fresh deterministic resimulation. Finite terrain visibility is ideal geometric DEM, not actual FAST-LIO.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'report.txt').write_text('\n'.join(lines)+'\n');return summary

def plot(cohort,index,rows,details,out,protocol):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 m=np.load('/home/ubuntu/eatnav-scene1-onix-20261111/experiment/converted/scene1_mesh.npz');h=m['vertices'][:225*347,2].reshape(225,347);fig,ax=plt.subplots(figsize=(12,8));ax.imshow(h,origin='lower',extent=[-1,5.92,-1,3.48],cmap='terrain',alpha=.7);arrays={}
 for i,x in enumerate(rows):
  z=np.load(Path(index[x['trial_id']])/'trace.npz');p=z['root'];color=plt.get_cmap('tab10')(i);ax.plot(p[:,0],p[:,1],c=color,lw=1.3,label=x['trial_id']+' '+x['result']);ax.scatter(p[-1,0],p[-1,1],c=[color],marker='x' if x['result']!='goal_reached' else '*',s=80);ax.annotate(str(i+1),p[-1,:2],fontsize=8);arrays[x['trial_id']+'_root']=p;arrays[x['trial_id']+'_time_s']=z['time_s']
 ax.scatter(*protocol['start_xy'],c='purple',s=100,marker='o');ax.arrow(*protocol['start_xy'],0,-.35,color='purple',width=.02);ax.scatter(*protocol['goal_xy'],c='red',s=180,marker='*');ax.add_patch(plt.Circle(protocol['goal_xy'],.2,fill=False,color='red'));ax.set(xlabel='Scene1 x [m]',ylabel='Scene1 y [m]',title=cohort+' actual trajectories/end failure locations',xlim=(-.5,5.4),ylim=(-.5,3.0));ax.set_aspect('equal');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1));fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig);np.savez_compressed(out/'trajectories.npz',**arrays)

if __name__=='__main__':
 cohort=sys.argv[1];analyze(cohort,json.loads((ROOT/'indices'/(cohort+'.json')).read_text()))
