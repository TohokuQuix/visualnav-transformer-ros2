"""Offline 91-candidate command preview using actual unmodified ROS follower."""
from __future__ import annotations
import argparse,gzip,json,time,hashlib,re
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from visualnav_transformer.eat_nav.command_preview import pose_from_root_and_map_path,preview_candidate,commands_to_waypoints,map_to_body,body_to_map,command_error
from eat_nav_command_preview_stage0 import ROOT,OUT,PRIOR,sha,save_once,read


def source_rows():
 with gzip.open(PRIOR/'candidate-identity-table.jsonl.gz','rt') as f:return [json.loads(line) for line in f]

def status(value):
 p=OUT/'preview-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(p)

def run(limit=None):
 start=time.monotonic();rows=source_rows();assert len(rows)==91
 provenance=read(OUT/'source-provenance.json');assert sha(PRIOR/'candidate-identity-table.jsonl.gz')==provenance['source_sha256']['candidate-identity-table.jsonl.gz']
 client=FollowerBridgeClient(timeout=50.)
 try:
  for index,row in enumerate(rows):
   sid=row['state_id'];cid=row['candidate_id'];rid=row['run_id']
   output=OUT/'preview-candidates'/f'{sid}--{cid}.json';marker=output.with_suffix('.marker.json')
   folder=ROOT/'pilot/runs'/rid;summary_path=folder/'live-0.json';trace_path=folder/'live-0.trace.json.gz'
   path_file=ROOT/'pilot-inputs'/f"state-{int(sid[-3:]):03d}"/f'{cid}-map-path.json'
   assert sha(summary_path)==row['summary_sha256'] and sha(trace_path)==row['trace_sha256'] and sha(path_file)==row['map_path_sha256']
   input_sha={'summary':sha(summary_path),'trace':sha(trace_path),'path':sha(path_file),'config':sha(OUT/'command-preview-config.yaml')}
   if marker.exists():
    m=read(marker)
    assert m['input_sha256']==input_sha and sha(output)==m['output_sha256'],f'invalid resume marker {cid}'
    print('PREVIEW_RESUME_SKIP',index,sid,cid,flush=True);continue
   summary=read(summary_path);path=read(path_file)
   assert summary['map_path']==path
   initial=pose_from_root_and_map_path(summary['anchor']['root_state'],path)
   # Preview is fully generated before any recorded-command trace is read.
   prediction=preview_candidate(client,path,initial,steps=40,dt=.05)
   closed=np.asarray(prediction['closed_commands'],dtype=float);opened=np.asarray(prediction['open_commands'],dtype=float)
   planned_local=map_to_body(path,initial)
   assert np.max(np.abs(body_to_map(planned_local,initial)-path))<1e-8
   cpath=commands_to_waypoints(closed);opath=commands_to_waypoints(opened)
   payload={'state_id':sid,'terrain':row['terrain'],'candidate_id':cid,'generator':row['generator'],'seed':row['diffusion_seed'],
            'source_run_id':rid,'input_sha256':input_sha,'initial_map_pose':initial,'map_path':path,
            'local_path':planned_local.tolist(),'closed_command':closed.tolist(),'open_command':opened.tolist(),
            'predicted_pose':prediction['predicted_poses'],'closed_trajectory':cpath.tolist(),'open_trajectory':opath.tolist(),
            'bridge_readiness':prediction['reset_readiness'],'control_dt_sec':.05,'trajectory_sample_steps':[5,10,15,20,25,30,35,40],
            'preview_inputs_only':['candidate_path','anchor_pose_yaw','follower_fixed_parameters','control_period','path_goal','fixed_unicycle_kinematics']}
   save_once(output,payload)
   save_once(marker,{'input_sha256':input_sha,'output_sha256':sha(output),'state_id':sid,'candidate_id':cid})
   status({'stage':'preview','state':'running','completed':index+1,'total':91,'last_candidate':f'{sid}/{cid}','elapsed_sec':time.monotonic()-start})
   print('PREVIEW_CANDIDATE_COMPLETE',index+1,sid,cid,'elapsed',round(time.monotonic()-start,1),flush=True)
   if limit is not None and index+1>=limit:break
 finally:client.close()
 if limit is None:
  assert len(list((OUT/'preview-candidates').glob('*.marker.json')))==91
  status({'stage':'preview','state':'complete','completed':91,'elapsed_sec':time.monotonic()-start})
  save_once(OUT/'stage-a-preview-complete.json',{'status':'complete','candidate_count':91,'source_identity_sha256':sha(PRIOR/'candidate-identity-table.jsonl.gz'),'config_sha256':sha(OUT/'command-preview-config.yaml')})

def evaluate():
 rows=source_rows();out=[];audit=[]
 for row in rows:
  sid=row['state_id'];cid=row['candidate_id'];p=OUT/'preview-candidates'/f'{sid}--{cid}.json';marker=p.with_suffix('.marker.json')
  pred=read(p);assert sha(p)==read(marker)['output_sha256']
  trace_path=ROOT/'pilot/runs'/row['run_id']/'live-0.trace.json.gz'
  assert sha(trace_path)==row['trace_sha256']
  with gzip.open(trace_path,'rt') as f:trace=json.load(f)
  actual=np.asarray([r['command'] for r in trace],dtype=float)
  assert actual.shape==(40,3)
  closed=np.asarray(pred['closed_command']);opened=np.asarray(pred['open_command'])
  ce=command_error(closed,actual,linear_threshold=.03,angular_threshold=.05)
  oe=command_error(opened,actual,linear_threshold=.03,angular_threshold=.05)
  observed=commands_to_waypoints(actual);predicted=np.asarray(pred['closed_trajectory'])
  local_path=np.asarray(pred['local_path']);initial=pred['initial_map_pose']
  if cid==('nomad-seed-1002' if sid=='coverage-000' else 'nomad-seed-1001'):
   follower_log=Path(pred['bridge_readiness']['follower_logs'][0])
   log_text=follower_log.read_text(errors='replace')
   match=re.search(r'carrot=\(([-+0-9.]+), ([-+0-9.]+)\)',log_text)
   assert match is not None,f'missing production lookahead log: {follower_log}'
   audit.append({'state_id':sid,'candidate_id':cid,'map_path':pred['map_path'],'initial_map_pose':initial,'robot_local_path':local_path.tolist(),
                 'map_body_map_roundtrip_max_abs_m':float(np.max(np.abs(body_to_map(local_path,initial)-pred['map_path']))),
                 'predicted_command':pred['closed_command'],'predicted_pose':pred['predicted_pose'],
                 'production_lookahead_point_map_xy':[float(match.group(1)),float(match.group(2))],
                 'production_follower_log_sha256':sha(follower_log),'production_follower_log_path':str(follower_log),'gt_8x2_trajectory':predicted.tolist(),
                 'sampling_control_steps':pred['trajectory_sample_steps'],'gt_action_normalization':'none; raw robot-local XY increments'})
  out.append({'state_id':sid,'terrain':row['terrain'],'candidate_id':cid,'generator':row['generator'],'seed':row['diffusion_seed'],
              'closed':ce,'open':oe,'trajectory_rms_m':float(np.sqrt(np.mean(np.sum((predicted-observed)**2,axis=1)))),
              'endpoint_distance_m':float(np.linalg.norm(predicted[-1]-observed[-1])),
              'heading_difference_rad':float(np.arctan2(np.sin(np.arctan2(predicted[-1,1]-predicted[-2,1],predicted[-1,0]-predicted[-2,0])-np.arctan2(observed[-1,1]-observed[-2,1],observed[-1,0]-observed[-2,0])),np.cos(np.arctan2(predicted[-1,1]-predicted[-2,1],predicted[-1,0]-predicted[-2,0])-np.arctan2(observed[-1,1]-observed[-2,1],observed[-1,0]-observed[-2,0])))),
              'recorded_command_sha256':sha(trace_path),'preview_sha256':sha(p)})
 assert len(out)==91 and len(audit)==7
 by_state={sid:[r for r in out if r['state_id']==sid] for sid in sorted({r['state_id'] for r in out})}
 aggregate={'same_cluster_state_macro':float(np.mean([np.mean([r['closed']['same_cluster'] for r in rs]) for rs in by_state.values()])),
            'closed_not_worse_state_count':int(sum(np.mean([r['closed']['linear_rmse_mps']/.03+r['closed']['angular_rmse_radps']/.05 for r in rs])<=np.mean([r['open']['linear_rmse_mps']/.03+r['open']['angular_rmse_radps']/.05 for r in rs]) for rs in by_state.values()))}
 save_once(OUT/'preview-command-fidelity.json',{'candidate_rows':out,'state_count':7,'candidate_count':91,'aggregate':aggregate,'statistical_unit':'state'})
 save_once(OUT/'preview-input-audit.json',{'states':audit,'phase_b_gt_input':'robot-local raw XY increments, 8 future points at .25 s','follower_source':'unmodified production binary via ROS bridge','future_simulator_information_used_for_preview':False})
 print('PREVIEW_FIDELITY_COMPLETE',aggregate,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
 if a.evaluate:evaluate()
 else:run(a.limit)
