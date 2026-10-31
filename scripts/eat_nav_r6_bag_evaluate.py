"""Offline top3 frozen R6 evaluation on the specified causal bag intermediates.
No ROS imports/publishers are needed. Outputs are counterfactual choices, not
closed-loop real robot successes. Current joint angles are never mean-imputed.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from eat_nav_r6_extract_bag import load_observations
from eat_nav_r4_star14 import pack_inputs,CANDIDATE_IDS,candidate_paths,ANGLES
from eat_nav_r6_run import read,write,sha
from visualnav_transformer.eat_nav.r6_geometry import calibrated_footprint
from visualnav_transformer.eat_nav.r6_policy import Footprint
from visualnav_transformer.eat_nav.r6_planner import R6Planner,verified_freeze


def overlay(records,observations,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Polygon
 chosen=[records[rank][0] for rank in range(3)]
 fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
 dem=observations['dem'][0,-1];height=np.where(dem[1]==1,dem[0],np.nan)
 for ax,record in zip(axes,chosen):
  ax.imshow(height,origin='lower',extent=[-1,1,-1,1],cmap='terrain');ax.set_facecolor('.75')
  d=record['decision'];selected=d['selected_index'];trajectories=np.asarray(d['predicted_trajectory'])
  for k,path in enumerate(trajectories):ax.plot(path[:,0],path[:,1],color='red' if d['excluded'][k] else 'steelblue',alpha=.25,lw=.7)
  fp=record['footprint'];corners=np.array([[-fp['rear_m']-fp['stopping_margin_m'],-fp['half_width_m']],[fp['front_m']+fp['stopping_margin_m'],-fp['half_width_m']],[fp['front_m']+fp['stopping_margin_m'],fp['half_width_m']],[-fp['rear_m']-fp['stopping_margin_m'],fp['half_width_m']]])
  ax.add_patch(Polygon(corners,fill=False,edgecolor='black',lw=1))
  if selected is not None:
   path=trajectories[selected];ax.plot(path[:,0],path[:,1],color='limegreen',lw=2)
   ax.plot(candidate_paths()[selected,:,0],candidate_paths()[selected,:,1],'--',color='green',lw=1)
   for t in (4,9,14,19):
    yaw=np.arctan2(path[t,3],path[t,4]);c,s=np.cos(yaw),np.sin(yaw)
    occupied=corners@np.array([[c,s],[-s,c]])+path[t,:2]
    ax.add_patch(Polygon(occupied,fill=False,edgecolor='green',alpha=.35,lw=.7))
  ax.scatter([0],[0],c='black',s=15);goal=d['goal_body'];ax.plot([0,goal[0]],[0,goal[1]],':',c='black',lw=.8)
  ax.set_title(f"Top{record['rank']}: "+('STOP' if selected is None else CANDIDATE_IDS[selected]));ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_aspect('equal');ax.set_xlabel('body forward x [m]');ax.set_ylabel('body left y [m]')
 fig.suptitle('First causal bag observation: DEM + predicted paths and current-CAD footprint; grey=unknown\nNo closed-loop robot success is implied')
 fig.savefig(output/'first-observation-top3.png',dpi=160);fig.savefig(output/'first-observation-top3.pdf');plt.close(fig)


def evaluate(intermediate,checkpoint,frozen_file,calibration_file,output,input_interpretation='observed'):
 output=Path(output)
 if output.exists():raise FileExistsError('fresh bag evaluation output required')
 observations,index=load_observations(intermediate)
 if input_interpretation not in ('observed','inpainted_reference'):raise ValueError('explicit input interpretation required')
 if input_interpretation=='observed' and 'observed elevation' not in index['contract'].get('validity_source',''):raise ValueError('primary bag evaluation requires strict observed-elevation mask')
 frozen=verified_freeze(frozen_file);cal=read(calibration_file)
 if frozen['model_sha256']!=sha(checkpoint):raise ValueError('frozen model mismatch')
 arrays_n=len(observations['valid']);goal=np.asarray(index['common_goal']['map_xy']);allrecords=[];summaries=[]
 output.mkdir(parents=True)
 for rank in (1,2,3):
  planner=R6Planner(checkpoint,frozen_file,rank=rank,expected_sha256=sha(checkpoint));records=[]
  for i in range(arrays_n):
   if not observations['valid'][i]:continue
   cutoff=int(observations['decision_receive_ns'][i])
   stamps=np.r_[observations['dem_receive_ns'][i],observations['state_receive_ns'][i],observations['tf_receive_ns'][i]]
   if np.any(stamps>cutoff):raise ValueError('future received input in bag decision')
   arrays=pack_inputs(observations['dem'][i],observations['actor_state'][i],observations['body_state'][i])
   pose=observations['pose_map'][i];c,s=np.cos(pose[3]),np.sin(pose[3]);goal_body=(goal-pose[:2])@np.array([[c,-s],[s,c]])
   flippers=dict(zip(('rr','rl','fr','fl'),map(float,observations['actor_state'][i,10:14])))
   dimensions=calibrated_footprint(cal,flippers,observations['body_state'][i,:3],planner.profile.get('stopping_margin_m',.06));fp=Footprint(**dimensions)
   start=time.perf_counter();decision=planner.select(arrays,goal_body,measured_footprint=fp);decision['goal_body']=goal_body.tolist();elapsed=time.perf_counter()-start
   raw_state=np.r_[observations['actor_state'][i],observations['body_state'][i]]
   keep=[j for j in range(31) if j not in (6,7,8)]
   state_z=(raw_state[keep]-np.asarray(planner.payload['normalization']['state_mean']))/np.asarray(planner.payload['normalization']['state_std'])
   selected=decision['selected_index'];geo=decision['geometry'][selected] if selected is not None else None
   obstacle_evidence=any(g['max_step_m']>.04 for g in decision['geometry'])
   straight_blocked_by_risk=any(reason.startswith('DEM_step_') or reason in ('predicted_roll','predicted_pitch') for reason in decision['reasons'][0])
   candidate_clear=bool(selected is not None and selected!=0 and straight_blocked_by_risk and geo['clearance_m'] is not None and geo['clearance_m']>planner.profile.get('position_error_q95_m',float('inf')) and decision['predicted_goal_progress_m'][selected]>0)
   record=dict(rank=rank,observation_index=i,decision_receive_ns=cutoff,profile_id=planner.profile['id'],goal_map_xy=goal.tolist(),pose_map=pose.tolist(),raw_flipper_angles_rad=flippers,footprint=dimensions,decision=decision,processing_sec=elapsed,
    known_dem_fraction=float((observations['dem'][i,-1,1]==1).mean()),position_error_q95_m=planner.profile.get('position_error_q95_m'),state_abs_zscore_max=float(np.abs(state_z).max()),state_indices_over5sigma=np.flatnonzero(np.abs(state_z)>5).tolist(),max_forecast_displacement_m=float(np.linalg.norm(np.asarray(decision['predicted_trajectory'])[...,:3],axis=-1).max()),out_of_map_centreline_candidates=sum(g.get('coverage_is_lower_bound',False) for g in decision['geometry']),obstacle_evidence_from_DEM=obstacle_evidence,straight_blocked_by_risk=straight_blocked_by_risk,clear_footprint_candidate=candidate_clear,
    geometry_assumption='current flipper pose using audited simulation CAD/FK; future RL flipper motion and manufactured real dimensions not independently verified')
   records.append(record)
  selected=[r['decision']['selected_index'] for r in records];stops=sum(k is None for k in selected);changed=sum(a!=b for a,b in zip(selected[:-1],selected[1:]));unknown=sum(all('map_unknown' in why for why in r['decision']['reasons']) for r in records)
  has_obstacle=any(r['obstacle_evidence_from_DEM'] for r in records);clear=sum(r['clear_footprint_candidate'] for r in records)
  stable_pairs=sum(a['clear_footprint_candidate'] and b['clear_footprint_candidate'] and a['decision']['selected_index']==b['decision']['selected_index'] for a,b in zip(records[:-1],records[1:]))
  classification='地図／観測不足で判定不能' if unknown==len(records) or not has_obstacle else ('継続する回避候補あり（現在姿勢・シミュレーター形状の仮定）' if stable_pairs else '単発の回避候補あり・継続未確認' if clear else '通過余裕を確認できた回避候補なし')
  summary=dict(rank=rank,profile_id=planner.profile['id'],decisions=len(records),STOP_decisions=stops,all_candidates_map_unknown_decisions=unknown,clear_footprint_candidate_decisions=clear,stable_clear_candidate_pairs=stable_pairs,candidate_changes=changed,selected_candidate_ids=['STOP' if k is None else CANDIDATE_IDS[k] for k in selected],classification=(classification if input_interpretation=='observed' else '補間推定を既知と仮定した参考：'+classification),
   return_to_forward_after_curve=sum(a is not None and b is not None and abs(ANGLES[a])>1e-9 and b==0 for a,b in zip(selected[:-1],selected[1:])),processing_mean_sec=float(np.mean([r['processing_sec'] for r in records])),processing_max_sec=max(r['processing_sec'] for r in records),closed_loop_real_robot_success=False)
  write(output/f'top{rank}-decisions.json',records);summaries.append(summary);allrecords.append(records)
 overlay(allrecords,observations,output)
 result=dict(schema='eatnav_r6_frozen_top3_bag_evaluation_v1',input_interpretation=input_interpretation,source_bag=index['source'],intermediate_observations_sha256=sha(Path(intermediate)/'observations.npz'),model_sha256=sha(checkpoint),freeze_sha256=frozen['freeze_sha256'],geometry_calibration_sha256=sha(calibration_file),common_goal=index['common_goal'],summaries=summaries,
  limitations=['Open-loop replay: selected paths do not change later recorded positions or terrain','Current-flipper footprint approximation does not predict future joint motion','Simulation CAD/FK verified against user reference; real encoder offsets/dimensions not independently measured','Real DEM history uses observed receive-time cadence, which differs from nominal250ms training cadence','Primary observed-mask analysis excludes unobserved/inpainted-only cells; reference analysis explicitly assumes inpainted estimates are known','No real robot commands were sent'])
 write(output/'report.json',result);return result


def main():
 torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__)
 for key in ('intermediate','checkpoint','frozen','calibration','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--input-interpretation',choices=('observed','inpainted_reference'),default='observed')
 a=p.parse_args();r=evaluate(a.intermediate,a.checkpoint,a.frozen,a.calibration,a.output,a.input_interpretation);print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
