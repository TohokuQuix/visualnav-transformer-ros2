"""Summarize completed exploratory selector runs; no formal accuracy/safety claim."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.single_pose2s_selector_summary import summarize_rollout


def write_csv(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads((a.root/'manifest.json').read_text());expected={p['parent_id'] for b in m['batches'] for p in b['parents']};identity=json.loads((a.root/'evaluation-identity.json').read_text());runtime_ids=set();runs=[];windows=[];sources=[];selection=Counter()
 for seal_path in sorted((a.root/'raw').glob('*/host-seal.json')):
  batch=seal_path.parent;seal=json.loads(seal_path.read_text())
  if seal['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('mixed evaluator checkpoints')
  runtime=json.loads((batch/'runtime-imports.json').read_text());runtime_ids.add(runtime['canonical_sha256'])
  if len(runtime_ids)>1 or runtime['worker_sha256']!=identity['worker_sha256']:raise ValueError('mixed rollout runtimes')
  decisions=json.loads((batch/'selector-decisions.json').read_text());goals=json.loads((batch/'selector-goals.json').read_text());init=json.loads((batch/'initial-selector-state.json').read_text());indices={p['parent_id']:i for i,p in enumerate(init['parents'])}
  for meta_path in sorted(batch.glob('*/metadata.json')):
   meta=json.loads(meta_path.read_text());pid=meta['parent']['parent_id'];rawpath=meta_path.with_name('raw.npz');h=hashlib.sha256(rawpath.read_bytes()).hexdigest()
   if pid not in expected or h!=meta['raw_sha256'] or h!=seal['raw_hashes'][str(rawpath.relative_to(batch))]:raise ValueError('unknown or changed raw rollout')
   with np.load(rawpath) as raw:r=summarize_rollout(raw,meta,decisions,goals['goals'][indices[pid]],goals['goal_step'][indices[pid]])
   windows.extend([dict(parent_id=pid,**w) for w in r.pop('window_outcomes')]);runs.append(r);sources.append(dict(parent_id=pid,raw_path=str(rawpath),raw_sha256=h,terrain_sha256=meta['terrain']['geometry_sha256'],initial_root=init['root'][indices[pid]],initial_joint_position=init['joint_position'][indices[pid]],initial_joint_velocity=init['joint_velocity'][indices[pid]]))
  if any(not d.get('actual_fastlio') or d.get('input_t0_ns')!=d['step']*50000000 for d in decisions):raise ValueError('noncausal or non-FAST-LIO decision')
  selection.update(d['candidate_id'] for d in decisions)
 if len(runs)!=len(expected) or len({r['parent_id'] for r in runs})!=len(expected):raise ValueError(f'incomplete evaluation: {len(runs)}/{len(expected)} runs; do not publish as completed')
 def aggregate(rows):
  return dict(valid_runs=len(rows),goal_reached_runs=sum(r['goal_reached'] for r in rows),tip_over_runs=sum(r['tip_over_observed'] for r in rows),mean_route_progress_m=float(np.mean([r['route_progress_m'] for r in rows])),mean_filter_stop_seconds=float(np.mean([r['filter_stop_seconds'] for r in rows])),mean_follower_zero_command_seconds=float(np.mean([r['follower_zero_command_seconds'] for r in rows])),linear_stall_windows=sum(r['linear_stall_windows'] for r in rows),turn_stall_windows=sum(r['turn_stall_windows'] for r in rows),mean_linear_stall_positive_seconds=float(np.mean([r['linear_stall_positive_seconds'] for r in rows])),mean_turn_stall_positive_seconds=float(np.mean([r['turn_stall_positive_seconds'] for r in rows])),mean_duration_seconds=float(np.mean([r['duration_seconds'] for r in rows])))
 summary=aggregate(runs);cells=[]
 for field in dict.fromkeys(b['field'] for b in m['batches']):
  for level in range(5):
   subset=[r for r in runs if r['field']==field and r['level']==level]
   if subset:cells.append(dict(field=field,level=level,**aggregate(subset)))
 a.output.mkdir(parents=True,exist_ok=False);write_csv(a.output/'per-run.csv',runs)
 if windows:write_csv(a.output/'per-window-events.csv',windows)
 write_csv(a.output/'field-level.csv',cells)
 report=dict(schema='eatnav_pose2s_exploratory_selector_report_v1',evaluation_identity=identity,runtime_identity_sha256=sorted(runtime_ids),conditions=sorted({r['condition'] for r in runs}),planned_runs=len(expected),completed_runs=len(runs),strict_accuracy_achieved=False,real_robot_compatibility_verified=False,comparison_A_available=any(r['condition']=='A' for r in runs),summary=summary,candidate_selection_counts={str(k):v for k,v in sorted(selection.items(),key=lambda x:-1 if x[0] is None else x[0])},definitions=dict(route_progress='initial minus final XY goal distance, signed metres',filter_stop_seconds='duration of all14 candidates rejected by predicted tilt; NOT computation waiting or measured stall',linear_stall='complete2s window: integrated requested |v|>=.20m, travelled distance/request<=.25, canonical arc progress<=.05m',turn_stall='complete2s window: all commanded |w|>=.01rad/s and each post-step measured body |wz|<=.5*command',tip_over='measured gravity tilt>=85deg or tilt_over termination',time_domain='physics time; host pauses for actual FAST-LIO do not count as STOP seconds'),limits=['B-only has no A/B causal comparison; no claim that tilt filter improves navigation.','Global mean accuracy acceptable, per-condition maximum accuracy unmet.','Genuine FAST-LIO on ideal simulated sensors, not calibrated real sensor noise/latency.'])
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'source-index.json').write_text(json.dumps(sources,indent=2)+'\n')
 lines=['条件Bのみの探索的シミュレータ評価','',json.dumps(summary,ensure_ascii=False,indent=2),'','Aは未実施のため、傾斜除外の有無による改善効果は主張しません。','STOP秒は候補の全除外による物理時間です。ホスト計算待ちや実測スタックとは別です。']
 (a.output/'summary.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
