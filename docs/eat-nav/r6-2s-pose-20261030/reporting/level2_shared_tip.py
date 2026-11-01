from pathlib import Path
import json,hashlib,time
import numpy as np
from eat_nav_r6_storage import read_trace
from visualnav_transformer.eat_nav.fall_stall_targets import post_state,tilt
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');queue=r/'validation-evaluation';ledger=json.loads((queue/'progress.json').read_text());seed=1756069594
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[];states={}
for jid,record in ledger.items():
 if not jid.endswith('-k_rail_field-l2-batch00') or record['state']!='completed':continue
 assignment=read(queue/jid/'assignment.json'); indices=[i for i,g in enumerate(assignment['groups']) if g['seed']==seed];assert len(indices)==1
 i=indices[0];base=Path(record['source']) if record['execution']=='verified_policy_equivalent_reuse' else queue/jid/'sim-output';folder=base/f'group-{i:03d}'
 group=read(folder/'group.json');result=read(folder/'result.json');assert result['seed']==seed and result['initial_valid'];assert assignment['groups'][i]['group_id']==group['group_id']
 meta=read(base/'trace-storage.json')[f'{folder.name}/trace.jsonl'];assert sha(base/meta['compressed_path'])==meta['compressed_sha256']
 rows=read_trace(folder/'trace.jsonl');assert len(rows)==round(result['observed_sec']/.05)
 initial=np.asarray(rows[0]['root_state'][:3]);post=np.asarray([post_state(x) for x in rows]);positions=np.vstack([initial,post[:,:3]]);deltas=np.diff(positions[:,:2],axis=0)
 physical=[{k:row[k] for k in ('root_state','post_root_state','joint_position','joint_velocity','applied_torque','actor_command','follower_command')} for row in rows]
 physicalsha=hashlib.sha256(json.dumps(physical,sort_keys=True,separators=(',',':')).encode()).hexdigest();states[jid]=physicalsha
 decisions=[read(p) for p in sorted(folder.glob('decision-*.json'))]
 angles=np.asarray([tilt(x.tolist()) for x in post]);commands=np.asarray([row['actor_command'] for row in rows]);torques=np.asarray([row['applied_torque'] for row in rows]);stop=sum(bool(x['protective_stop']) for x in rows)*.05
 assert abs(stop-result['protective_stop_sec'])<1e-9
 records.append(dict(job=jid,execution=record['execution'],source_job=base.parent.name,group_id=group['group_id'],observed_rows=len(rows),observed_sec=result['observed_sec'],tip_count=result['tip_count'],STOP_request_sec=stop,all_actor_commands_zero=bool(np.all(commands==0)),maximum_abs_actor_command=float(abs(commands).max()),rows_with_nonzero_joint_torque=int(np.any(abs(torques)>1e-6,axis=1).sum()),physical_planar_distance_m=float(np.linalg.norm(deltas,axis=1).sum()),planar_endpoint_displacement_m=float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),initial_gravity_tilt_deg=float(tilt(rows[0]['root_state'])),terminal_gravity_tilt_deg=float(angles[-1]),max_observed_gravity_tilt_deg=float(angles.max()),route_progress_m=result['route_progress_m'],goal_reached=result['goal_reached'],residual_unavailable_sec=result['residual_unavailable_sec'],decisions=len(decisions),STOP_decisions=sum(x['selected_index'] is None for x in decisions),physical_and_command_rows_sha256=physicalsha,compressed_trace_sha256=meta['compressed_sha256'],result_sha256=sha(folder/'result.json')))
ref='pose_q3_0-k_rail_field-l2-batch00';identical=[x['job'] for x in records if states[x['job']]==states[ref]]
assert len(identical)>=7
report=dict(schema='eatnav_r6_level2_shared_tip_descriptive_audit_v1',verified_unix=time.time(),parent_seed=seed,records=records,identical_physical_command_trajectories=identical,reference_job=ref,script_sha256=sha(Path(__file__)),limitations=['Incomplete validation: no threshold ranking or safety conclusion','Same parent seed shared across conditions; reused records are not new independent physical trials','STOP is a zero requested command to the unchanged low-level RL controller, not torque disabling or a physical brake','Observed association does not prove STOP caused the fall or identify its mechanical cause','No runtime/model/threshold changes; no test or bag reads'])
text='# K-Rail level2: 同一seedのSTOP要求中転倒（途中診断）\n\n'
text+='親seed1756069594では、7設定で同一の物理状態・関節状態・指令の時系列が記録され、24.95秒で85度基準の転倒が観測された。うち2設定は方策一致による再利用であり、7件の独立した事故ではない。\n\n'
text+='|設定|実行区分|転倒|観測秒|STOP要求秒|実移動距離m|route進捗m|終端傾斜deg|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for x in records:text+=f"|{x['job'].split('-k_rail')[0]}|{x['execution']}|{x['tip_count']}|{x['observed_sec']:.2f}|{x['STOP_request_sec']:.2f}|{x['physical_planar_distance_m']:.3f}|{x['route_progress_m']:.3f}|{x['terminal_gravity_tilt_deg']:.2f}|\n"
x=next(x for x in records if x['job']==ref)
text+=f"\n基準q3_0ではActorへの指令が全{ x['observed_rows']}行でゼロでも、全行で関節トルクが非ゼロで、平面移動距離は{x['physical_planar_distance_m']:.3f}mだった。低レベルRLは動作しているため、STOP秒を物理的静止時間とは解釈できない。終端はreset前の状態を使用した。\n\n"
text+='同一初期条件の旧参照・イベント方式では本seedに転倒は観測されなかったが、これは単一seedの記述であり、停止が転倒を起こしたという因果関係や全体の優位性を示さない。他のseed・field/levelを含む事前固定比較を継続する。転倒後の35.05秒の残余は順位用STOP上限に保持する。\n'
for base in [r/'validation-progress',Path('docs/eat-nav/r6-2s-pose-20261030/validation-progress')]:
 for name,contents in [('level2-shared-STOP-tip-diagnostic.json',json.dumps(report,indent=2)+'\n'),('level2-shared-STOP-tip-diagnostic.md',text),('level2-shared-STOP-tip-diagnostic.txt',text)]:
  dest=base/name;assert not dest.exists();dest.write_text(contents)
print(json.dumps(dict(records=len(records),identical=identical,reference=x),indent=2))
