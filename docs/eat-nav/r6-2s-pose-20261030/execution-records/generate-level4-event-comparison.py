from pathlib import Path
import json,hashlib,time
from visualnav_transformer.eat_nav.r6_policy import wilson
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');w=Path('docs/eat-nav/r6-2s-pose-20261030');q=r/'validation-evaluation'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ledger=read(q/'progress.json');binding=read(q/'frozen-binding.json');conditions={x['id']:x for x in binding['conditions']}
# No report is emitted before all six actual batches complete.
for name in ('old_reference','event_2s'):
 for i in range(3):
  rec=ledger[f'{name}-k_rail_field-l4-batch{i:02d}'];assert rec['state']=='completed' and rec['execution']=='simulated'
rows=[];batches=[];origins={};sets={};invalid_sets={}
for name in ('old_reference','event_2s'):
 all_results=[]
 for i in range(3):
  jid=f'{name}-k_rail_field-l4-batch{i:02d}';base=q/jid/'sim-output';results=read(base/'complete.json')['results'];assert len(results)==30 and results==ledger[jid]['results']
  assignment=read(base.parent/'assignment.json');assert assignment['condition']==conditions[name]
  ap=r/'validation-progress'/('audit-'+jid+'.json');audit=read(ap);assert audit['copy_sha_verified'] and audit['compressed_and_raw_sha_verified'] and audit['complete_sha256']==sha(base/'complete.json')
  if name=='old_reference':
   mp=r/'validation-progress'/('initial-invalid-old-reference-level4-batch00-audit.json' if i==0 else 'initial-invalid-'+jid+'-audit.json');mask=read(mp)
   assert mask['source_complete_sha256']==sha(base/'complete.json') and mask['source_manual_audit_sha256']==sha(ap)
   assert mask['invalid_trace_and_decisions_zero'] and mask['invalid_goal_false_tip_zero_fall_unknown_true'] and mask['all_initial_arrays_finite']
  else:
   mp=r/'validation-progress'/('paired-mask-'+jid+'.json');mask=read(mp);ref=q/f'old_reference-k_rail_field-l4-batch{i:02d}'/'sim-output'
   assert audit['paired_initial_and_physics_verified'] and mask['target_complete_sha256']==sha(base/'complete.json') and mask['source_complete_sha256']==sha(ref/'complete.json')
   assert mask['all31_initial_and_physics_files_byte_identical'] and mask['all_initial_masks_identical'] and mask['invalid_trace_and_decisions_zero'] and mask['invalid_goal_false_tip_zero_fall_unknown_true']
   for rel,h in mask['files_sha256'].items():assert sha(base/rel)==sha(ref/rel)==h
  valid=[x for x in results if x['initial_valid']];invalid=[x for x in results if not x['initial_valid']]
  assert {x['group_id'] for x in valid}==set(mask['valid_group_ids']) and {x['group_id'] for x in invalid}==set(mask['invalid_group_ids'])
  assert len(valid)==audit['initial_valid_trials'] and all(x['full_evaluation_valid'] and not x['fall_outcome_unknown'] for x in valid)
  for x in invalid:assert x['observed_sec']==0 and x['decision_count']==0 and not x['full_evaluation_valid'] and x['fall_outcome_unknown'] and not x['goal_reached'] and x['tip_count']==0 and x['linear_windows']==x['turn_windows']==0
  origins[jid]=dict(complete_sha256=sha(base/'complete.json'),manual_audit_sha256=sha(ap),mask_audit_sha256=sha(mp),execution='simulated')
  batches.append(dict(condition=name,batch=i,protocol_trials=30,initial_valid_trials=len(valid),initial_invalid_trials=len(invalid),tip_observed_trials=sum(x['tip_count']>0 for x in valid),goal_reached_trials=sum(x['goal_reached'] for x in valid),mean_route_progress_m=sum(x['route_progress_m'] for x in valid)/len(valid),observed_rows=audit['observed_rows']))
  all_results.extend(results)
 assert len(all_results)==90 and len({x['group_id'] for x in all_results})==90
 valid=[x for x in all_results if x['initial_valid']];invalid=[x for x in all_results if not x['initial_valid']];assert len(valid)==48 and len(invalid)==42
 sets[name]={x['group_id'] for x in valid};invalid_sets[name]={x['group_id'] for x in invalid};n=len(valid);mean=lambda k:sum(x[k] for x in valid)/n;tips=sum(x['tip_count']>0 for x in valid)
 row=dict(condition=name,protocol_trials=90,initial_valid_trials=n,initial_invalid_trials=42,tip_observed_trials=tips,tip_rate_wilson95=wilson(tips,n),near_fall_count=sum(x['near_fall_count'] for x in valid),goal_reached_trials=sum(x['goal_reached'] for x in valid),linear_windows=sum(x['linear_windows'] for x in valid),turn_windows=sum(x['turn_windows'] for x in valid),observed_rows=sum(x['observed_rows'] for x in batches if x['condition']==name),observed_seconds=sum(x['observed_sec'] for x in valid),residual_unavailable_seconds=sum(x['residual_unavailable_sec'] for x in valid),unknown_stall_windows=sum(x['unknown_stall_windows'] for x in valid),timeout_count=sum(x['timeout'] for x in valid),intervention_count=sum(x['intervention'] for x in valid))
 for key in ['protective_stop_sec','ranking_STOP_upper_bound_sec','route_progress_m','physical_distance_m','net_goal_approach_m','linear_stall_sec','turn_stall_sec','processing_sec']:row['mean_'+key]=mean(key)
 assert abs(row['mean_linear_stall_sec']-2*row['linear_windows']/n)<1e-9 and abs(row['mean_turn_stall_sec']-2*row['turn_windows']/n)<1e-9
 rows.append(row)
assert sets['old_reference']==sets['event_2s'] and invalid_sets['old_reference']==invalid_sets['event_2s']
report=dict(schema='eatnav_r6_level4_reference_event_three_batch_interim_v1',interim=True,verified_unix=time.time(),field='k_rail_field',level=4,role='validation',rows=rows,batches=batches,common_initial_valid_groups=sorted(sets['old_reference']),common_initial_invalid_groups=sorted(invalid_sets['old_reference']),source_sha256=origins,binding_sha256=sha(q/'frozen-binding.json'),generator_sha256=sha(Path(__file__)),paired_batches=3,full_cell_complete=False,completed_cells=4,planned_cells=35,top3_frozen=False,test_and_bag_evaluation_performed=False,full_goal_complete=False,limitations=['Each condition shares90planned environments with48common initial-valid and42excluded pre-policy invalid environments; not96independent terrains.','Pooled means use all48valid trials rather than a mean of3batch means.','Only2of13L4conditions completed; no global ranking or safety claim.','Wilson intervals are descriptive, not a paired significance test.','STOP request is not physical static; windows are nonoverlapping2seconds, not independent accidents.'])
text=['# K-Rail level4・旧参照と2秒イベント方式の途中比較','両方式とも事前固定3バッチ・90環境を実行し、同じ48初期有効走行で比較しました。42環境は走行前tilt_overで無効となり、policyの成功・失敗・転倒なしの陰性例に加算しません。対応する同一バッチの初期状態・物理ファイルと全初期maskを照合しています。96種類の独立地形ではありません。','| 条件 | 計画 | 有効 | 初期無効 | 転倒観測走行 | 転倒率95%区間 | 平均route進捗m | 平均STOP要求秒 | 順位用STOP上限秒 | 到達走行 | 直線window | 旋回window | 直線秒/走行 | 旋回秒/走行 |\n|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:
 lo,hi=x['tip_rate_wilson95'];text[-1]+=f"\n| {x['condition']} | 90 | 48 | 42 | {x['tip_observed_trials']} | {lo*100:.1f}–{hi*100:.1f}% | {x['mean_route_progress_m']:.3f} | {x['mean_protective_stop_sec']:.2f} | {x['mean_ranking_STOP_upper_bound_sec']:.2f} | {x['goal_reached_trials']} | {x['linear_windows']} | {x['turn_windows']} | {x['mean_linear_stall_sec']:.2f} | {x['mean_turn_stall_sec']:.2f} |"
text+=['全体平均は48有効走行から直接計算しています。Wilson区間は記述用であり、方式間の対応あり有意差検定ではありません。STOPは要求時間で、Actor入力ゼロ・モータ停止・物理静止と同義ではありません。転倒等で未観測になった残余は順位用STOP上限へ加えます。スタックは非重複2秒windowで、陽性秒=window数×2です。独立事故件数として数えません。','| 条件 | バッチ | 有効 | 初期無効 | 転倒 | 到達 | 平均route進捗m |\n|---|---:|---:|---:|---:|---:|---:|']
for x in batches:text[-1]+=f"\n| {x['condition']} | {x['batch']} | {x['initial_valid_trials']} | {x['initial_invalid_trials']} | {x['tip_observed_trials']} | {x['goal_reached_trials']} | {x['mean_route_progress_m']:.3f} |"
text+=['この表はレベル4の13条件中2条件だけの途中結果です。全35field/levelのうち完了はK-Rail level0～3の4cellで、残りの姿勢閾値・combinedのレベル4評価を継続しています。上位3設定の凍結・未使用test・実機bag評価は未実施です。この表だけで方式全体の優劣や安全性を結論しません。正確な値・追加診断・48有効/42無効group ID・各source/binding/generator SHAは同名JSONに保存しています。']
for root in [r,w]:
 for suffix,data in [('json',json.dumps(report,indent=2)+'\n'),('md','\n\n'.join(text)+'\n'),('txt','\n\n'.join(text)+'\n')]:
  out=root/'validation-progress'/('level4-reference-event-three-batch-interim.'+suffix);assert not out.exists();out.write_text(data)
print(json.dumps({'rows':rows,'paired_batches':3,'common_valid':48,'initial_invalid':42}))
