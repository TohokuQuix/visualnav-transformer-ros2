from pathlib import Path
import json,hashlib,time
from visualnav_transformer.eat_nav.r6_policy import wilson
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');w=Path('docs/eat-nav/r6-2s-pose-20261030');q=r/'validation-evaluation'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ledger=read(q/'progress.json');all_results=[];sources=[]
for i in range(3):
 jid=f'old_reference-k_rail_field-l4-batch{i:02d}';rec=ledger[jid]
 assert rec['state']=='completed' and rec['execution']=='simulated'
 base=q/jid/'sim-output';results=read(base/'complete.json')['results'];assert results==rec['results'] and len(results)==30
 ap=r/'validation-progress'/('audit-'+jid+'.json');a=read(ap)
 mp=r/'validation-progress'/('initial-invalid-old-reference-level4-batch00-audit.json' if i==0 else 'initial-invalid-'+jid+'-audit.json');m=read(mp)
 assert a['copy_sha_verified'] and a['compressed_and_raw_sha_verified'] and a['complete_sha256']==sha(base/'complete.json')
 assert m['source_complete_sha256']==a['complete_sha256'] and m['source_manual_audit_sha256']==sha(ap)
 assert m['all_initial_arrays_finite'] and m['invalid_trace_and_decisions_zero'] and m['invalid_goal_false_tip_zero_fall_unknown_true'] and m['invalid_not_policy_success_or_failure']
 valid=[x for x in results if x['initial_valid']];invalid=[x for x in results if not x['initial_valid']]
 assert len(valid)==a['initial_valid_trials']==m['initial_valid_trials'] and len(invalid)==m['initial_invalid_trials']
 assert {x['group_id'] for x in valid}==set(m['valid_group_ids']) and {x['group_id'] for x in invalid}==set(m['invalid_group_ids'])
 sources.append(dict(job=jid,planned=30,valid=len(valid),initial_invalid=len(invalid),tips=sum(x['tip_count']>0 for x in valid),linear_windows=sum(x['linear_windows'] for x in valid),turn_windows=sum(x['turn_windows'] for x in valid),goals=sum(x['goal_reached'] for x in valid),mean_route_progress_m=sum(x['route_progress_m'] for x in valid)/len(valid),complete_sha256=sha(base/'complete.json'),manual_audit_sha256=sha(ap),initial_mask_audit_sha256=sha(mp)))
 all_results.extend(results)
assert len({x['group_id'] for x in all_results})==90
valid=[x for x in all_results if x['initial_valid']];invalid=[x for x in all_results if not x['initial_valid']];assert len(valid)==48 and len(invalid)==42
n=len(valid);mean=lambda k:sum(x[k] for x in valid)/n;tips=sum(x['tip_count']>0 for x in valid)
report=dict(schema='eatnav_r6_preregistered_level4_reference_summary_v1',interim=True,verified_unix=time.time(),field='k_rail_field',level=4,condition='old_reference',planned_batches=3,completed_batches=3,protocol_trials=90,initial_valid_trials=n,initial_invalid_trials=len(invalid),valid_group_ids=[x['group_id'] for x in valid],invalid_group_ids=[x['group_id'] for x in invalid],tip_observed_trials=tips,tip_rate_wilson95=wilson(tips,n),near_fall_count=sum(x['near_fall_count'] for x in valid),linear_windows=sum(x['linear_windows'] for x in valid),turn_windows=sum(x['turn_windows'] for x in valid),mean_linear_stall_sec=mean('linear_stall_sec'),mean_turn_stall_sec=mean('turn_stall_sec'),mean_STOP_request_sec=mean('protective_stop_sec'),mean_ranking_STOP_upper_bound_sec=mean('ranking_STOP_upper_bound_sec'),mean_route_progress_m=mean('route_progress_m'),mean_physical_distance_m=mean('physical_distance_m'),goal_reached_trials=sum(x['goal_reached'] for x in valid),observed_rows=sum(a['observed_rows'] for a in [read(r/'validation-progress'/('audit-'+s['job']+'.json')) for s in sources]),observed_seconds=sum(x['observed_sec'] for x in valid),residual_unavailable_seconds=sum(x['residual_unavailable_sec'] for x in valid),source_batches=sources,generator_sha256=sha(Path(__file__)),binding_sha256=sha(q/'frozen-binding.json'),cross_condition_initial_pairing_verified=False,full_cell_complete=False,top3_frozen=False,test_and_bag_evaluation_performed=False,limitations=['Only preregistered old_reference batches; other L4 conditions pending.','42pre-policy settling-invalid environments are excluded, not policy successes/failures or negative tip labels.','Means are pooled over48valid trials, not an unweighted mean of the3batch means.','No cross-policy comparison or safety conclusion from this reference-only aggregate.','Windows are disjoint2seconds; not independent accident counts.'])
lo,hi=report['tip_rate_wilson95']
text=['# K-Rail level4・旧参照3固定バッチの途中集計','事前固定した3バッチ（90計画）を全て実行しました。初期有効は14＋15＋19＝48走行、初期無効は16＋15＋11＝42環境です。42環境は走行前tilt_overで、policyの成功・失敗や転倒なしの陰性例には数えません。他のレベル4条件は評価中であり、この表だけで方式間の優劣・安全性を結論しません。','| バッチ | 計画 | 有効 | 初期無効 | 転倒観測走行 | 平均route進捗m | 到達走行 | 直線window | 旋回window |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in sources:text[-1]+=f"\n| {x['job']} | 30 | {x['valid']} | {x['initial_invalid']} | {x['tips']} | {x['mean_route_progress_m']:.3f} | {x['goals']} | {x['linear_windows']} | {x['turn_windows']} |"
text+=[f"合計：有効{n}走行、転倒観測{tips}走行（Wilson95%区間{lo*100:.1f}～{hi*100:.1f}%）、near-fall {report['near_fall_count']}、平均route進捗{report['mean_route_progress_m']:.3f}m、到達{report['goal_reached_trials']}走行。平均STOP要求{report['mean_STOP_request_sec']:.2f}秒、残余を含む順位用STOP上限{report['mean_ranking_STOP_upper_bound_sec']:.2f}秒です。",f"直線スタック陽性{report['linear_windows']}window、旋回{report['turn_windows']}window、平均陽性秒は各{report['mean_linear_stall_sec']:.2f}秒／走行・{report['mean_turn_stall_sec']:.2f}秒／走行です。非重複2秒windowであり、独立した事故件数ではありません。",'全体平均は有効48走行から直接計算し、各バッチ平均の単純平均は使っていません。各バッチの原本・圧縮trace・初期maskを監査済みです。異なるseedバッチ同士の物理SHA一致は求めず、後続policyは対応する同一バッチの旧参照と初期・物理条件を照合します。上位3凍結・test/bag評価は未実施です。']
for root in [r,w]:
 for suffix,data in [('json',json.dumps(report,indent=2)+'\n'),('md','\n\n'.join(text)+'\n'),('txt','\n\n'.join(text)+'\n')]:
  out=root/'validation-progress'/('level4-reference-three-batch-interim.'+suffix);assert not out.exists();out.write_text(data)
print(json.dumps({k:report[k] for k in ['protocol_trials','initial_valid_trials','initial_invalid_trials','tip_observed_trials','mean_route_progress_m','goal_reached_trials','linear_windows','turn_windows','mean_ranking_STOP_upper_bound_sec']}))
