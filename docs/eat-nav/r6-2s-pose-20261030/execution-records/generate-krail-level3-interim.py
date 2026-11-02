from pathlib import Path
import json,time,hashlib
from visualnav_transformer.eat_nav.r6_policy import wilson
from eat_nav_r6_eval_queue import verify_initial_cohort
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');q=r/'validation-evaluation';w=Path.cwd();ledger=json.loads((q/'progress.json').read_text());binding=json.loads((q/'frozen-binding.json').read_text());rows=[];sources={};origins={};group_sets=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for c in binding['conditions']:
 jid=c['id']+'-k_rail_field-l3-batch00';rec=ledger[jid];assert rec['state']=='completed'
 assert rec['execution'] in ('simulated','verified_policy_equivalent_reuse')
 directory=q/jid;assignment=json.loads((directory/'assignment.json').read_text())
 base=Path(rec['source']) if rec['execution']=='verified_policy_equivalent_reuse' else directory/'sim-output'
 complete=json.loads((base/'complete.json').read_text());results=complete['results'];valid=[x for x in results if x['initial_valid']]
 assert len(results)==30 and len(valid)==29 and all(x['full_evaluation_valid'] for x in valid)
 invalid=[x for x in results if not x['initial_valid']]
 assert len(invalid)==1 and invalid[0]['group_id']=='closed_loop/k_rail_field/level3/validation/seed1558235351/tile00' and invalid[0]['observed_sec']==0 and not invalid[0]['full_evaluation_valid'] and invalid[0]['fall_outcome_unknown'] and not invalid[0]['goal_reached'] and invalid[0]['tip_count']==0
 if rec['execution']=='simulated':
  assert results==rec['results']
  audit=json.loads((r/'validation-progress'/('audit-'+jid+'.json')).read_text())
  assert audit['copy_sha_verified'] and audit['compressed_and_raw_sha_verified'] and audit['complete_sha256']==sha(base/'complete.json')
 else:
  proof=json.loads((directory/'equivalence-audit.json').read_text())
  assert proof['source_manifest_sha256']==sha(base/'manifest.json') and len(proof['checks'])==30 and all(x['equivalent'] for x in proof['checks'])
  assert assignment['groups']==json.loads((base.parent/'assignment.json').read_text())['groups']
  audit=json.loads((r/'validation-progress'/('masked-reuse-audit-'+jid+'.json')).read_text())
  assert audit['all_valid_selected_indices_identical'] and audit['all_saved_results_identical'] and audit['initial_invalid_has_zero_trace_and_decisions'] and audit['initial_invalid_not_credited_as_success'] and audit['equivalence_proof_sha256']==sha(directory/'equivalence-audit.json')
  assert audit['initial_valid_environments']==29 and audit['initial_invalid_environments']==1 and not audit['independent_simulation'] and not audit['independent_timing']
  assert audit['invalid_group_ids']==[invalid[0]['group_id']]
  expected=[dict(x,mode=c['mode'],execution='verified_policy_equivalent_reuse',timing_origin='source actual full inference; not independently timed',source=str(base/f'group-{i:03d}')) for i,x in enumerate(results)]
  assert rec['results']==expected
 group_sets.append({x['group_id'] for x in valid})
 origins[jid]=dict(execution=rec['execution'],source_job=base.parent.name,independent_timing=rec['execution']=='simulated',audit_sha256=sha(r/'validation-progress'/(('audit-' if rec['execution']=='simulated' else 'masked-reuse-audit-')+jid+'.json')))
 if c['id']!='old_reference':verify_initial_cohort(q/'old_reference-k_rail_field-l3-batch00/sim-output',base)
 n=len(valid);mean=lambda key:sum(x[key] for x in valid)/n;tips=sum(x['tip_count']>0 for x in valid);unknown=sum(x['fall_outcome_unknown'] for x in valid);assert unknown==0
 rows.append(dict(condition=c['id'],execution=rec['execution'],source_job=base.parent.name,independent_timing=rec['execution']=='simulated',mode=c['mode'],profile=c.get('profile'),trials=n,tip_count=tips,tip_rate_wilson95=wilson(tips,n),near_fall_count=sum(x['near_fall_count'] for x in valid),mean_STOP_request_sec=mean('protective_stop_sec'),mean_ranking_STOP_upper_bound_sec=mean('ranking_STOP_upper_bound_sec'),mean_route_progress_m=mean('route_progress_m'),mean_net_goal_approach_m=mean('net_goal_approach_m'),mean_physical_distance_m=mean('physical_distance_m'),goal_reached_count=sum(x['goal_reached'] for x in valid),linear_windows=sum(x['linear_windows'] for x in valid),turn_windows=sum(x['turn_windows'] for x in valid),mean_linear_stall_sec=mean('linear_stall_sec'),mean_turn_stall_sec=mean('turn_stall_sec'),timeout_count=sum(x['timeout'] for x in valid),intervention_count=sum(x['intervention'] for x in valid),unknown_stall_windows=sum(x['unknown_stall_windows'] for x in valid),mean_processing_sec=mean('processing_sec'),all_candidates_excluded_decisions=sum(x['all_candidates_excluded_decisions'] for x in valid),decision_count=sum(x['decision_count'] for x in valid)))
 sources[jid]=sha(base/'complete.json')
assert len(rows)==13 and all(x==group_sets[0] for x in group_sets) and len(group_sets[0])==29
assert sum(x['mode']=='pose' for x in rows)==10
assert sum(x['execution']=='simulated' for x in rows)==9
assert sum(x['execution']=='verified_policy_equivalent_reuse' for x in rows)==4
assert invalid[0]['group_id']=='closed_loop/k_rail_field/level3/validation/seed1558235351/tile00'
for x in rows:assert abs(x['mean_linear_stall_sec']-2*x['linear_windows']/x['trials'])<1e-9 and abs(x['mean_turn_stall_sec']-2*x['turn_windows']/x['trials'])<1e-9
report=dict(schema='eatnav_r6_completed_cell_13_condition_interim_v1',interim=True,observed_unix=time.time(),field='k_rail_field',level=3,complete_conditions=13,planned_conditions=13,protocol_groups=30,common_initial_groups=29,initial_invalid_group_ids=[invalid[0]['group_id']],unique_group_ids=sorted(group_sets[0]),rows=rows,complete_source_sha256=sources,source_origins=origins,execution_counts={kind:sum(x['execution']==kind for x in rows) for kind in ('simulated','verified_policy_equivalent_reuse')},binding_sha256=sha(q/'frozen-binding.json'),generator_sha256=sha(Path(__file__)),completed_cells=4,planned_cells=35,full_goal_complete=False,limitations=['390 protocol condition records share30parents;377valid condition results from261actual simulated valid trials plus116exact policy-equivalent valid results;13initial-invalid records all refer to one excluded parent; not377independent simulations/terrains','no top3 freeze or final ranking from partial study','test and offline bag model evaluation not performed','STOP is request time, not physical static; zero events under STOP is not driving success'])
text=['# K-Rail level3・13条件の途中比較','全35field/levelのうち、K-Rail level0～3の4cellが完了しました。本表はlevel3のみです。各条件30走行を計画し、全13条件で同じ1初期環境が無効となったため、共通の初期有効29走行で比較します。9条件は実シミュレーション（有効261走行）、4条件は全選択一致による再利用（有効116条件結果）です。377有効条件結果は29親環境を共有し、377回の独立シミュレーションや377種類の独立地形ではありません。初期無効13記録は同一1環境の反復で、policyの成功・失敗・転倒なしの陰性例として数えません。他の31cell・test/bagは未完了です。本表から上位3設定を選定しません。','| 条件 | 実行区分 | 有効走行 | 転倒走行 | 転倒率95%区間 | 平均STOP要求秒 | 順位用STOP上限秒 | 平均route進捗m | 到達走行 | 直線window計 | 旋回window計 | 直線秒/走行 | 旋回秒/走行 |\n|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
lines=[]
for x in rows:
 lo,hi=x['tip_rate_wilson95'];lines.append(f"| {x['condition']} | {x['execution']} | {x['trials']} | {x['tip_count']} | {lo*100:.1f}–{hi*100:.1f}% | {x['mean_STOP_request_sec']:.2f} | {x['mean_ranking_STOP_upper_bound_sec']:.2f} | {x['mean_route_progress_m']:.3f} | {x['goal_reached_count']} | {x['linear_windows']} | {x['turn_windows']} | {x['mean_linear_stall_sec']:.2f} | {x['mean_turn_stall_sec']:.2f} |")
text[-1]+='\n'+'\n'.join(lines)
text+=['STOPは要求時間です。指令遅延と固定低レベルRLがあるため、Actor入力ゼロ・モータ停止・物理静止と同義ではありません。ほぼ全時間STOPの条件で転倒やスタックが0でも走破性能の良さを示しません。転倒等で失った残余は順位用STOP上限へ加えます。スタックは非重複2秒window、陽性秒=window数×2であり、独立事故件数ではありません。','転倒0/29でもWilson95%上限は約11.7%です。少数の転倒数の違いを有意な改善や安全性の確定としません。route進捗は固定goal線への射影で、旧R4/R5のwaypoint更新型指標と直接比較できません。','## 移動・未達などの診断','| 条件 | 平均移動距離m | 平均goal正味接近m | near-fall計 | timeout走行 | 介入走行 | unknown stall window | 全候補除外/判定数 | 平均処理秒 |\n|---|---:|---:|---:|---:|---:|---:|---|---:|']
for x in rows:text[-1]+=f"\n| {x['condition']} | {x['mean_physical_distance_m']:.3f} | {x['mean_net_goal_approach_m']:.3f} | {x['near_fall_count']} | {x['timeout_count']} | {x['intervention_count']} | {x['unknown_stall_windows']} | {x['all_candidates_excluded_decisions']}/{x['decision_count']} | {x['mean_processing_sec']:.3f} |"
text+=['処理秒は保存された試行ごとのprocessing_secの平均です。再利用条件は実行元q3_0の計時を継承しており、独立に計測した推論時間ではありません。全候補除外の分母はdecision数であり、走行数ではありません。','## 実走行・選択一致で照合した10姿勢閾値','| 設定 | pitch度 | roll度 | 基本段差cm | 階段cm | 壁cm | 角度マージン度 |\n|---|---:|---:|---:|---:|---:|---:|']
for x in rows:
 if x['mode']=='pose':
  p=x['profile'];text[-1]+=f"\n| {x['condition']} | {p['pitch_deg']:.6f} | {p['roll_deg']:.6f} | {p['step_cm']:.6f} | {p['conditional_step_cm']['stairs']:.6f} | {p['conditional_step_cm']['wall']:.6f} | {p['tilt_margin_deg']:.6f} |"
text+=['閾値はvalidation由来の候補であり、短時間event-freeを2m走破保証としません。階段/壁の形状条件はthreshold-method.mdと凍結profileを参照してください。正確な全profile・complete.jsonのSHA・共通group ID・binding SHAはkrail-level3-13-condition-interim.jsonに保存しています。モデル・閾値・cohortは変更していません。','このcellの結果だけで方式全体の優劣や安全性を確定しません。条件別の転倒、STOP、進捗、スタックと到達を合わせて参照し、全35cellの結果を待って事前固定した順位規則で評価します。q2_2・q2_0・q1_0・q0_0はq3_0の同30環境コホートについて、29初期有効環境の全観測決定で選択が一致し、1初期無効環境はtrace・決定0のまま除外される再利用です。追加の独立事故や独立計時として数えません。']
for folder in [r/'validation-progress',w/'docs/eat-nav/r6-2s-pose-20261030/validation-progress']:
 for suffix,data in [('json',json.dumps(report,indent=2)+'\n'),('md','\n\n'.join(text)+'\n'),('txt','\n\n'.join(text)+'\n')]:
  p=folder/('krail-level3-13-condition-interim.'+suffix);assert not p.exists();p.write_text(data)
print(json.dumps({'conditions':len(rows),'common_valid_groups':29,'tips':sum(x['tip_count'] for x in rows),'linear_windows':sum(x['linear_windows'] for x in rows),'turn_windows':sum(x['turn_windows'] for x in rows)}))
