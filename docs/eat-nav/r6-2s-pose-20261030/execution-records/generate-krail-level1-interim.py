from pathlib import Path
import json,time,hashlib
from visualnav_transformer.eat_nav.r6_policy import wilson
from eat_nav_r6_eval_queue import verify_initial_cohort
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');q=r/'validation-evaluation';w=Path.cwd();ledger=json.loads((q/'progress.json').read_text());binding=json.loads((q/'frozen-binding.json').read_text());rows=[];sources={};group_sets=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for c in binding['conditions']:
 jid=c['id']+'-k_rail_field-l1-batch00';rec=ledger[jid];assert rec['state']=='completed' and rec['execution']=='simulated';base=q/jid/'sim-output';complete=json.loads((base/'complete.json').read_text());results=complete['results'];valid=[x for x in results if x['initial_valid']];assert len(results)==len(valid)==30 and all(x['full_evaluation_valid'] for x in valid);assert results==rec['results'];group_sets.append({x['group_id'] for x in valid})
 if c['id']!='old_reference':verify_initial_cohort(q/'old_reference-k_rail_field-l1-batch00/sim-output',base)
 n=len(valid);mean=lambda key:sum(x[key] for x in valid)/n;tips=sum(x['tip_count']>0 for x in valid);unknown=sum(x['fall_outcome_unknown'] for x in valid);assert unknown==0
 rows.append(dict(condition=c['id'],mode=c['mode'],profile=c.get('profile'),trials=n,tip_count=tips,tip_rate_wilson95=wilson(tips,n),near_fall_count=sum(x['near_fall_count'] for x in valid),mean_STOP_request_sec=mean('protective_stop_sec'),mean_ranking_STOP_upper_bound_sec=mean('ranking_STOP_upper_bound_sec'),mean_route_progress_m=mean('route_progress_m'),mean_net_goal_approach_m=mean('net_goal_approach_m'),mean_physical_distance_m=mean('physical_distance_m'),goal_reached_count=sum(x['goal_reached'] for x in valid),linear_windows=sum(x['linear_windows'] for x in valid),turn_windows=sum(x['turn_windows'] for x in valid),mean_linear_stall_sec=mean('linear_stall_sec'),mean_turn_stall_sec=mean('turn_stall_sec'),timeout_count=sum(x['timeout'] for x in valid),intervention_count=sum(x['intervention'] for x in valid),unknown_stall_windows=sum(x['unknown_stall_windows'] for x in valid),mean_processing_sec=mean('processing_sec'),all_candidates_excluded_decisions=sum(x['all_candidates_excluded_decisions'] for x in valid),decision_count=sum(x['decision_count'] for x in valid)))
 sources[jid]=sha(base/'complete.json')
assert len(rows)==13 and all(x==group_sets[0] for x in group_sets) and len(group_sets[0])==30
assert sum(x['mode']=='pose' for x in rows)==10
for x in rows:assert abs(x['mean_linear_stall_sec']-2*x['linear_windows']/x['trials'])<1e-9 and abs(x['mean_turn_stall_sec']-2*x['turn_windows']/x['trials'])<1e-9
report=dict(schema='eatnav_r6_completed_cell_13_condition_interim_v1',interim=True,observed_unix=time.time(),field='k_rail_field',level=1,complete_conditions=13,planned_conditions=13,common_initial_groups=30,unique_group_ids=sorted(group_sets[0]),rows=rows,complete_source_sha256=sources,binding_sha256=sha(q/'frozen-binding.json'),generator_sha256=sha(Path(__file__)),completed_cells=2,planned_cells=35,full_goal_complete=False,limitations=['390 condition-trials share30parents; not390independentterrains','no top3 freeze or final ranking from partial study','test and offline bag model evaluation not performed','STOP is request time, not physical static; zero events under STOP is not driving success'])
text=['# K-Rail level1・13条件の途中比較','全35field/levelのうち、K-Rail level0とlevel1の2cellが完了しました。本表はlevel1のみです。同じ30初期条件を13条件で共有しており、390種類の独立地形ではありません。他の33cell・test/bagは未完了です。本表から上位3設定を選定しません。','| 条件 | 有効走行 | 転倒走行 | 転倒率95%区間 | 平均STOP要求秒 | 順位用STOP上限秒 | 平均route進捗m | 到達走行 | 直線window計 | 旋回window計 | 直線秒/走行 | 旋回秒/走行 |\n|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
lines=[]
for x in rows:
 lo,hi=x['tip_rate_wilson95'];lines.append(f"| {x['condition']} | {x['trials']} | {x['tip_count']} | {lo*100:.1f}–{hi*100:.1f}% | {x['mean_STOP_request_sec']:.2f} | {x['mean_ranking_STOP_upper_bound_sec']:.2f} | {x['mean_route_progress_m']:.3f} | {x['goal_reached_count']} | {x['linear_windows']} | {x['turn_windows']} | {x['mean_linear_stall_sec']:.2f} | {x['mean_turn_stall_sec']:.2f} |")
text[-1]+='\n'+'\n'.join(lines)
text+=['STOPは要求時間です。指令遅延と固定低レベルRLがあるため、Actor入力ゼロ・モータ停止・物理静止と同義ではありません。ほぼ全時間STOPの条件で転倒やスタックが0でも走破性能の良さを示しません。転倒等で失った残余は順位用STOP上限へ加えます。スタックは非重複2秒window、陽性秒=window数×2であり、独立事故件数ではありません。','転倒0/30でもWilson95%上限は約11.4%です。0/30と1/30の違いを有意な改善や安全性の確定としません。route進捗は固定goal線への射影で、旧R4/R5のwaypoint更新型指標と直接比較できません。','## 移動・未達などの診断','| 条件 | 平均移動距離m | 平均goal正味接近m | near-fall計 | timeout走行 | 介入走行 | unknown stall window | 全候補除外/判定数 | 平均処理秒 |\n|---|---:|---:|---:|---:|---:|---:|---|---:|']
for x in rows:text[-1]+=f"\n| {x['condition']} | {x['mean_physical_distance_m']:.3f} | {x['mean_net_goal_approach_m']:.3f} | {x['near_fall_count']} | {x['timeout_count']} | {x['intervention_count']} | {x['unknown_stall_windows']} | {x['all_candidates_excluded_decisions']}/{x['decision_count']} | {x['mean_processing_sec']:.3f} |"
text+=['処理秒は保存された試行ごとのprocessing_secの平均です。全候補除外の分母はdecision数であり、走行数ではありません。','## 実走行済みの10姿勢閾値','| 設定 | pitch度 | roll度 | 基本段差cm | 階段cm | 壁cm | 角度マージン度 |\n|---|---:|---:|---:|---:|---:|---:|']
for x in rows:
 if x['mode']=='pose':
  p=x['profile'];text[-1]+=f"\n| {x['condition']} | {p['pitch_deg']:.6f} | {p['roll_deg']:.6f} | {p['step_cm']:.6f} | {p['conditional_step_cm']['stairs']:.6f} | {p['conditional_step_cm']['wall']:.6f} | {p['tilt_margin_deg']:.6f} |"
text+=['閾値はvalidation由来の候補であり、短時間event-freeを2m走破保証としません。階段/壁の形状条件はthreshold-method.mdと凍結profileを参照してください。正確な全profile・complete.jsonのSHA・共通group ID・binding SHAはkrail-level1-13-condition-interim.jsonに保存しています。モデル・閾値・cohortは変更していません。','このcellでは、旧参照は転倒0・STOP0秒・route1.852m、新2秒イベント方式は転倒0・STOP0.133秒・route1.701mでした。姿勢閾値や併用を加えた方式で、旧参照より走破性能が改善したとは本表から言えません。全35cellの結果を待ち、事前固定した順位規則で評価します。']
for folder in [r/'validation-progress',w/'docs/eat-nav/r6-2s-pose-20261030/validation-progress']:
 for suffix,data in [('json',json.dumps(report,indent=2)+'\n'),('md','\n\n'.join(text)+'\n'),('txt','\n\n'.join(text)+'\n')]:
  p=folder/('krail-level1-13-condition-interim.'+suffix);assert not p.exists();p.write_text(data)
print(json.dumps({'conditions':len(rows),'common_valid_groups':30,'tips':sum(x['tip_count'] for x in rows),'linear_windows':sum(x['linear_windows'] for x in rows),'turn_windows':sum(x['turn_windows'] for x in rows)}))
