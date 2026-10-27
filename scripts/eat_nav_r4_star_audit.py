"""R4* six-candidate saved-input inference and full guard-state replay; immutable baselines."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from eat_nav_all_fields import compare_anchor
from eat_nav_all_fields_run import atomic
from eat_nav_all_fields_learning import npz,teacher_rows
from eat_nav_command_free_learning import load_checkpoint,predict
from eat_nav_krail_ablation import window_labels,exposure_partition
from eat_nav_r4_star import R4StarGuard as R4Guard,CANDIDATE_IDS
from eat_nav_r4_star import candidate_paths
from visualnav_transformer.eat_nav.fall_stall_targets import tilt
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,verify_derived,require

def comparison_eligible(audit,pair):
    return bool(audit.get('eligible') and not audit.get('replay_errors') and pair.get('matched'))

def observed_windows(rows):
    # Keep the censored final cycle partial; never turn its crossing into a stall.
    safe=teacher_rows(rows)
    return [safe[i:i+40] for i in range(0,len(safe),40)]

def prediction_replay_errors(actual,selected,raw):
    errors=[]
    if not np.allclose(actual['probabilities'],selected['probabilities'],rtol=0,atol=1e-5):errors.append('risk_inference_mismatch')
    if not np.allclose(np.asarray(actual['raw_probabilities'],dtype=float),raw,rtol=0,atol=1e-5,equal_nan=True):errors.append('raw_risk_inference_mismatch')
    return errors

def audit_evaluation(folder,model):
    torch.set_num_threads(1)
    folder=Path(folder);verify_derived(folder);out=folder/'sim-output';result=read(folder/'result.json')
    rows=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
    if not rows:return dict(eligible=False,reason='no_control_steps',result=result)
    condition=result['condition'];guard=R4Guard()
    require(condition=='R4*','R4* audit only')
    net,payload=load_checkpoint(model);fit=dict(payload,model=net);errors=[];max_error=0.;decisions=0
    for row in rows:
        step=row['control_step']
        if guard.snapshot()!=row['guard_state_before_step']:errors.append(dict(step=step,reason='guard_before_state_mismatch'))
        guard.observe_tilt(tilt(row['root_state']),step)
        if step%40==0:
            record=read(out/f'selection-{step:04d}.json');arrays=npz(out/record['input_file'])
            p,q=predict(fit,arrays,[0]);p=p[0];q=q[0]
            common=dict(arrays)
            for name in ('paths','directions','candidate_mask'):common[name]=arrays[name][:,:4]
            p4,q4=predict(fit,common,[0])
            if not np.allclose(p[:4],p4[0],rtol=0,atol=1e-5) or not np.allclose(q[:4],q4[0],rtol=0,atol=1e-5):errors.append(dict(step=step,reason='common_four_inference_mismatch'))
            selected=guard.decide(p,q,record['goal_body'],step,tilt(row['root_state']))
            actual=record['selection'];decisions+=1
            error=float(np.max(np.abs(np.asarray(actual['progress_m'])-q)));max_error=max(max_error,error)
            if selected['selected_index']!=actual['selected_index']:errors.append(dict(step=step,reason='selected_candidate_mismatch'))
            errors.extend(dict(step=step,reason=reason) for reason in prediction_replay_errors(actual,selected,p))
        if guard.snapshot()!=row['guard_state_after_step']:errors.append(dict(step=step,reason='guard_after_state_mismatch'))
        if guard.stopped!=row['guard_stopped']:errors.append(dict(step=step,reason='guard_state_mismatch'))
        command=np.asarray(row['follower_command'])
        if row['guard_stopped'] and np.any(command!=0):errors.append(dict(step=step,reason='nonzero_stop_command'))
    require(max_error<=1e-5,'progress inference replay mismatch')
    labels=[]
    for chunk in observed_windows(rows):
        adapted=[]
        for r in chunk:
            v=dict(r,selected_candidate=r['candidate_id'])
            if r['terminal']:v['terminal']=r['terminal']['pre_reset_state']
            adapted.append(v)
        first=adapted[0];candidate=first['selected_candidate']
        k=CANDIDATE_IDS.index(candidate) if candidate in CANDIDATE_IDS else 0
        labels.append(window_labels(adapted,candidate_paths()[k].tolist()))
    high=0;near=False;pending=0;tip=False
    for row in rows:
        if row.get('boundary_censor'):break
        t=tilt(row['post_root_state']);high=high+1 if t>=60 else 0;pending=high
        near=near or high>=5
        terminal=(row.get('terminal') or {}).get('pre_reset_state') or {}
        tip=tip or t>=85 or 'tilt_over' in terminal.get('termination_reasons',[])
    reason=result['stop_reason'];eligible=reason in ('horizon_complete','natural_termination','boundary_censor') and not errors
    if reason=='horizon_complete' and len(rows)!=1200:eligible=False
    return dict(candidate_decisions={name:sum(read(out/f'selection-{step:04d}.json')['selection']['selected_candidate']==name for step in range(0,len(rows),40)) for name in (*CANDIDATE_IDS,'STOP')},candidate_steps={name:sum(r['candidate_id']==name for r in rows) for name in (*CANDIDATE_IDS,'STOP')},eligible=eligible,result=result,inference_decisions=decisions,replay_errors=errors,max_progress_replay_error=max_error,
        linear_positive_windows=sum(v['linear'] is True for v in labels),turn_positive_windows=sum(v['turn'] is True for v in labels),
        linear_exposure=exposure_partition(teacher_rows(rows),[v['linear'] for v in labels]),turn_exposure=exposure_partition(teacher_rows(rows),[v['turn'] for v in labels]),
        tip_over_observed=tip,near_fall_observed=near,pending_high_tilt_samples=pending,
        physical_distance_m=sum(float(np.linalg.norm(np.asarray(r['post_root_state'][:2])-r['root_state'][:2])) for r in rows))

def report(root,results,baseline_root):
    root=Path(root);baseline_root=Path(baseline_root)
    prior=read(baseline_root/'suite-results.json');manifest=read(root/'manifest.json')
    def key(e):
        a=e['assignment'];return a['field'],a['seed'],a['model_seed']
    matched={key(e) for e in results if e.get('comparison_eligible')}
    aggregates=[]
    lines=['# R4 / R4*: 後退曲線2候補だけを追加した比較','',
        f"予定{len(manifest['runs'])}、処理済み{len(results)}。再学習なし。R4*は先頭4候補を保持し、後退−30度/＋30度だけを追加。",'',
        '| field | 条件 | 有効case数 | 走行seed数 | 転倒観測走行 | 平均進捗m | 平均STOP秒 | 平均観測秒 | 正常waypoint | linear陽性window | 評価可能走行秒 | 未知走行秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for field in manifest['fields']:
        for mode,source in [('R4',prior),('R4*',results)]:
            xs=[e for e in source if e['assignment']['field']==field and e.get('comparison_eligible') and key(e) in matched]
            if not xs:continue
            a=dict(field=field,condition=mode,runs=len(xs),rollout_seeds=len({e['assignment']['seed'] for e in xs}),
                tip_over_runs=sum(e['audit']['tip_over_observed'] for e in xs),mean_progress_m=float(np.mean([e['result']['route_progress_m'] for e in xs])),
                mean_stop_sec=float(np.mean([e['result']['stop_steps']/20 for e in xs])),mean_observed_sec=float(np.mean([e['result']['control_steps']/20 for e in xs])),
                normal_waypoints=sum(e['result']['normal_waypoints'] for e in xs),linear_positive_windows=sum(e['audit']['linear_positive_windows'] for e in xs),
                linear_evaluable_sec=sum(e['audit']['linear_exposure']['evaluable_running_steps'] for e in xs)/20,
                linear_unknown_running_sec=sum(e['audit']['linear_exposure']['unknown_running_steps'] for e in xs)/20)
            aggregates.append(a)
            lines.append(f"| {field} | {mode} | {a['runs']} | {a['rollout_seeds']} | {a['tip_over_runs']} | {a['mean_progress_m']:.3f} | {a['mean_stop_sec']:.2f} | {a['mean_observed_sec']:.2f} | {a['normal_waypoints']} | {a['linear_positive_windows']} | {a['linear_evaluable_sec']:.2f} | {a['linear_unknown_running_sec']:.2f} |")
    lines+=['','## R4*候補別の実行時間','',
        '| field | 候補 | 指令採用step | 採用時間秒 | decision選択回数 |','|---|---|---:|---:|---:|']
    for field in manifest['fields']:
        xs=[e for e in results if e['assignment']['field']==field and e.get('comparison_eligible')]
        if not xs:continue
        for name in (*CANDIDATE_IDS,'STOP'):
            steps=sum(e['audit'].get('candidate_steps',{}).get(name,0) for e in xs)
            decisions=sum(e['audit'].get('candidate_decisions',{}).get(name,0) for e in xs)
            lines.append(f'| {field} | {name} | {steps} | {steps/20:.2f} | {decisions} |')
    lines+=['','## 全予定の処理結果','',
        '| field | 学習seed | 走行seed | 状態 | 対応比較 | 理由 |','|---|---:|---:|---|---|---|']
    for e in results:
        a=e['assignment'];err=e.get('error',{});reason=(err.get('simulator') or {}).get('message',err.get('message',e.get('result',{}).get('stop_reason','')))
        lines.append(f"| {a['field']} | {a['model_seed']} | {a['seed']} | {e['state']} | {e.get('comparison_eligible',False)} | {str(reason).replace('|','/')} |")
    lines+=['','候補別採用stepはSTOPを除いた実際の候補ID、decision数は2秒再評価時の最終選択。候補継続と新規選択を混同しない。',
        '対応するR4/R4*の初期状態・共通入力一致と再現監査が成立した集合だけで比較する。集計途中は結論を出さない。',
        '追加後退曲線は現行4候補datasetで直接学習していない。同じweightsによる候補拡張であり追加学習の効果ではない。',
        '転倒・STOP・進捗・観測時間を併記する。早期転倒でSTOP秒が短くなる可能性がある。linear陽性windowは独立事故件数ではない。',
        '走行seed反復、既存K-Rail validation不成立、初期PhysX警告とsettling失敗の制約を引き継ぐ。reserved test未開封。']
    (root/'comparison-report.md').write_text('\n'.join(lines)+'\n');atomic(root/'aggregates.json',aggregates)
