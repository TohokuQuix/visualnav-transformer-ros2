"""R4 saved-input inference and full guard-state replay; immutable baselines."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from eat_nav_all_fields import compare_anchor
from eat_nav_all_fields_learning import npz,teacher_rows
from eat_nav_command_free_learning import load_checkpoint,predict
from eat_nav_krail_ablation import window_labels,exposure_partition
from eat_nav_r4 import R4Guard
from eat_nav_risk_navigation import candidate_paths
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
    require(condition=='R4','R4 audit only')
    net,payload=load_checkpoint(model);fit=dict(payload,model=net);errors=[];max_error=0.;decisions=0
    for row in rows:
        step=row['control_step']
        if guard.snapshot()!=row['guard_state_before_step']:errors.append(dict(step=step,reason='guard_before_state_mismatch'))
        guard.observe_tilt(tilt(row['root_state']),step)
        if step%40==0:
            record=read(out/f'selection-{step:04d}.json');arrays=npz(out/record['input_file'])
            p,q=predict(fit,arrays,[0]);p=p[0];q=q[0]
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
        k={'forward_direct':0,'forward_-30':1,'forward_+30':2,'reverse_direct':3}.get(candidate,0)
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
    return dict(eligible=eligible,result=result,inference_decisions=decisions,replay_errors=errors,max_progress_replay_error=max_error,
        linear_positive_windows=sum(v['linear'] is True for v in labels),turn_positive_windows=sum(v['turn'] is True for v in labels),
        linear_exposure=exposure_partition(teacher_rows(rows),[v['linear'] for v in labels]),turn_exposure=exposure_partition(teacher_rows(rows),[v['turn'] for v in labels]),
        tip_over_observed=tip,near_fall_observed=near,pending_high_tilt_samples=pending,
        physical_distance_m=sum(float(np.linalg.norm(np.asarray(r['post_root_state'][:2])-r['root_state'][:2])) for r in rows))

def report(root,results,baseline_root):
    root=Path(root);baseline_root=Path(baseline_root)
    prior=read(baseline_root/'suite-results.json');manifest=read(root/'manifest.json');aggregates=[]
    lines=['# R4: 直線スタックのみ・実測姿勢STOPなし GUI比較','',
        'R4は既存モデルの直線スタック予測だけを採点・除外・予測STOP/再開へ使用する。転倒・near-fall・旋回risk、実測45度STOP/35度再開は無効。再学習・閾値調整は行わない。',
        '候補切替の4秒保持・0.15マージン・前後進2回確認、および予測STOP時の最短2秒・2回再開確認はR2/R3と共通。Actor・追従器・waypoint・最大60秒・自然85度terminationも共通。',
        f"予定{len(manifest['runs'])}走行、現在処理済み{len(results)}。完了数と有効比較数を区別し、settling failureや初期状態不一致も保持する。",'',
        '## 同じcase集合でのR1/R2/R3/R4比較','',
        '各fieldでR4と既存R1/R2/R3の開始状態が一致したcaseだけを集計。学習seedの反復は独立した地形seedではない。',
        '| field | 条件 | 有効走行数 | 走行seed数 | 転倒観測走行数 | 平均route進捗m | 平均STOP秒 | 正常waypoint合計 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def key(e):
        a=e['assignment'];return a['field'],a['seed'],a['model_seed']
    matched={key(e) for e in results if e.get('comparison_eligible')}
    for field in manifest['fields']:
        for cond in ['R1','R2','R3','R4']:
            data=[e for e in (results if cond=='R4' else prior) if e['assignment']['field']==field and e['assignment']['condition']==cond and e.get('comparison_eligible') and key(e) in matched]
            if not data:continue
            a=dict(field=field,condition=cond,runs=len(data),rollout_seeds=len({e['assignment']['seed'] for e in data}),
                tip_over_runs=sum(e['audit']['tip_over_observed'] for e in data),
                mean_progress_m=float(np.mean([e['result']['route_progress_m'] for e in data])),
                mean_stop_sec=float(np.mean([e['result']['stop_steps']/20 for e in data])),
                normal_waypoints=sum(e['result']['normal_waypoints'] for e in data),
                linear_positive_windows=sum(e['audit']['linear_positive_windows'] for e in data),
                linear_positive_sec=2.*sum(e['audit']['linear_positive_windows'] for e in data),
                linear_evaluable_sec=sum(e['audit']['linear_exposure']['evaluable_running_steps'] for e in data)/20,
                linear_unknown_running_sec=sum(e['audit']['linear_exposure']['unknown_running_steps'] for e in data)/20)
            aggregates.append(a)
            lines.append(f"| {field} | {cond} | {a['runs']} | {a['rollout_seeds']} | {a['tip_over_runs']} | {a['mean_progress_m']:.3f} | {a['mean_stop_sec']:.2f} | {a['normal_waypoints']} |")
    lines+=['','### 動作中の直線スタック観測','',
        'STOPを除き、同じ候補で40 interval観測できたwindowだけを評価。陽性windowは2秒単位で、独立事故数ではない。未知区間は陰性へ補完しない。',
        '| field | 条件 | 陽性window | 陽性占有秒 | 評価可能走行秒 | 未知走行秒 |','|---|---|---:|---:|---:|---:|']
    for a in aggregates:
        lines.append(f"| {a['field']} | {a['condition']} | {a['linear_positive_windows']} | {a['linear_positive_sec']:.2f} | {a['linear_evaluable_sec']:.2f} | {a['linear_unknown_running_sec']:.2f} |")
    lines+=['','## R4全予定の実測結果','',
        '| field | 学習seed | 走行seed | 時間s | route進捗m | STOP秒 | 最大傾斜度 | 転倒観測 | 終了・比較成立 |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for e in results:
        a=e['assignment'];r=e.get('result');audit=e.get('audit')
        if not r or not audit:
            detail=e.get('error',{});detail=(detail.get('simulator') or {}).get('message',detail.get('message',''))
            status=e['state']+(': '+detail if detail else '')
            status=status.replace('|','/')
            lines.append(f"| {a['field']} | {a['model_seed']} | {a['seed']} | — | — | — | — | — | {status} |")
        else:lines.append(f"| {a['field']} | {a['model_seed']} | {a['seed']} | {r['control_steps']/20:.2f} | {r['route_progress_m']:.3f} | {r['stop_steps']/20:.2f} | {r['max_tilt_deg']:.2f} | {audit['tip_over_observed']} | {r['stop_reason']} / {'比較可' if e.get('comparison_eligible') else 'N/A'} |")
    lines+=['','## 解釈上の条件','',
        '- R4とR2では旋回riskと姿勢STOPの2項目が変わるため、差を45度STOPの単独効果にしない。R1とも直線riskと姿勢STOPの2項目が異なる。',
        '- 正常waypointの45度未満という既存の到達基準は比較を揃えるため維持する。これは45度で速度指令を0にする保護とは別であり、R4で姿勢STOPは出さない。',
        '- STOP時間が少なくても、早期転倒で観測が短くなった場合がある。最終進捗・転倒・実時間・動作中stallを併記して判断する。',
        '- 同じmulti-head学習済みweightsの使用headだけを切り替える実験であり、転倒データを除外して学習し直したモデルではない。',
        '- 旧K-Rail専用validation不成立、走行seedの少なさ、PhysX mass/inertia初期化警告の制約は引き継ぐ。reserved testは未開封。',
        '- 本追加条件は既存R1/R2/R3の後日に実行する。image/source/geometry/inputの照合結果を保存するが、新しいランダム順同時実験ではない。',
        '- source/model/初期状態照合、保存入力からの全推論・guard状態再現、STOP全3軸0を監査する。不成立は比較N/Aとする。']
    (root/'comparison-report.md').write_text('\n'.join(lines)+'\n')
    (root/'aggregates.json').write_text(json.dumps(aggregates,indent=2,allow_nan=False)+'\n')
