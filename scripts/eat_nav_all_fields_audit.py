"""Saved-input inference/guard replay and descriptive all-field GUI outcomes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from eat_nav_all_fields import compare_anchor
from eat_nav_all_fields_learning import npz,teacher_rows
from eat_nav_command_free_learning import load_checkpoint,predict
from eat_nav_krail_ablation import AblationGuard,window_labels,exposure_partition
from eat_nav_risk_navigation import candidate_paths
from visualnav_transformer.eat_nav.fall_stall_targets import tilt
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,verify_derived,require

def comparison_eligible(audit,pair):
    return bool(audit.get('eligible') and not audit.get('replay_errors') and pair.get('matched'))

def observed_windows(rows):
    # Keep the censored final cycle partial; never turn its crossing into a stall.
    safe=teacher_rows(rows)
    return [safe[i:i+40] for i in range(0,len(safe),40)]

def audit_evaluation(folder,model):
    folder=Path(folder);verify_derived(folder);out=folder/'sim-output';result=read(folder/'result.json')
    rows=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
    if not rows:return dict(eligible=False,reason='no_control_steps',result=result)
    condition=result['condition'];guard=AblationGuard({'R0':'A2','R1':'A0','R2':'A1','R3':'A2'}[condition])
    net,payload=load_checkpoint(model);fit=dict(payload,model=net);errors=[];max_error=0.;decisions=0
    for row in rows:
        step=row['control_step'];guard.observe_tilt(tilt(row['root_state']),step)
        if step%40==0:
            record=read(out/f'selection-{step:04d}.json');arrays=npz(out/record['input_file'])
            p,q=predict(fit,arrays,[0]);p=p[0];q=q[0]
            if condition=='R1':p=np.zeros((4,4))
            elif condition=='R2':p[:,:2]=0
            selected=guard.decide(p,q,record['goal_body'],step,tilt(row['root_state']))
            actual=record['selection'];decisions+=1
            error=float(np.max(np.abs(np.asarray(actual['progress_m'])-q)));max_error=max(max_error,error)
            if selected['selected_index']!=actual['selected_index']:errors.append(dict(step=step,reason='selected_candidate_mismatch'))
            if not np.allclose(actual['probabilities'],p,rtol=0,atol=1e-5):errors.append(dict(step=step,reason='risk_inference_mismatch'))
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

def report(root,results):
    root=Path(root);lines=['# 全フィールドEvaluator比較の実測結果','',
        '探索的development比較。reserved test未開封。転倒は85度以上/自然tilt_over。window数は独立事故数ではない。',
        'R0=旧全risk、R1=新riskなし、R2=新stallのみ、R3=新全risk。学習seedと走行seedを区別する。','',
        '| field | 学習seed | 条件 | 走行seed | 時間s | 正常waypoint | lap | route進捗m | 最大傾斜° | STOP秒 | 転倒 | 終了・比較成立 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for entry in results:
        a=entry['assignment'];audit=entry.get('audit');r=entry.get('result')
        if not audit or not r:
            lines.append(f"| {a['field']} | {a.get('model_seed','old')} | {a['condition']} | {a['seed']} | — | — | — | — | — | — | — | {entry['state']} |")
            continue
        lines.append(f"| {a['field']} | {a.get('model_seed','old')} | {a['condition']} | {a['seed']} | {r['control_steps']/20:.2f} | {r['normal_waypoints']} | {r['laps']} | {r['route_progress_m']:.3f} | {r['max_tilt_deg']:.2f} | {r['stop_steps']/20:.2f} | {audit['tip_over_observed']} | {r['stop_reason']} / {'比較可' if entry.get('comparison_eligible',False) else '比較N/A'} |")
    lines+=['','境界逸脱以降は地形転倒の教師・集計に加えない。正常waypointは傾斜45度未満などの操作的条件であり、安全性の保証ではない。',
        '従来と同様のPhysX mass/inertia初期化警告が記録されている。設定とログを保持し、物理モデルの完全な妥当性や実機安全性を保証しない。',
        '各fieldのdatasets/coverage.jsonとanchor-audit.json、各modelのtraining.jsonとmetrics.jsonを併せて読む。学習seedを反復しても独立した地形データは増えない。']
    (root/'comparison-report.md').write_text('\n'.join(lines)+'\n')
