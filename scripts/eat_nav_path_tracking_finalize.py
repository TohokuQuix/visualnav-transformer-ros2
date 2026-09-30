"""Freeze decision, compact report, and SHA ledger for the completed CPU audit."""
from __future__ import annotations
import collections,hashlib,json,pathlib
import numpy as np
from eat_nav_path_tracking_stages import DOC,ROOT,UNIFIED,PRIOR,read,atomic,_sha

def main():
    for name,path in [('a0','source-provenance.json'),('a2','path-tracking-metrics.json'),('a3','collapse-classification.json'),('a4','tracking-outcome-analysis.json'),('a5','evaluator-tracking-strata.json'),('a6','evaluator-input-schema-audit.json')]:
        if not (ROOT/f'{name}.done.json').exists() or not (DOC/path).exists():raise ValueError('incomplete stage '+name)
    m=read(DOC/'path-tracking-metrics.json')['runs'];reach=read(DOC/'endpoint-reach-summary.json');collapse=read(DOC/'collapse-classification.json');dist=read(DOC/'path-command-realized-distances.json')['states_and_sets'];out=read(DOC/'tracking-outcome-analysis.json');ev=read(DOC/'evaluator-tracking-strata.json');schema=read(DOC/'evaluator-input-schema-audit.json')
    expected_figures=['planned-realized-nomad.png','planned-realized-bank.png','planned-realized-nominal.png','endpoint-distance.png','progress-ratio.png','cross-track-p95.png','reach-two-vs-six.png','pcr-clusters.png','pcr-mode-transitions.png','delta-v-vs-tracking.png','evaluator-misrank-by-collapse.png']
    if any(not (DOC/f).exists() for f in expected_figures):raise ValueError('missing figure')
    timing=read(DOC/'endpoint-reach-timing.json')['groups']
    per_state={s['state_id']:{name:next(x for x in dist if x['state_id']==s['state_id'] and x['set']==name)['cluster_count'] for name in ('nomad','bank','all_candidate')} for s in collapse['states']}
    ccounts=[v['nomad']['C'] for v in per_state.values()];rcounts=[v['nomad']['R'] for v in per_state.values()]
    macro=ev['state_macro_loso']['all'];cross=out['state_macro_loso']['delta_v']['cross_track_spearman'];progress=out['state_macro_loso']['delta_v']['progress_spearman']
    decisions={'N1_nomad_improvement':'priority','N2_follower_interface':'secondary diagnosis; not dominant P-to-C collapse',
        'N3_embodiment_aware_evaluator':'research candidate after input/teacher remediation; not ready to train at scale',
        'N4_scaled_counterfactual_collection':'not approved','safety_head':'not ready; zero failure-positive termination in this set',
        'new_rollouts_or_training_started':False}
    answers={
        '1_nomad_K8_diverse':'No: planned Path one cluster in 6/7 states; two in 1/7 under frozen threshold.',
        '2_nomad_endpoint_reach':f"{timing['nomad']['reached_by_2s']}/56 by 2 s and {reach['groups']['nomad']['reached']}/56 by 6 s entered 0.2 m; {reach['groups']['nomad']['escaped_after_arrival']}/56 later left; this does not establish shape tracking.",
        '3_path_to_command':'NoMaD command modes range '+str(min(ccounts))+'–'+str(max(ccounts))+'; only the sole multi-Path state retained multiple command modes. No dominant follower collapse.',
        '4_command_to_realized':'NoMaD realized modes: '+str(rcounts)+'; multiple realized modes in only '+str(sum(x>1 for x in rcounts))+'/7 states despite multiple command modes in all seven.',
        '5_primary_loss_stage':'Planned Path is one mode in 6/7 states. C-to-R compression is additional: NoMaD realized trajectories have multiple modes in only one state.',
        '6_tracking_vs_outcome':f"Within-state Spearman cross-track P95 vs ΔV macro {cross['macro_mean']:.3f}; progress ratio vs ΔV {progress['macro_mean']:.3f}. Descriptive, weak association; it does not explain ranking causally.",
        '7_priority':'Improve NoMaD planned candidate diversity first; investigate command-to-realization compression and same-step embodiment input before Evaluator redesign or more collection.',
        '8_missing_input_teacher':'Raw same-step joint effort is absent in 112/112; scaled/noisy Actor observation is not invertible to raw effort. Need verified online joint effort and candidate-specific, failure-positive counterfactual teachers with state-level splits.'}
    decision={'scope':{'states':7,'live_runs':112,'nominal':21,'nomad':56,'bank':35,'excluded_anchors':7,
        'train_only':True,'validation_or_heldout_generalization':False},'decisions':decisions,'answers':answers,
        'evidence':{'nomad_planned_one_mode_states':collapse['counts']['generator_collapse'],
            'nomad_realized_multiple_mode_states':sum(x>1 for x in rcounts),'nomad_command_modes_by_state':ccounts,
            'nomad_realized_modes_by_state':rcounts,'nomad_endpoint_reached':reach['groups']['nomad']['reached'],
            'nomad_endpoint_reached_by_2s':timing['nomad']['reached_by_2s'],
            'nomad_endpoint_escape':reach['groups']['nomad']['escaped_after_arrival'],
            'nomad_cross_track_p95_median_m':float(np.median([r['two_second']['cross_track_p95_m'] for r in m if r['generator']=='nomad'])),
            'delta_v_cross_track_state_macro_spearman':cross,'delta_v_progress_state_macro_spearman':progress,
            'frozen_evaluator_top1_state_macro':macro['top1_recall']['macro_mean'],
            'frozen_evaluator_top3_state_macro':macro['top3_recall']['macro_mean'],
            'frozen_evaluator_pairwise_accuracy_state_macro':macro['pairwise_accuracy']['macro_mean'],
            'frozen_evaluator_oracle_regret_state_macro':macro['oracle_regret']['macro_mean'],
            'raw_joint_effort_available_runs':schema['availability_count']['joint_effort_raw'],
            'termination_count':sum(r['termination'] is not None for r in m if r['outcome_source']=='matched_run')},
        'limitations':['7 train states only','candidate pairs are descriptive, not independent observations',
            '2 s Path and 6 s execution have different horizons','all candidates enter tolerance, so reached/not reached strata lack contrast',
            'raw joint effort absent','failure-positive support zero; safety head not ready',
            'no new rollout, training, candidate regeneration, threshold adjustment, or online selector approval']}
    atomic(DOC/'final-decision.json',decision,once=False)
    lines=['# EAT-Nav NoMaD Path追従性・候補縮退箇所診断','',
        '## 対象と整合性','',
        '7 train stateの112 complete live rolloutを解析した。21 nominal、56 NoMaD、35 bankで、7 anchorをoutcomeから除外した。各runは120 step、20 Hz、6秒。8 future pointのPathは0.25秒間隔、2秒horizonである。既存のPath・trace SHA、command応答、step時刻、座標変換を全件で照合した。map Pathはsimulator環境ローカル、root poseはworldであり、env originを復元して比較した。入力は変更していない。','',
        '## Path追従と終点','',
        '| generator | run | 2秒まで到達 | 6秒まで到達 | 到達後逸脱 | 0–2秒cross-track P95中央値 |',
        '|---|---:|---:|---:|---:|---:|']
    for g in ('nomad','bank','nominal'):
        r=[x for x in m if x['generator']==g];v=reach['groups'][g]
        lines.append(f"| {g} | {len(r)} | {timing[g]['reached_by_2s']} | {v['reached']} | {v['escaped_after_arrival']} | {np.median([x['two_second']['cross_track_p95_m'] for x in r]):.4f} m |")
    lines+=['','「Pathを追従目標として走行した」は全112件で成り立つ。全件が終点0.2 m圏に一度入ったが、到達後の逸脱もあり、Path形状を高精度に追ったことや6秒間終点に留まったこととは異なる。6秒間は全runが非terminalで、failure-positive outcomeはない。0–2秒の幾何学的追従と0–6秒の終点到達を別集計した。','',
        '## 多様性の消失','',
        '| state | terrain | NoMaD P/C/R mode | bank P/C/R mode | NoMaD分類 |',
        '|---|---|---|---|---|']
    for s in collapse['states']:
        k=s['state_id'];n=per_state[k]['nomad'];b=per_state[k]['bank'];lines.append(f"| {k} | {s['terrain']} | {n['P']}/{n['C']}/{n['R']} | {b['P']}/{b['C']}/{b['R']} | {s['classification']} |")
    lines+=['',f"NoMaDのPは6/7 stateで1 mode。Cは{min(ccounts)}–{max(ccounts)} modeだが、Rは6/7 stateで1 modeだった。P段階の候補生成が最初の主なボトルネックで、C→Rでも差が縮む。followerがP→Cを潰すことが主因という証拠はこの7 stateにはない。距離分布と各候補のP/C/R対応はJSONに保存した。cluster閾値は既存値を固定した。",'',
        '## Outcomeと凍結GT(P)','',
        f"cross-track P95とΔVのstate内Spearman平均は{cross['macro_mean']:.3f}、progress ratioとΔVは{progress['macro_mean']:.3f}。leave-one-state-out値は`tracking-outcome-analysis.json`に記録した。追従指標だけでoutcome順位を説明できるとは言えない。全件到達のためreached/not reached比較はNAである。",
        '',f"凍結GT(P)のall-candidate top-1は0/7、top-3は1/7、state-macro pairwise accuracyは{macro['pairwise_accuracy']['macro_mean']:.3f}、平均oracle regretは{macro['oracle_regret']['macro_mean']:.3f}。層別表には支持不足をNAとして残し、selectorとしての再承認はしない。",'',
        '## Evaluator入力と設計判断','',
        'DEM history、relative goal、実際のpost-delay Actor観測、latent、root/joint state、previous action、Path、初手follower commandは112件で取得可能。raw joint effortは0件で、Actor observationに含むscaled/clipped/noisy joint effort with track sumから正確なraw値は逆算できない。同stepのオンラインjoint effort取得とfailure-positive教師を確認するまで、execution-aware Evaluatorの大規模再学習やscaled collectionは提案しない。',
        '', '**優先順:** N1 NoMaD候補多様性を改善。C→R差の縮小をfollower/Actor/車体応答に切り分ける。N3の入力と教師を整備する。N4 scaled counterfactual collectionは保留。Safety headはfailure-positive 0件のため未ready。','',
        '## 8つの問いへの回答','']
    for i,key in enumerate(answers,1):lines.append(f'{i}. {answers[key]}')
    lines+=['','## 図と再現性','',
        'Path重ね描きはNoMaD・bank・nominalで分図した。終点、progress、cross-track、P/C/R mode、遷移、ΔV、GT(P)誤順位の図も同ディレクトリにある。raw seriesは`/tmp/eatnav-candidate-coverage/path-tracking-audit/tracking-series.jsonl.gz`。SHA台帳、実装コード、テストを同branchに保存する。','',
        '適用範囲はこの7 train stateに限る。validation、held-out、実機へ一般化しない。新規simulator rollout、学習、candidate再生成、閾値変更はしていない。','']
    report='\n'.join(lines)
    path=DOC/'path-tracking-audit-report.md'
    temp=path.with_name(path.name+'.tmp');temp.write_text(report);temp.replace(path)
    files=[p for p in DOC.iterdir() if p.is_file() and p.name!='artifact-sha-ledger.json']
    ledger={'artifact_root':str(DOC),'file_count':len(files),'files':{p.name:{'sha256':_sha(p),'bytes':p.stat().st_size} for p in sorted(files)},
        'raw_series':{'path':str(ROOT/'tracking-series.jsonl.gz'),'sha256':_sha(ROOT/'tracking-series.jsonl.gz'),'bytes':(ROOT/'tracking-series.jsonl.gz').stat().st_size},
        'input_shas':{'schedule':_sha(PRIOR/'schedule.json'),'outcomes':_sha(UNIFIED/'analysis/per-candidate-outcomes.json'),
            'evaluator_scores':_sha(UNIFIED/'analysis/frozen-evaluator.json'),'thresholds':_sha(PRIOR/'analysis-thresholds.json')}}
    atomic(DOC/'artifact-sha-ledger.json',ledger,once=False)
    atomic(ROOT/'final.done.json',{'stage':'COMPLETE','decision_sha256':_sha(DOC/'final-decision.json'),
        'report_sha256':_sha(path),'ledger_sha256':_sha(DOC/'artifact-sha-ledger.json')},once=False)
    atomic(ROOT/'status.json',{'stage':'COMPLETE','live_runs':112,'state_count':7,'new_rollouts':False,'new_training':False},once=False)
    print(json.dumps(decision['evidence'],indent=2))
if __name__=='__main__':main()
