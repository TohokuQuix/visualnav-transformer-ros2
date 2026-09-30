"""Offline gate decisions, simulator pilot proposal, report and SHA ledger."""
from __future__ import annotations
import hashlib,json,math,pathlib
import numpy as np
from eat_nav_goal_ood_prepare import REPO,DOC,RAW,sha,read,write

def main():
    if read(RAW/'status.json')['stage'] not in ('G3_ANALYSIS_COMPLETE','COMPLETE') or not (RAW/'g3.done.json').exists():raise ValueError('G3 not complete')
    dist=read(DOC/'train-goal-distribution.json');adh=read(DOC/'goal-adherence-results.json');sens=read(DOC/'goal-sensitivity-results.json');modes=read(DOC/'goal-diversity-results.json')
    pairs=read(DOC/'goal-direction-pairs.json');neighborhood=read(DOC/'dem-goal-neighborhood.json');selection=read(DOC/'state-selection.json')['states']
    if len(selection)!=64 or len(list((RAW/'response-maps').glob('*.png')))!=64:raise ValueError('state or figure support')
    id_train=adh['macro']['train:ID'];id_val=adh['macro']['val:ID'];near_train=adh['macro']['train:near_ood'];near_val=adh['macro']['val:near_ood'];far_train=adh['macro']['train:far_ood'];far_val=adh['macro']['val:far_ood']
    if modes['ID_multi_mode_states']!=64 or sens['macro']['train:ID']['goal_collapse_states']+sens['macro']['val:ID']['goal_collapse_states']!=0:
        raise ValueError('gate summary changed')
    write(DOC/'layer-goal-sensitivity.json',{'status':'not_triggered','rule':'G4 only if G3 observes goal-conditioning collapse',
        'goal_conditioning_collapse_states':0,'checkpoint_or_hooks_changed':False,'attention_causal_claim':False})
    decision={'scope':{'complete_only_train_states':32,'validation_states':32,'goals_per_state':13,'diffusion_seeds':8,
        'sampled_paths':6656,'heldout_used':False,'simulator_rollouts_started':False},
        'gates':{'G_ID':'pass: strong ID endpoint, direction, radius and mode response in both splits',
            'G_OOD':'near_ood_only: near-OOD retains response; far-OOD endpoint error and 0.2m miss rate degrade; physical execution unverified',
            'G_Diversity':'pass: goal sweep yields multiple frozen-threshold Path modes in 64/64 states',
            'G4_internal':'not_triggered: zero goal-conditioning collapse states'},
        'design':{'primary':'D1 local subgoal proposal with D3 train-support guard',
            'D2_retraining':'not indicated by ID/near offline evidence; no model/loss change',
            'D4_maneuver_conditioning':'not priority from this offline result',
            'simulator_pilot':'proposal only; requires separate review and approval'},
        'evidence':{'ID_endpoint_error_median_m':{'train':id_train['state_macro_error_median_m'],'validation':id_val['state_macro_error_median_m']},
            'near_endpoint_error_median_m':{'train':near_train['state_macro_error_median_m'],'validation':near_val['state_macro_error_median_m']},
            'far_endpoint_error_median_m':{'train':far_train['state_macro_error_median_m'],'validation':far_val['state_macro_error_median_m']},
            'far_within_0p2m':{'train':far_train['state_macro_within_0p2m'],'validation':far_val['state_macro_within_0p2m']},
            'ID_goal_effect_exceeds_sampling_states':read(DOC/'goal-vs-sampling-results.json')['id_goal_effect_exceeds_sampling_states'],
            'ID_goal_effect_over_sampling_state_median':{'train':sens['macro']['train:ID']['ratio_state_median'],'validation':sens['macro']['val:ID']['ratio_state_median']},
            'ID_multiple_path_modes_states':modes['ID_multi_mode_states'],
            'ID_same_goal_sampling_collapse_states':sens['macro']['train:ID']['sampling_collapse_states']+sens['macro']['val:ID']['sampling_collapse_states'],
            'near_left_right_consistency':pairs['summary']['train']['near_left_vs_right']['state_macro_success_fraction'],
            'ID_radial_monotonicity':pairs['summary']['train']['ID_long_vs_short']['state_macro_success_fraction']},
        'limits':['Offline geometry_screen_valid is not physically feasible',
            'Goal support threshold fitted only on complete-only train; held-out never used',
            'Validation is evaluation only; 32 states/split and fixed 12 sweep goals do not establish arbitrary-goal planning',
            'No new simulator rollout or model training performed',
            'Validation selection has no boxes terrain; do not claim validation generalization over all seven terrains']}
    write(DOC/'offline-gate-decision.json',decision,mutable=True)
    pilot_ids=['train-003','train-004','train-019','train-021']
    selected=[next(x for x in selection if x['state_id']==sid) for sid in pilot_ids]
    if len({x['terrain'] for x in selected})!=4 or any(x['local_step']>40 for x in selected):raise ValueError('pilot proposal state budget drift')
    proposals=[];inputs=read(DOC/'goal-sweep-inputs.json')['inputs']
    goal_names=['logged','id_forward_long','near_left_lateral','far_diag_right']
    for state in selected:
        for name in goal_names:
            row=next(x for x in inputs if x['state_id']==state['state_id'] and x['goal_id']==name)
            if not row['geometry_screen_valid']:raise ValueError('pilot proposal includes screen-invalid goal')
            proposals.append({'state_id':state['state_id'],'terrain':state['terrain'],'goal_id':name,
                'goal_class':row['declared_class'],'goal_xy':row['goal_xy'],'K':2,'seeds':[1001,1002],
                'geometry_screen_valid':True})
    pilot={'status':'proposal_only_requires_separate_approval','state_count':4,'states':selected,
        'goal_types':['original logged','ID','near-OOD screen-valid','far-OOD screen-valid'],
        'goal_count_per_state':4,'K_per_goal':2,'paired_seeds':[1001,1002],
        'nominal_duplicate_runs_per_state':3,'candidate_runs':32,'duplicate_control_runs':12,'total_runs':44,
        'goal_conditioning_collapse_goal':'NA: no collapse observed in 64 offline states; no invented target/run',
        'estimated_wall_time_sec':44*158.52*1.2,'estimated_wall_time_hours':44*158.52*1.2/3600,
        'time_basis':'prior 112 readiness live run median 158.52s including follower lifecycle; 20% reserve; prefix <=40 steps',
        'estimated_gpu_peak_mib':5551,'gpu_basis':'prior fresh-bridge readiness peak',
        'estimated_storage_bytes':int(1.088e9/112*44*1.5),'storage_basis':'prior 112-run 1.088GB, 50% reserve',
        'planned_metrics':['endpoint arrival','0-2s cross-track','6s arrival','realized mode','delta V','H-step advantage','drawdown','stack','excessive tilt','termination'],
        'offline_screen_not_feasibility':True,'rollout_started':False,'candidate_goal_entries':proposals}
    write(DOC/'simulator-pilot-proposal.json',pilot)
    with np.load(RAW/'train-derived.npz') as raw:goals=raw['goals']
    gr=np.linalg.norm(goals,axis=1);back=float(np.mean(goals[:,0]<0));lateral=float(np.mean(np.abs(goals[:,1])>=.1))
    report=f'''# EAT-Nav NoMaD Goal-conditioning / Goal-OOD offline診断

## 実行範囲と監査

凍結best EMA checkpoint、complete-only train 49,598 window・validation 6,171 window、既存split hash、train-only action normalizationを照合した。train 32 state、validation 32 stateへ、各13 goal × 同一8 seedを固定し、合計6,656 Pathをoffline生成した。held-out、simulator、再学習、候補再生成、閾値変更は使っていない。推論前にstate・goal・seed・順序・入力SHAを固定した。

relative goalは現在のrobot-local frame（x前方、y左）の2秒後までの最後のvalid future endpointを半径0.8 mへclipした値である。complete-onlyでは8点目そのものを使う。clip前endpointは累積actionから復元でき、保存goalはclip後である。partial terminalは最後のvalid point、zero-validはゼロgoalを使うが、この診断には含めていない。

## Train goal support

train goal半径の中央値は{dist['postclip_radius_quantiles_m'][3]:.3f} m、P90は{dist['postclip_radius_quantiles_m'][5]:.3f} m。clip率{dist['clip_rate']:.1%}。方向sectorは前方{dist['direction_fractions']['forward']:.1%}、後方{dist['direction_fractions']['backward']:.1%}、左{dist['direction_fractions']['left']:.1%}、右{dist['direction_fractions']['right']:.1%}。半径<0.2 mは{np.mean(gr<.2):.1%}、≥0.6 mは{np.mean(gr>=.6):.1%}。x<0の割合{back:.1%}、|y|≥0.1 mは{lateral:.1%}（sector定義と異なる）。

trainだけで極座標binの独立episode支持とkNN k=16をfitした。kNN q95={read(DOC/'goal-support-thresholds.json')['near_distance_m']:.4f} m、q99={read(DOC/'goal-support-thresholds.json')['far_distance_m']:.4f} m。validationは境界決定に使っていない。1,024個の固定train DEM probeのうち、近傍DEMで別episodeが見つかったのは{neighborhood['probes_with_cross_episode_near']}件、そのうち複数goal binは{neighborhood['probes_with_multiple_goal_bins']}件。この近傍検索は候補shortlistを用いる標本診断であり、全件厳密近傍保証ではない。

## Goal adherence と OOD

| 区分 | train endpoint誤差（state中央値平均） | validation | train 0.2 m内 | validation 0.2 m内 |
|---|---:|---:|---:|---:|
| ID | {id_train['state_macro_error_median_m']:.4f} m | {id_val['state_macro_error_median_m']:.4f} m | {id_train['state_macro_within_0p2m']:.1%} | {id_val['state_macro_within_0p2m']:.1%} |
| near-OOD | {near_train['state_macro_error_median_m']:.4f} m | {near_val['state_macro_error_median_m']:.4f} m | {near_train['state_macro_within_0p2m']:.1%} | {near_val['state_macro_within_0p2m']:.1%} |
| far-OOD | {far_train['state_macro_error_median_m']:.4f} m | {far_val['state_macro_error_median_m']:.4f} m | {far_train['state_macro_within_0p2m']:.1%} | {far_val['state_macro_within_0p2m']:.1%} |

同一DEMとseedを保ちgoalだけを変えると、ID Path RMS差中央値はtrain {sens['macro']['train:ID']['goal_effect_state_median_m']:.3f} m、validation {sens['macro']['val:ID']['goal_effect_state_median_m']:.3f} m。同一goalのseed変更差はそれぞれ{sens['macro']['train:ID']['sampling_effect_state_median_m']:.3f} m、{sens['macro']['val:ID']['sampling_effect_state_median_m']:.3f} mで、比は約{sens['macro']['train:ID']['ratio_state_median']:.1f}倍。分母へepsilonを足していない。goal効果がseed効果を上回るstateは64/64。

同距離左右goalのPath endpoint移動方向はnear・farともtrain/validationで全32 state × 8 seedで整合した。IDの同方位0.2→0.4 mへの延長も全件でendpoint距離が増えた。ID goal sweepは64/64 stateで複数Path mode（中央値4）を生成した。一方、同一ID goalの8 seedは平均約1.25 modeで、前回K=8 collapseと整合する。

far-OODは方向反応を保つ一方、endpoint誤差と0.2 m未達率が増えた。DEM mask・範囲・footprint・高さ不連続のscreenは別途保存した。screen通過は`geometry_screen_valid`であり、物理的到達可能性を意味しない。

## Gates と設計判断

- G-ID: **pass**。ID goalの位置・方向・距離・Path modeはtrain/validationで一貫して反応。
- G-OOD: **near_ood_only**。near-OODは維持し、far-OODで誤差が増える。validationのboxes supportはない。実走行は未確認。
- G-Diversity: **pass**。goal sweepは64/64 stateで複数mode。
- G4内部activation: goal-conditioning collapseが0件のため**未発火**。hookは実行していない。

validation選択には`boxes`が存在しないため、全7 terrainへのvalidation一般化は主張しない。`unsupported` goal stratumは固定sweepに含めておらず、性能評価はNAである。

主な推奨は**D1 local subgoal proposalをID/near-OODへ制限して検討し、D3 support guardを併用**すること。IDでもgoal無視というD2の根拠はなく、D4 maneuver tokenは今回の結果から優先しない。offlineだけで「任意goalへ到達するplanner」とは認定しない。

## 10の問いへの回答

1. train goalは半径0～0.8 mの保存領域を持ち、方位は全周に例があるが、横方向は薄い。分布図とterrain別表を参照。
2. 後方sector {dist['direction_fractions']['backward']:.1%}、左右sector合計{dist['direction_fractions']['left']+dist['direction_fractions']['right']:.1%}、短距離<0.2 m {np.mean(gr<.2):.1%}、長距離≥0.6 m {np.mean(gr>=.6):.1%}。
3. IDでendpointはgoalへ平均約6 mmのstate中央値誤差で移動した。
4. ID goal効果はsampling効果の約5倍で、64/64 stateで上回った。
5. near-OODは約4.4～4.5 cmの誤差、far-OODは約17 cmへ悪化した。
6. goalを変えると64/64 stateで複数のplanned Path modeを得た。
7. 前回同一goal K=8のcollapseは主にsampling多様性不足を示し、ID/nearでgoal入力は無視されていない。
8. local subgoal sweepはoffline Path多様化に有望。ただし実走行のmode・outcomeへ保存されるか未確認。
9. 今回のID/near結果だけでは再学習・maneuver conditioningを優先しない。far-OOD運用にはtrain support guardが必要。
10. 承認後の提案は4 train state×4 goal×K2＋nominal duplicate 12＝44 run、約{pilot['estimated_wall_time_hours']:.2f}時間、GPU約5.55 GiB、保存量上限約{pilot['estimated_storage_bytes']/1e9:.2f} GB。goal-conditioning collapse対象は0件のためNA。

## 将来のsimulator pilot案

[simulator-pilot-proposal.json](simulator-pilot-proposal.json)に短prefix・4 terrainの4 state、logged/ID/near/farのscreen-valid goal、seed、重複control、実測に基づく時間・容量、測定項目を固定した。今回このpilotは実行していない。offline結果のレビューと別承認が必要。

## 図とraw

11枚の要約図を同ディレクトリに、64 stateのresponse mapを`/tmp/eatnav-candidate-coverage/goal-ood/response-maps/`に保存した。per-path結果と推論NPZは`/tmp/eatnav-candidate-coverage/goal-ood/`にある。stateを統計単位とし、goal/seed pairを独立標本とは扱わない。欠損支援stratumはNAとする。
'''
    target=DOC/'goal-ood-report.md';target.write_text(report)
    files=[p for p in DOC.iterdir() if p.is_file() and p.name!='artifact-sha-ledger.json']
    ledger={'artifact_root':str(DOC),'file_count':len(files),'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)},
        'raw':{'selected_states_npz_sha256':sha(RAW/'selected-states.npz'),'train_derived_npz_sha256':sha(RAW/'train-derived.npz'),
            'per_path_results_sha256':sha(RAW/'per-path-results.jsonl.gz'),'inference_count':64,'response_map_count':64},
        'source':{'checkpoint_sha256':sha('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt'),
            'manifest_sha256':sha(RAW/'input-copy/manifest.json'),'config_sha256':sha(DOC/'goal-ood-config.yaml')}}
    write(DOC/'artifact-sha-ledger.json',ledger,mutable=True)
    write(RAW/'final.done.json',{'stage':'COMPLETE','decision_sha256':sha(DOC/'offline-gate-decision.json'),
        'report_sha256':sha(target),'ledger_sha256':sha(DOC/'artifact-sha-ledger.json')},mutable=True)
    write(RAW/'status.json',{'stage':'COMPLETE','paths':6656,'simulator_rollouts':0,'new_training':False},mutable=True)
    print('OFFLINE_GOAL_OOD_COMPLETE',decision['gates'],flush=True)
if __name__=='__main__':main()
