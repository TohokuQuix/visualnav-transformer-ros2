"""Small, reproducible design artifacts from audited E1 traces and train geometry."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
matplotlib.rcParams['svg.hashsalt']='eatnav-evaluator-design-review-v1'

from eat_nav_evaluator_design_audit import NEW, OLD, OLD_CORRECTED, read, sha
from eat_nav_evaluator_design_analyze import OUTPUT, CANDIDATES


def write(name, data):
    (OUTPUT/name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def fig(name):
    plt.tight_layout()
    path=OUTPUT/'figures'/name
    plt.savefig(path, format='svg', bbox_inches='tight', metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close()


def input_contract():
    def item(name, shape, unit, frame, sim, real, timing, norm, missing, parity, source):
        return dict(name=name, shape=shape, unit=unit, frame=frame, simulator_available=sim,
                    real_robot_available=real, same_anchor_time=timing, normalization=norm,
                    missing_policy=missing, production_parity=parity, source=source)
    fields = [
        item('dem_history',[4,2,20,20],'height m + binary mask','base-yaw raster',True,'GridMap adapter exists; measured parity unverified',
             '4 frames at -0.75,-0.50,-0.25,0 s; exact anchor identity','subtract scalar reference height only; train-only valid-cell scaling','mask unknown, never free-fill','sim source-to-base transform verified; physical GridMap parity pending','anchor.dem_history'),
        item('unknown_mask',[4,20,20],'binary','base-yaw raster',True,'GridMap unknown cells available; parity unverified',
             'paired with each DEM frame','none','explicit unknown indicator','physical unknown encoding pending','DEM channel 1'),
        item('relative_goal',[2],'m','base',True,'ROS goal transform available; timestamp parity pending',
             'same anchor goal for all candidates','fixed 2 m coordinate scale','reject absent goal','sim Path/goal identity verified; hardware pending','frozen candidate endpoint'),
        item('body_orientation',[3],'gravity vector, unitless','base',True,'IMU/TF may provide; parity unverified',
             'anchor snapshot','unit vector, no absolute map yaw','coverage mask and reject if absent','sim quaternion verified; real IMU extrinsic pending','anchor.root_state[3:7]'),
        item('body_velocity',[3],'m/s','base',True,'odometry may provide; parity unverified',
             'anchor snapshot','train-only channel statistics','mask and reject critical absence','sim world-to-base transform required; hardware pending','anchor.root_state[7:10]'),
        item('angular_velocity',[3],'rad/s','base',True,'IMU may provide; parity unverified',
             'anchor snapshot','train-only channel statistics','mask and reject critical absence','sim world-to-base transform required; hardware pending','anchor.root_state[10:13]'),
        item('joint_position',[52],'joint-specific rad/m','ordered RL joint vector',True,'JointState likely available; exact 52-order parity unverified',
             'anchor snapshot','joint-wise train-only statistics','joint availability mask; reject order mismatch','joint order and units require physical audit','anchor.joint_position'),
        item('joint_velocity',[52],'joint-specific rad/s or m/s','ordered RL joint vector',True,'JointState likely available; exact 52-order parity unverified',
             'anchor snapshot','joint-wise train-only statistics','joint availability mask; reject order mismatch','joint order and timing require physical audit','anchor.joint_velocity'),
        item('previous_action',[6],'RL action units','Actor action order',True,'possible only via deployed Actor/ActionManager; parity unverified',
             'anchor previous-action snapshot','checkpoint action metadata; train-only stats if learned','availability mask; exclude for portable model if absent','source checkpoint and action semantics fixed','anchor.action_manager_previous_action'),
        item('anchor_raw_joint_effort',[52],'sim applied torque/force units','ordered RL joint vector',False,'physical equivalent/order/timing unverified',
             'not saved in anchor snapshot; first live step is post-Path','none until parity established','exclude from primary input','not established','live trace raw_joint_effort only'),
        item('anchor_actor_latent',[256],'checkpoint latent units','frozen RL Actor',True,'requires same deployed checkpoint and latent API; unverified',
             'anchor-only latent exists; live first-step latent is post-Path and excluded','frozen checkpoint; train-only stats','portable model excludes; conditional model masks','sim anchor only, physical parity pending','anchor.actor_latent'),
        item('candidate_path',[8,2],'m','base',True,'ROS Path base/map transform exists; physical timing unverified',
             'known before selection, same anchor/goal','fixed physical scale; 0.25 s waypoint interval','candidate invalid if absent','E1 frozen bank and Path SHA verified','8 future waypoints; implicit start (0,0)'),
        item('feature_coverage_mask',['feature-dependent'],'binary','base DEM/Path',True,'derived if DEM and Path available',
             'deterministic from anchor inputs','none','retain unknown/partial coverage, no free imputation','deterministic feature code to be parity-tested','DEM mask x frozen Path')]
    return {'schema_version':'evaluator_preexecution_input_v1','primary_portable_fields':[x['name'] for x in fields if x['name'] not in ('anchor_actor_latent','anchor_raw_joint_effort')],
            'optional_checkpoint_dependent_field':'anchor_actor_latent',
            'excluded_until_capture_and_parity':'anchor_raw_joint_effort',
            'fields':fields,'all_normalization_statistics_train_only':True,
            'post_Path_live_trace_not_anchor_input':True}


def figures(labels, concord, support, aggregate):
    (OUTPUT/'figures').mkdir(exist_ok=True)
    groups=list(labels['candidate_values_by_group'])
    short=[g.replace('expanded-train-','E').replace('g0e-train-','G') for g in groups]
    noise=read(OUTPUT/'direct-repeat-noise.json')
    fig1,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,metric in zip(axes.ravel(),['delta_v','H_step_advantage','cumulative_reward','progress_m']):
        ranges=[]
        for sid in groups:
            v=labels['candidate_values_by_group'][sid][metric];ranges.append(max(v.values())-min(v.values()))
        ax.bar(short,ranges,label='candidate range')
        ax.plot(short,[noise[g][metric]['duplicate_variation_floor'] for g in groups], 'ko',label='direct repeat floor')
        ax.set_title(metric);ax.tick_params(axis='x',rotation=30)
    axes[0,0].legend(fontsize=8);fig1.suptitle('StateGroup outcome range and direct repeat noise')
    fig('01-outcome-range-noise.svg')
    fig2,axes=plt.subplots(1,4,figsize=(13,4),sharey=True)
    for ax,metric in zip(axes,['delta_v','H_step_advantage','cumulative_reward','progress_m']):
        rank=np.asarray([[concord[sid]['rankings'][metric].index(cid)+1 for cid in CANDIDATES] for sid in groups])
        ax.imshow(rank,vmin=1,vmax=7,cmap='viridis_r');ax.set_title(metric);ax.set_xticks(range(7),CANDIDATES,rotation=75,fontsize=7)
    axes[0].set_yticks(range(7),short);fig2.suptitle('Within-StateGroup candidate ranks (1=best)')
    fig('02-label-rank-comparison.svg')
    fig3,axes=plt.subplots(1,2,figsize=(13,5))
    metrics=['delta_v','H_step_advantage','cumulative_reward','progress_m','drawdown','cross_track_0_2s_p95_m','endpoint_error_at_2s_m']
    for ax,kind in zip(axes,['spearman','kendall']):
        matrix=np.full((len(metrics),len(metrics)),np.nan)
        for i,a in enumerate(metrics):
            for j,b in enumerate(metrics):
                if i==j:
                    matrix[i,j]=1
                else:
                    key=a+'__'+b if i<j else b+'__'+a
                    values=[concord[sid]['all_label_pairs'][key][kind] for sid in groups]
                    values=[v for v in values if v is not None]
                    if values:matrix[i,j]=statistics.mean(values)
        ax.imshow(np.ma.masked_invalid(matrix),vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(len(metrics)),metrics,rotation=70,ha='right',fontsize=7)
        ax.set_yticks(range(len(metrics)),metrics,fontsize=7);ax.set_title(kind+' macro (7 groups)')
    fig('03-rank-concordance.svg')
    fig4,ax=plt.subplots(figsize=(8,4))
    effects=[max(x['absolute_difference'] for x in labels['candidate_vs_direct_by_group'][g]['delta_v'].values()) for g in groups]
    ax.bar(short,effects);ax.set_ylabel('maximum |candidate - direct| ΔV');ax.set_title('Duplicate denominator is exactly zero in all 7 groups; ratios NA')
    fig('04-candidate-effect-vs-duplicate.svg')
    fig5,ax=plt.subplots(figsize=(7,4))
    stage=[]
    old_stage=read(OLD_CORRECTED/'stagewise-signal-results.json')['groups']
    new_stage=read(NEW/'analysis/stagewise-P-C-A-R-Y.json')
    for group in groups:
        g=next(x for x in aggregate['group_gates'] if x['state_group_id']==group)
        actor_response=(old_stage[group]['candidate_exceeds_duplicate']['actor_action'] if group in old_stage
                        else max(new_stage[group]['actor_action']['rms'][0][1:])>0)
        stage.append([1,int(g['P_to_C_pass']),int(actor_response),int(g['C_to_R_pass']),int(g['R_to_Y_difference_corrected'])])
    im=ax.imshow(stage,vmin=0,vmax=1,cmap='RdYlGn');ax.set_xticks(range(5),['P','C','A','R','Y']);ax.set_yticks(range(7),short)
    for i in range(7):
        for j in range(5):ax.text(j,i,str(stage[i][j]),ha='center',va='center')
    ax.set_title('Descriptive stage retention (1=observed)');fig('05-pcary-retention.svg')
    fig6,ax=plt.subplots(figsize=(11,3));ax.axis('off')
    text='4-frame DEM + mask  |  base body state  |  relative goal  |  8-point Path\n'
    text+='↓ deterministic Path/DEM geometry + coverage\n'
    text+='shared encoder → group-relative quality / ΔV-advantage / progress / drawdown-tracking / uncertainty\n'
    text+='Safety head: design only; no training labels'
    ax.text(.5,.5,text,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round',fc='white',ec='black'))
    fig('06-evaluator-input-output.svg')
    fig7,ax=plt.subplots(figsize=(9,4))
    terr=list(support['terrain']);v=[support['terrain'][t]['independent_episode_support'] for t in terr]
    ax.bar(terr,v);ax.axhline(2,color='red',linestyle='--',label='2 per terrain target')
    ax.set_xticks(range(len(terr)),terr,rotation=30,ha='right');ax.set_ylabel('independent episode support');ax.legend()
    fig('07-hard-state-terrain-support.svg')
    fig8,ax=plt.subplots(figsize=(7,4));ax.plot([0,.25,.5,.75,1],[0,0,0,0,0],alpha=0)
    ax.text(.5,.72,'Prospective evaluation only',ha='center',transform=ax.transAxes,fontsize=12)
    ax.text(.5,.50,'Bin by predicted uncertainty within held-out StateGroups',ha='center',transform=ax.transAxes)
    ax.text(.5,.32,'Report misranking rate and calibration with group bootstrap',ha='center',transform=ax.transAxes)
    ax.set_xlabel('Predicted uncertainty (to be measured)');ax.set_ylabel('Misranking (NA: no model yet)')
    fig('08-uncertainty-misranking-plan.svg')


def main():
    labels=read(OUTPUT/'evaluator-label-vector.json')
    concord=read(OUTPUT/'label-ranking-concordance.json')
    critic=read(OUTPUT/'critic-label-audit.json')
    support=read(OUTPUT/'hard-state-support-audit.json')
    aggregate=read(NEW/'analysis/seven-terrain-aggregate.json')
    old_prov=read(OLD_CORRECTED/'source-provenance.json')
    provenance=read(OUTPUT/'source-provenance.json')
    provenance['checkpoint_sha256']=old_prov['checkpoint_sha256']
    provenance['reference_config_sha256']=old_prov['reference_config_sha256']
    provenance['old_frozen_source_sha256']={k:v['sha256'] for k,v in old_prov['frozen_source'].items()}
    provenance['expanded_source_sha256']=read(NEW/'runtime-source-freeze-v2.json')['source_sha256']
    provenance['expanded_postprocess_path']='/tmp/eatnav-expanded-feasibility-gpu-diagnostic-20261005/postprocess_after_50.py'
    provenance['expanded_postprocess_sha256']=sha(Path(provenance['expanded_postprocess_path']))
    provenance['candidate_generator_config_sha256']=sha(Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/expanded-feasibility/expanded-pilot-config.yaml')
    provenance['frozen_candidate_bank_sha256']=sha(Path('/tmp/eatnav-candidate-bank-g0e/g0e-candidate-bank.jsonl.gz'))
    provenance['coordinate_addendum_sha256']=sha(OLD_CORRECTED/'corrected-outcome-results.json')
    write('source-provenance.json',provenance)
    contract=input_contract();write('evaluator-input-contract.json',contract)
    write('online-input-availability.json',{'simulator_anchor_snapshot_groups':7,
        'live_trace_count':63,'complete_candidate_steps_per_trace':120,
        'prechoice_available':['dem_history','relative_goal','body_orientation','body_velocity','angular_velocity','joint_position','joint_velocity','previous_action','candidate_path','anchor_actor_latent'],
        'anchor_raw_joint_effort_capture':False,'real_robot_joint_effort_parity':'unverified',
        'live_actor_latent_after_Path_is_future_response':True,
        'production_parity':'not established for physical robot',
        'exact_stamp_sim_protocol':'validated by sealed protocol-valid markers'})
    exclusions=['future_follower_command','future_Actor_action','live_post_Path_Actor_latent','realized_trajectory','post_rollout_contact','post_rollout_effort','post_rollout_reward','post_rollout_Critic','termination','candidate_outcome','success_or_failure_label','candidate_id','family_name','amplitude_name','run_order','environment_absolute_origin','future_trace_feature']
    write('leakage-exclusion-list.json',{'primary_model_exclusions':exclusions,
        'metadata_only':['candidate_id','family_name','amplitude_name','run_order'],
        'future_trace_used_only_as_teacher':True,'group_split_required':True})
    write('path-dem-feature-contract.json',{'source':'anchor DEM history + candidate Path only',
        'features':['Path_length_m','waypoint_spacing_m','heading_change_rad','curvature_per_m','start_end_tangent_rad',
                    'DEM_height_profile_m','slope_profile','roughness_std_m','height_discontinuity_m',
                    'unknown_cell_coverage','centerline_known_fraction','footprint_coverage_diagnostic','direct_Path_difference_m'],
        'frame':'base-yaw DEM and base Path', 'sampling':'frozen E0 dense centerline; 0.1 m DEM; all 4 history frames retain masks',
        'unknown_policy':'never treat unknown as free; preserve coverage mask and NA values',
        'footprint_note':'diagnostic only until robot collision and hardware footprint parity checked',
        'candidate_id_family_amplitude_are_not_inputs':True})
    proposal={'status':'prospective_design_not_train_ready','preferred_architecture':'multi_task_ranking_after_teacher_revision',
       'losses':{'pairwise_logistic':'good for relative order; apply duplicate-aware ties and StateGroup weighting',
                 'margin_ranking':'sensitive to fixed margin and sparse labels; secondary ablation',
                 'listwise':'requires stable group-level total order; defer while label discordance exists',
                 'Huber_regression':'for ΔV/advantage/progress/drawdown calibration heads; group-normalized targets and raw values both reported',
                 'multi_task_weighted':'shared representation; weights must be frozen prospectively, no optimization on these 7 groups',
                 'tie_aware_ranking':'exclude pairs within max(direct-repeat floor, measurement floor); denominator-zero recorded separately',
                 'uncertainty_aware':'heteroscedastic head is exploratory; calibrate on held-out groups, do not use missing safety positives'},
       'prospective_measurement_floors':{'delta_v_critic_units':0.01,'source':'pre-existing frozen E1 critic difference floor; not fitted here'},
       'numerical_loss_weights':None,'reason_weights_unset':'primary teacher direction unresolved across ΔV/advantage/reward/progress',
       'safety_head_training':False}
    (OUTPUT/'loss-design-proposal.yaml').write_text(yaml.safe_dump(proposal,sort_keys=False,allow_unicode=True))
    split={'unit':'source trajectory/environment instance/terrain seed and episode; keep all anchors/candidates/repeats together',
           'random_candidate_split_forbidden':True,'source_episode_id_zero_can_repeat_across_distinct_environment_instances':True,
           'train_validation_test':'partition distinct source trajectories and terrain seeds before state selection; nested grouping by environment then episode',
           'terrain_generalization':'separate leave-one-terrain-out evaluation only after enough independent groups per terrain; current 1 per terrain is insufficient',
           'model_selection':'validation aggregate StateGroup macro only; test and held-out never select model or thresholds',
           'metrics':['StateGroup-macro pairwise accuracy','top-1 oracle recall','top-3 oracle recall','oracle regret',
                      'improvement versus fixed direct','fraction worse than direct','calibration','uncertainty versus misranking',
                      'terrain breakdown with support','easy/difficult geometry breakdown with support'],
           'uncertainty':'bootstrap StateGroups/seed clusters, not candidate pairs; report NA when support insufficient',
           'current_7_group_model_performance':'NA / no Evaluator trained'}
    (OUTPUT/'split-and-metrics-proposal.yaml').write_text(yaml.safe_dump(split,sort_keys=False,allow_unicode=True))
    measured_seconds=8529.150555419921; measured_processes=50; measured_bytes=423_930_000
    forecast={'proposed_new_groups':14,'processes_per_group':{'anchor_only':1,'direct_repeat':3,'non_direct':6},
              'new_max_processes':140,'source_measurement':{'sealed_processes':50,'active_seconds':measured_seconds,
                    'combined_storage_bytes_approx':measured_bytes,'gpu_device_total_peak_mib':5959.56,
                    'simulator_pss_peak_gib':5.38,'anonymous_plus_shmem_peak_gib':3.724},
              'serial_runtime_point_sec':measured_seconds/50*140,
              'serial_runtime_with_20pct_margin_sec':measured_seconds/50*140*1.2,
              'final_storage_point_bytes_approx':int(measured_bytes/50*140),
              'final_storage_with_20pct_margin_bytes_approx':int(measured_bytes/50*140*1.2),
              'gpu_peak_forecast_mib':'about 5960 if serial and same process topology; not a guarantee',
              'ram_forecast':'PSS about 5.38 GiB; anon+shmem about 3.72 GiB if serial, measured topology',
              'zenoh':'external router preflight required; runner neither starts nor stops router',
              'fresh_process':'anchor/duplicate/candidate must retain per-run sealed lifecycle',
              'support_by_terrain':{k:{'independent_episode_support':v['independent_episode_support'],
                                       'target_two_supported':v['two_state_target_supported']} for k,v in support['terrain'].items()},
              'all_seven_terrain_two_state_target_supported':all(v['two_state_target_supported'] for v in support['terrain'].values()),
              'decision':'not executable as 14 groups under frozen short-prefix geometry support; no collection authorization'}
    write('hard-state-collection-proposal.json',forecast)
    safety={'ready_to_train':False,'failure_positive_live_trajectories':0,'verified_stack_trajectories':0,
            'events':['verified_stack','excessive_tilt','termination_by_cause','recovery','high_effort','contact_anomaly','progress_failure','censored_timeout','infrastructure_failure'],
            'censored_timeout_is_failure':False,'infrastructure_failure_is_robot_failure':False,
            'prospective_per_event_minimum':{'train_positive_distinct_StateGroups':40,'validation_positive_distinct_StateGroups':10,
                'test_positive_distinct_StateGroups':10,'train_negative_distinct_StateGroups':40,
                'distinct_terrain_types_with_positives':3,'distinct_terrain_seeds_with_positives':10},
            'note':'event-specific counts; do not pool causes to satisfy threshold'}
    write('safety-head-readiness.json',safety)
    decision={'gate':'needs_label_revision','source_integrity':'pass','coordinate_frame':'corrected map-frame audited',
              'label_issue':'ΔV ordering conflicts with H-step advantage/reward/progress across StateGroups; no unique primary teacher',
              'input_contract':'sim anchor portable subset definable; real production parity unverified; raw anchor effort absent',
              'hard_state_support':'boxes and hf_pyramid_slope have zero independent short-prefix support under prospective frozen geometry rules',
              'evaluator_training_ready':False,'safety_head_training_ready':False,
              'next_approval_recommendation':'teacher-definition and production-input parity review, then revise prospective hard-state selection; no 14-group rollout yet',
              'candidate_generator_change_recommended':False,'new_rollout_started':False,'model_training_started':False}
    write('design-decision.json',decision)
    figures(labels,concord,support,aggregate)
    return forecast,decision


if __name__=='__main__':
    forecast,decision=main()
    print(json.dumps({'gate':decision['gate'],'runtime_hours_20pct':forecast['serial_runtime_with_20pct_margin_sec']/3600,
                      'storage_gb_20pct':forecast['final_storage_with_20pct_margin_bytes_approx']/1e9},sort_keys=True))
