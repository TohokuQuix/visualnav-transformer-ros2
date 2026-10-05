"""Read-only, source-checked 40-transition teacher audit of sealed E1 traces."""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import kendalltau, spearmanr

from eat_nav_evaluator_design_audit import main as source_audit, read, sha, OLD, NEW
from eat_nav_evaluator_design_analyze import input_records
from eat_nav_e1_analyze import load_trace
from eat_nav_e1_coordinate_addendum import _along_path_progress, yaw_from_root

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/quality-identifiability-gate/g40-teacher-review'
RUN = Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16')
AGENT = RUN / 'params/agent.yaml'
ENV = RUN / 'params/env.yaml'
CHECKPOINT = RUN / 'model_10000.pt'
ONIX_ENV_SOURCE = Path('/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/envs.py')
ONIX_OBS_SOURCE = Path('/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/mdp/observations.py')
REWARD_MANAGER_SOURCE = Path('/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py')
RL_WRAPPER_SOURCE = Path('/home/kojima/lib/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py')
CANDIDATES = ('direct_reference', 'left_1', 'left_2', 'left_3', 'right_1', 'right_2', 'right_3')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def validate_trace_index(trace, horizon=40):
    if len(trace) <= horizon:
        raise ValueError('bootstrap pre-step row absent')
    first = trace[0]['logical_sim_time_ns']
    for k, row in enumerate(trace):
        if row['logical_sim_time_ns'] != first + 50_000_000 * k:
            raise ValueError(f'timestamp step {k}')
        if k + 1 < len(trace) and row['post_root_state'] != trace[k + 1]['root_state']:
            raise ValueError(f'boundary step {k}')
        if row['reward'] is None or not math.isfinite(row['reward']) or not math.isfinite(row['critic_value']):
            raise ValueError(f'nonfinite reward/Critic step {k}')
    if trace[horizon]['logical_sim_time_ns'] - first != 2_000_000_000:
        raise ValueError('40-step clock mismatch')


def return_at_horizon(trace, horizon, gamma, *, terminal_pre_reset_value=None):
    """Row k stores pre-transition V_k and post-transition reward r_k."""
    if not 0 < gamma <= 1 or horizon <= 0:
        raise ValueError('invalid discount or horizon')
    n = min(horizon, len(trace))
    terminal = next(((k, row['terminal']) for k, row in enumerate(trace[:n]) if row['terminal']), None)
    if terminal:
        n = terminal[0] + 1
    rewards = [float(row['reward']) for row in trace[:n]]
    discounted = math.fsum(gamma ** k * r for k, r in enumerate(rewards))
    out = {'observed_transitions': n, 'reward_discounted': discounted,
           'reward_undiscounted': math.fsum(rewards), 'bootstrap_index': None,
           'bootstrap_value': None, 'g': None, 'advantage': None, 'status': 'ok'}
    if terminal:
        event = terminal[1]
        if event.get('kind') in {'measurement_invalid', 'infrastructure_failure', 'manual_stop'}:
            out['status'] = 'excluded_' + event['kind']
            return out
        if event.get('terminated'):
            value = 0.0
            out['status'] = 'physical_terminal_no_bootstrap'
        elif event.get('truncated'):
            if terminal_pre_reset_value is None:
                out['status'] = 'timeout_missing_terminal_pre_reset_value'
                return out
            value = float(terminal_pre_reset_value)
            out['status'] = 'time_limit_terminal_pre_reset_bootstrap'
        else:
            out['status'] = 'unknown_terminal_excluded'
            return out
    elif len(trace) > n:
        value = float(trace[n]['critic_value'])
        out['bootstrap_index'] = n
    else:
        out['status'] = 'missing_bootstrap_row'
        return out
    if terminal and event.get('terminated'):
        out['g'] = discounted
    else:
        out['bootstrap_value'] = value
        out['g'] = discounted + gamma ** n * value
    out['advantage'] = out['g'] - float(trace[0]['critic_value'])
    return out


def rank_compare(a, b):
    av, bv = np.asarray(a, float), np.asarray(b, float)
    if len(av) != 7:
        raise ValueError('seven candidate values required')
    rho = float(spearmanr(av, bv).statistic) if len(set(av)) > 1 and len(set(bv)) > 1 else None
    tau = float(kendalltau(av, bv).statistic) if rho is not None else None
    pairs = [(i, j) for i, j in itertools.combinations(range(7), 2) if av[i] != av[j] and bv[i] != bv[j]]
    agree = sum((av[i] - av[j]) * (bv[i] - bv[j]) > 0 for i, j in pairs)
    top_a = sorted(range(7), key=lambda i: (-av[i], CANDIDATES[i]))
    top_b = sorted(range(7), key=lambda i: (-bv[i], CANDIDATES[i]))
    return {'spearman': rho, 'kendall': tau,
            'pairwise_agreement': agree / len(pairs) if pairs else None,
            'comparable_pairs': len(pairs), 'descriptive_pairs': 21,
            'top1_agreement': top_a[0] == top_b[0],
            'top3_overlap': len(set(top_a[:3]) & set(top_b[:3])),
            'top1_a': CANDIDATES[top_a[0]], 'top1_b': CANDIDATES[top_b[0]]}


def map_metrics(trace, path):
    path = np.asarray(path, float)
    origin = np.asarray(trace[0]['root_state'][:2], float) - path[0]
    post = np.asarray([r['post_root_state'][:2] for r in trace[:40]], float) - origin
    start = np.asarray(trace[0]['root_state'][:2], float) - origin
    endpoint = path[-1]
    end = np.linalg.norm(post - endpoint, axis=1)
    cross = np.linalg.norm(post[:, None, :] - path[None, :, :], axis=2).min(axis=1)
    path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    along = float(_along_path_progress(path, post[-1]) - _along_path_progress(path, start))
    reached = np.flatnonzero(end <= .2)
    yaw = yaw_from_root(trace[39]['post_root_state'])
    tangent = math.atan2(*(path[-1] - path[-2])[::-1])
    heading_error = abs(math.atan2(math.sin(yaw-tangent), math.cos(yaw-tangent)))
    return {'goal_distance_reduction_m': float(np.linalg.norm(start-endpoint) - end[-1]),
            'progress_m': along, 'progress_ratio': along/path_len if path_len else None,
            'endpoint_encounter_0_2m': bool(len(reached)),
            'endpoint_error_m': float(end[-1]), 'minimum_endpoint_error_m': float(end.min()),
            'cross_track_mean_m': float(cross.mean()), 'cross_track_p95_m': float(np.quantile(cross, .95)),
            'realized_path_length_m': float(np.linalg.norm(np.diff(np.vstack([start,post]), axis=0), axis=1).sum()),
            'final_heading_error_rad': float(heading_error),
            'post_reach_departure_0_2s_m': float(end[reached[0]:].max() - end[reached[0]]) if len(reached) else None,
            'first_reach_step': int(reached[0]) if len(reached) else None}


def body_metrics(trace):
    vals = []
    for row in trace[:40]:
        w,x,y,z = map(float,row['root_state'][3:7])
        roll = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch = math.asin(max(-1.,min(1.,2*(w*y-z*x))))
        vals.append(max(abs(roll),abs(pitch)))
    action = np.asarray([r['actor_action'] for r in trace[:40]],float)
    joint = np.asarray([r['joint_position'] for r in trace[:40]],float)
    contact = np.asarray([r['contact_force'] for r in trace[:40]],float)
    effort = np.asarray([r['raw_joint_effort'] for r in trace[:40]],float)
    root = np.asarray([r['post_root_state'][:2] for r in trace[:40]],float)
    cmd = np.asarray([r['follower_command'][0] for r in trace[:40]],float)
    stack = bool(np.linalg.norm(root[-1]-root[-20])<.02 and np.mean(abs(cmd[-20:]))>.1)
    contact_magnitude=np.linalg.norm(contact,axis=-1)
    return {'maximum_roll_pitch_rad':max(vals),'tilt_integral_rad_s':float(np.sum(vals)*.05),
            'maximum_contact_n':float(contact_magnitude.max()),
            'mean_max_contact_per_step_n':float(contact_magnitude.max(axis=1).mean()),
            'contact_body_step_fraction_above_1n':float((contact_magnitude>1.).mean()),
            'action_variation_l2_sum':float(np.linalg.norm(np.diff(action,axis=0),axis=1).sum()),
            'joint_motion_l2_sum':float(np.linalg.norm(np.diff(joint,axis=0),axis=1).sum()),
            'maximum_raw_joint_effort':float(abs(effort).max()),
            'raw_effort_at_3000_fraction':float(np.mean(np.isclose(abs(effort),3000.,atol=1e-4))),
            'stack_proxy':stack,'verified_stack':False,
            'recovery_event':'NA_not_in_trace', 'termination':False,'censored':False}


def drawdown_metrics(trace):
    critic = np.asarray([r['critic_value'] for r in trace[:41]],float)
    index=int(np.argmin(critic))
    recovered=np.flatnonzero(critic[index:]>=critic[0])
    return {'minimum':float(critic.min()),'maximum':float(critic.max()),
            'initial_to_minimum':float(critic.min()-critic[0]),
            'recovery_from_minimum':float(critic[-1]-critic.min()),
            'time_to_minimum_s':index*.05,
            'time_to_recovery_s':float((index+recovered[0])*.05) if len(recovered) else None}


def metrics_for_run(record, gamma):
    sid,pid=record['state_group_id'],record['process_id']
    root=OLD if sid.startswith('g0e-') else NEW
    folder=root/'runs'/pid
    trace=load_trace(folder/'live-0.trace.json.gz')
    validate_trace_index(trace)
    def sequence_sha(values):
        return hashlib.sha256(json.dumps(values,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    reward_sha=sequence_sha([row['reward'] for row in trace])
    critic_sha=sequence_sha([row['critic_value'] for row in trace])
    if reward_sha!=record['reward_sequence_sha256'] or critic_sha!=record['critic_sequence_sha256']:
        raise ValueError(f'{pid}: saved reward or Critic trace SHA mismatch')
    terminal_sha=sequence_sha([row['terminal'] for row in trace])
    path=read(folder/'live-0.json')['map_path']
    ret=return_at_horizon(trace,40,gamma)
    if ret['status']!='ok':raise ValueError(f'{pid}: unexpected termination/bootstrap: {ret["status"]}')
    v=np.asarray([r['critic_value'] for r in trace[:41]],float)
    rew=np.asarray([r['reward'] for r in trace[:40]],float)
    td=rew+gamma*v[1:]-v[:-1]
    command=np.asarray([r['command'] for r in trace[:41]],float)
    command_jump=np.linalg.norm(np.diff(command,axis=0),axis=1)
    return {'process_id':pid,'state_group_id':sid,'terrain':record['terrain'],
            'candidate_id':record['candidate_id'],'repeat_id':record['repeat_id'],
            'trace_sha256':sha(folder/'live-0.trace.json.gz'),
            'reward_sequence_sha256':reward_sha,'critic_sequence_sha256':critic_sha,
            'termination_sequence_sha256':terminal_sha,
            'v0':float(v[0]),'v40':float(v[40]),'R_gamma_40':ret['reward_discounted'],
            'R_1_40':ret['reward_undiscounted'],'G40':ret['g'],'A40':ret['advantage'],
            'deltaV40':float(v[40]-v[0]),'critic_drawdown':drawdown_metrics(trace),
            'map_metrics':map_metrics(trace,path),'body_metrics':body_metrics(trace),
            'six_sec_deltaV':record['outcome']['delta_v'],
            'six_sec_H_step_advantage':record['outcome']['H_step_advantage'],
            'critic_0_40':v.tolist(),'reward_0_39':rew.tolist(),'td_residual_0_39':td.tolist(),
            'command_jump_l2_0_39':command_jump.tolist(),
            'command_39':trace[39]['command'],'command_40':trace[40]['command'],
            'horizon_residual':{str(h):return_at_horizon(trace,h,gamma)['advantage'] for h in (10,20,40)}}


def aggregate_group(rows):
    sid=rows[0]['state_group_id']
    by=defaultdict(list)
    for row in rows:by[row['candidate_id']].append(row)
    if set(by)!=set(CANDIDATES) or len(by['direct_reference'])!=3 or any(len(by[c])!=1 for c in CANDIDATES[1:]):
        raise ValueError(f'{sid}: candidate/repeat composition mismatch')
    paths={
        'G40':('G40',1),'R_gamma_40':('R_gamma_40',1),'R_1_40':('R_1_40',1),
        'A40':('A40',1),'deltaV40':('deltaV40',1),
        'six_sec_deltaV':('six_sec_deltaV',1),
        'six_sec_H_step_advantage':('six_sec_H_step_advantage',1),
        'progress':('map_metrics.progress_m',1),
        'goal_distance_reduction':('map_metrics.goal_distance_reduction_m',1),
        'endpoint_error':('map_metrics.endpoint_error_m',-1),
        'cross_track_p95':('map_metrics.cross_track_p95_m',-1),
        'critic_drawdown':('critic_drawdown.initial_to_minimum',1),
        'tilt':('body_metrics.maximum_roll_pitch_rad',-1),
    }
    def value(row,path):
        result=row
        for part in path.split('.'):result=result[part]
        return result
    medians={}; noise={}; comparison={}
    for name,(path,direction) in paths.items():
        medians[name]={c:float(statistics.median(value(r,path) for r in by[c])) for c in CANDIDATES}
        rep=[value(r,path) for r in by['direct_reference']]
        diffs=[abs(a-b) for a,b in itertools.combinations(rep,2)]
        noise[name]={'direct_repeat_values':rep,'range':max(rep)-min(rep),
                     'sample_std':statistics.stdev(rep),'max_pairwise_absolute_difference':max(diffs),
                     'effect_to_noise_ratio_denominator_zero':max(diffs)==0}
        comparison[name]={c:direction*(medians[name][c]-medians[name]['direct_reference'])
                          for c in CANDIDATES[1:]}
    better_counts={name:sum(x>0 for x in values.values()) for name,values in comparison.items()}
    primary=[medians['G40'][c] for c in CANDIDATES]
    concordance={}
    for name,(path,direction) in paths.items():
        if name=='G40':continue
        concordance[name]=rank_compare(primary,[direction*medians[name][c] for c in CANDIDATES])
    return {'state_group_id':sid,'terrain':rows[0]['terrain'],
            'candidate_values':medians,'direct_repeat_noise':noise,
            'signed_improvement_over_direct':comparison,
            'G40_vs_other_metrics':concordance,
            'candidates_better_than_direct_by_metric':better_counts,
            'candidates_better_than_direct_G40':better_counts['G40'],
            'G40_progress_rank_reversal_candidates':[
                c for c in CANDIDATES[1:] if comparison['G40'][c]*comparison['progress'][c]<0],
            'G40_endpoint_rank_reversal_candidates':[
                c for c in CANDIDATES[1:] if comparison['G40'][c]*comparison['endpoint_error'][c]<0],
            'G40_rank':sorted(CANDIDATES,key=lambda c:(-medians['G40'][c],c))}


def summarize_bellman(rows):
    result={}; byterrain=defaultdict(list); bycandidate=defaultdict(list)
    for row in rows:
        byterrain[row['terrain']].append(row)
        bycandidate[row['candidate_id']].append(row)
    def stats(group):
        td=np.asarray([r['td_residual_0_39'] for r in group],float)
        return {'run_count':len(group),'mean':float(td.mean()),'std':float(td.std()),
                'mean_abs':float(abs(td).mean()),
                'per_step_mean':np.mean(td,axis=0).tolist(),
                'horizon_residual_mean':{str(h):float(np.mean([r['horizon_residual'][str(h)] for r in group]))
                                         for h in (10,20,40)}}
    result['all']=stats(rows)
    result['by_terrain']={k:stats(v) for k,v in byterrain.items()}
    result['by_candidate']={k:stats(v) for k,v in bycandidate.items()}
    result['boundary_command_zero_count']=sum(max(abs(x) for x in r['command_40'])<1e-12 for r in rows)
    early=np.asarray([r['td_residual_0_39'][:30] for r in rows],float)
    late=np.asarray([r['td_residual_0_39'][35:40] for r in rows],float)
    result['path_end_td_mean_abs']={'steps_0_29':float(abs(early).mean()),
                                    'steps_35_39':float(abs(late).mean())}
    jumps=np.asarray([r['command_jump_l2_0_39'] for r in rows],float)
    residual=np.asarray([r['td_residual_0_39'] for r in rows],float)
    result['command_update_diagnostic']={
        'nonzero_jump_threshold_l2':1e-6,
        'changed_transition_count':int(np.count_nonzero(jumps>1e-6)),
        'unchanged_transition_count':int(np.count_nonzero(jumps<=1e-6)),
        'mean_abs_td_after_command_change':float(abs(residual[jumps>1e-6]).mean()) if np.any(jumps>1e-6) else None,
        'mean_abs_td_without_command_change':float(abs(residual[jumps<=1e-6]).mean()) if np.any(jumps<=1e-6) else None,
        'spearman_command_jump_vs_abs_td':float(spearmanr(jumps.ravel(),abs(residual).ravel()).statistic)}
    result['warning']='TD residual is a critic approximation diagnostic, not a Bellman proof; 40-step bootstrap often sees a new command.'
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    provenance=source_audit()
    agent=yaml.load(AGENT.read_text(),Loader=yaml.BaseLoader)
    env=yaml.load(ENV.read_text(),Loader=yaml.BaseLoader)
    gamma=float(agent['algorithm']['gamma']); lam=float(agent['algorithm']['lam'])
    physics=float(env['sim']['dt']); decimation=int(env['decimation'])
    step=physics*decimation
    if gamma!=.99 or step!=.05 or sha(CHECKPOINT)!=provenance['checkpoint_sha256_rechecked']:
        raise ValueError('training run gamma/step/checkpoint mismatch')
    records=input_records()
    results=[metrics_for_run(r,gamma) for r in records]
    if len(results)!=63:raise ValueError('source audit passed but result count differs')
    for source,result in zip(records,results):
        geo=source['geometry']; measured=result['map_metrics']
        for saved,current in [('endpoint_error_at_2s_m','endpoint_error_m'),
                              ('cross_track_0_2s_p95_m','cross_track_p95_m'),
                              ('along_path_progress_0_2s_m','progress_m')]:
            if not math.isclose(geo[saved],measured[current],rel_tol=0,abs_tol=1e-10):
                raise ValueError(f'{result["process_id"]}: corrected geometry mismatch {saved}')
    groups=defaultdict(list)
    for r in results:groups[r['state_group_id']].append(r)
    group_summary={sid:aggregate_group(r) for sid,r in sorted(groups.items())}
    save('source-provenance.json',{'status':'pass','starting_head':'1159cd35a1d8d4c8e0805eadd77b507b119f903b',
        'worktree':str(ROOT),'branch':'exp/eatnav-g40-teacher-review',
        'source_audit':provenance,'training_files_sha256':{str(p):sha(p) for p in (AGENT,ENV,CHECKPOINT)},
        'interpretation_source_sha256':{str(p):sha(p) for p in (ONIX_ENV_SOURCE,ONIX_OBS_SOURCE,REWARD_MANAGER_SOURCE,RL_WRAPPER_SOURCE)},
        'source_manifest_sha256':{str(p):sha(p) for p in (OLD/'original-20-row-manifest.json',NEW/'run-manifest.json')},
        'trace_count':63,'state_group_count':7,'excluded_anchor_only':5,
        'split_identity':'complete-only train StateGroups only; no held-out or validation used',
        'training_config_link':'params/agent.yaml and params/env.yaml share the checkpoint run directory; checkpoint internal iter=10000, infos=None',
        'training_image_caveat':'available isaac-lab-onix image was built 2026-08-06, after 2026-07-24 training; its PPO source is a reference, not proven byte-identical to training code'})
    save('rl-return-definition.json',{
        'gamma':gamma,'gae_lambda':lam,'gamma_power_40':gamma**40,
        'physics_step_sec':physics,'physics_hz':1/physics,'control_decimation':decimation,
        'policy_step_sec':step,'policy_hz':1/step,'forty_steps_sec':40*step,
        'discount_effective_horizon_sec':step/(1-gamma),
        'discount_half_life_sec':step*math.log(.5)/math.log(gamma),
        'discount_weight_1s':gamma**20,'discount_weight_2s':gamma**40,'discount_weight_6s':gamma**120,
        'episode_timeout_sec':float(env['episode_length_s']),'is_finite_horizon':env['is_finite_horizon'],
        'reward_policy_step':True,'reward_dt_already_applied':True,
        'reward_manager_source':'/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:129-150',
        'onix_env_source':'/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/envs.py:22-67',
        'ppo_config_source':str(AGENT)+':44-52',
        'source_sha256':{str(p):sha(p) for p in (AGENT,ENV,ONIX_ENV_SOURCE,REWARD_MANAGER_SOURCE,RL_WRAPPER_SOURCE)},
        'value_normalization':agent['empirical_normalization'],
        'reward_normalization':'not configured in saved agent/env yaml; no RND',
        'reward_clipping':'no aggregate clipping seen in saved config',
        'value_loss_clipped':agent['algorithm']['use_clipped_value_loss'],
        'value_clip_parameter':float(agent['algorithm']['clip_param']),
        'ppo_source_qualification':'2026-08-06 image is later than training; timeout correction/GAE code version unproven',
        'rl_training_target':'GAE(lambda=0.95) in later available PPO source; G40 is a diagnostic n-step return, not identical to PPO training target'})
    obs=env['observations']['critic']
    obs_terms=[k for k,v in obs.items() if isinstance(v,dict) and 'func' in v]
    save('critic-input-contract.json',{'critic_dim':267,'actor_dim':22,'terms':obs_terms,
        'includes_current_command':'velocity_commands' in obs_terms,'includes_goal':False,
        'includes_candidate_path':False,'includes_dem':'height_scan' in obs_terms,
        'includes_privileged_state':'base_lin_vel and height_scan and contact_binary',
        'includes_contact':'roller_contact_binary' in obs_terms,
        'includes_effort':'joint_effort' in obs_terms,'includes_joint_state':True,
        'phase':'pre-step; follower command applied before observation, then Actor/Critic evaluated together',
        'interpretation':'current state and command RL return estimator, not a whole-Path outcome predictor',
        'config_sha256':sha(ENV)})
    save('trace-index-contract.json',{'status':'pass','row_k':'pre-step V_k, actor input/action/command and post-step reward r_k, post_root_state',
        'V0':'trace[0].critic_value','reward_interval':'trace[0:40].reward','V40':'trace[40].critic_value',
        'path_interval':'control steps 0-39, 0.00-2.00 s',
        'v40_command_is_after_command_manager_compute':True,
        'v40_zero_command_run_count':sum(max(abs(x) for x in r['command_40'])<1e-12 for r in results),
        'checked_root_boundaries':63*119,'mismatched_root_boundaries':0,'timestamp_mismatch_count':0,
        'source_lines':'scripts/eat_nav_e1_sim.py:485-530; onix envs.py:22-67',
        'diagram':'t_k: follower command -> obs_k -> V_k/Actor_k -> env.step(action_k) -> r_k/post_state_(k+1) -> command_manager/obs_(k+1); row k+1 holds V_(k+1)'})
    with gzip.open(OUT/'g40-per-candidate.jsonl.gz','wt') as f:
        for r in results:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    save('g40-stategroup-summary.json',{'statistical_unit':'StateGroup','groups':group_summary})
    save('direct-repeat-noise-2s.json',{'groups':{s:g['direct_repeat_noise'] for s,g in group_summary.items()},
        'zero_denominator_rule':'ratio NA; no epsilon inserted'})
    comparisons=defaultdict(list)
    for group in group_summary.values():
        for metric,item in group['G40_vs_other_metrics'].items():comparisons[metric].append(item)
    save('teacher-rank-concordance-2s.json',{'groups':{s:g['G40_vs_other_metrics'] for s,g in group_summary.items()},
        'macro':{m:{'spearman_mean':float(np.mean([x['spearman'] for x in v if x['spearman'] is not None]))
                        if any(x['spearman'] is not None for x in v) else None,
                     'top1_agreement_groups':sum(x['top1_agreement'] for x in v),
                     'mean_top3_overlap':float(np.mean([x['top3_overlap'] for x in v])),
                     'pairwise_agreement_mean':float(np.mean([x['pairwise_agreement'] for x in v if x['pairwise_agreement'] is not None]))}
                 for m,v in comparisons.items()},
        'no_pair_independence':'21 candidate pairs per group are descriptive only'})
    save('bellman-consistency.json',summarize_bellman(results))
    effort=np.asarray([r['body_metrics']['maximum_raw_joint_effort'] for r in results])
    save('effort-saturation-audit.json',{'run_count':63,'max_raw_effort_3000_count':int(np.sum(np.isclose(effort,3000))),
        'range_max_raw_effort':[float(effort.min()),float(effort.max())],
        'actuator_limit_source':str(ENV)+':227,264 effort_limit_sim=3000',
        'observation_transform_source':'onix observations.py:156-170; track joint signed sum, then clip +/-100 and scale 0.01 in env.yaml',
        'raw_tensor_source':'robot.data.applied_torque pre-step; no logging-side clip in E1 simulator',
        'physical_saturation_proven':False,'ranking_teacher_allowed':False,
        'reason':'3000 is configured per-joint actuator limit; raw tensor equality alone does not distinguish requested clamp from achieved physical torque'})
    reward_terms=[]
    for name,cfg in env['rewards'].items():
        weight=float(cfg['weight'])
        reward_terms.append({'name':name,'function':cfg['func'],'weight':weight,
                             'sign':'positive' if weight>0 else 'negative' if weight<0 else 'zero',
                             'effective_weight_per_policy_step':weight*step,
                             'recorded_component_values':False,'discounted_component_contribution':None,
                             'candidate_rank_contribution':None,'terrain_dominance':None})
    save('reward-component-audit.json',{'config_sha256':sha(ENV),'reward_total_recorded':True,
        'reward_component_trace_recorded':False,'exact_decomposition_possible':False,
        'reason':'RewardManager component tensors and previous action/acceleration/sensor state at reward evaluation were not recorded; no unsupported estimates',
        'terms':reward_terms,'command_tracking_dominates':'NA_unidentifiable_from_scalar_total',
        'progress_absence_can_score_high':'possible_by_reward_definition_no_explicit_goal_progress_term'})
    save('termination-bootstrap-contract.json',{'observed_63_traces':{'physical_termination':0,'timeout':0,'censoring':0},
        'physical_termination':'sum rewards through terminal transition; no V after physical termination',
        'time_limit_truncation':'preserve separate flag and pre-reset terminal observation/value; exclude if unavailable',
        'training_time_limit_note':'later available PPO source adds gamma*pre-step V to timeout reward; exact 2026-07 training source unverified',
        'censored_timeout':'not equivalent to physical failure; mask or separately bootstrap only with verified pre-reset value',
        'infrastructure_failure':'exclude from robot outcome',
        'measurement_invalid':'exclude from robot outcome',
        'manual_stop':'exclude or censor, never failure without physical evidence',
        'required_future_fields':['terminated','truncated','termination_reason','pre_reset_terminal_critic','last_valid_step','censor_reason']})
    return results,group_summary
