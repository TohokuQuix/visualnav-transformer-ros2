"""Numerical contracts for the approved fall/stall targets v2 specification."""
import copy
import importlib
import importlib.util
import math

import pytest


PATH = [[0., 0.], [.75, 0.]]


def build(rows, path=PATH):
    name = 'visualnav_transformer.eat_nav.fall_stall_targets'
    assert importlib.util.find_spec(name) is not None, 'v2 teacher implementation is missing'
    return importlib.import_module(name).build_targets(rows, path)


def state(x=0., tilt=0., yaw=0., omega=(0., 0., 0.)):
    a, b = math.radians(tilt)/2, yaw/2
    # world yaw times body roll, quaternion wxyz.
    return [x, 0., 0., math.cos(b)*math.cos(a), math.cos(b)*math.sin(a),
            math.sin(b)*math.sin(a), math.sin(b)*math.cos(a), 0., 0., 0., *omega]


def trace(count=200, *, distance_per_step=0., vx=.3, wz=.02, body_wz=.03):
    return [{'control_step':i, 'logical_sim_time_ns':1_000_000_000+i*50_000_000,
             'candidate_cycle':i//40, 'candidate_replan':bool(i and i%40==0),
             'root_state':state(i*distance_per_step),
             'post_root_state':state((i+1)*distance_per_step),
             'follower_command':[vx,0.,wz], 'body_angular_velocity_radps':[0.,0.,body_wz],
             'terminal':None} for i in range(count)]


def target(result, horizon, event):
    return result['horizons'][f'{horizon}s']['events'][event]


def test_initial_horizon_includes_first_and_last_post_action_sample():
    rows=trace()
    rows[0]['post_root_state']=state(tilt=86.)
    result=build(rows)
    assert result['schema']=='eatnav_fall_stall_targets_v2'
    assert target(result,1,'tip_over')['label']==1
    rows=trace()
    rows[19]['post_root_state']=state(tilt=86.)
    assert target(build(rows),1,'tip_over')['label']==1
    rows[19]['post_root_state']=state()
    rows[20]['post_root_state']=state(tilt=86.)
    assert target(build(rows),1,'tip_over')['label']==0


def test_near_fall_confirmation_does_not_look_past_horizon():
    rows=trace()
    for i in range(17,22):rows[i]['post_root_state']=state(tilt=60.)
    result=build(rows)
    assert target(result,1,'near_fall')['label']==0
    assert target(result,2,'near_fall')['label']==1
    event=next(e for e in result['events'] if e['event']=='near_fall')
    assert (event['onset_step'],event['confirmation_step'])==(17,21)


@pytest.mark.parametrize('samples,expected',[(4,0),(5,1)])
def test_near_fall_requires_five_samples_with_normalized_quaternion(samples,expected):
    rows=trace()
    for i in range(samples):
        rows[i]['post_root_state']=state(tilt=60.)
        rows[i]['post_root_state'][3:7]=[2*x for x in rows[i]['post_root_state'][3:7]]
    assert target(build(rows),1,'near_fall')['label']==expected


def test_partial_observation_masks_each_event_separately():
    rows=trace(30)
    rows[-1]['terminal']={'termination_reasons':['tilt_over'],
                          'pre_reset_state':{'root_state':state(tilt=86.)}}
    result=build(rows)
    assert target(result,10,'tip_over')['label']==1
    assert target(result,10,'tip_over')['target_mask'] is True
    assert target(result,10,'linear_stall')['label'] is None
    assert target(result,10,'linear_stall')['target_mask'] is False
    assert result['event_free_10s']['label']==0
    assert result['horizons']['10s']['progress']['value_m'] is None


def test_parallel_censor_is_not_a_physical_event_or_safe_future():
    rows=trace(100,distance_per_step=.01)
    rows[-1]['terminal']={'termination_reason':'parallel_env_censored_by_other_env'}
    result=build(rows)
    assert target(result,5,'tip_over')['label']==0
    assert target(result,10,'tip_over')['label'] is None
    assert result['event_free_10s']['label'] is None


def test_terminal_snapshot_wins_over_reset_posture_and_velocity():
    rows=trace(40,body_wz=.001)
    rows[-1]['post_root_state']=state(tilt=90.)
    rows[-1]['body_angular_velocity_radps']=[0.,0.,99.]
    rows[-1]['terminal']={'termination_reason':'timeout',
                          'pre_reset_state':{'root_state':state(tilt=0.,omega=(0.,0.,.001))}}
    result=build(rows)
    assert target(result,2,'tip_over')['label']==0
    assert result['cycles'][0]['turn_stall']['label']==1


def test_terminal_world_omega_is_rotated_to_body_frame():
    rows=trace(40,body_wz=.001)
    # With a 90 degree roll, world z omega is body y, not body z.
    rows[-1]['terminal']={'termination_reason':'timeout',
                          'pre_reset_state':{'root_state':state(tilt=90.,omega=(0.,0.,1.))}}
    assert build(rows)['cycles'][0]['turn_stall']['label']==1


def test_terminal_snapshot_missing_keeps_signal_unknown():
    rows=trace(40,body_wz=.001)
    rows[-1]['terminal']={'termination_reason':'pre_reset_snapshot_missing'}
    result=build(rows)
    assert target(result,2,'tip_over')['label'] is None
    assert result['cycles'][0]['turn_stall']['label'] is None
    assert result['cycles'][0]['linear_stall']['label'] is None


def test_linear_stall_uses_last_interval_and_continuous_best_progress():
    rows=trace(40)
    rows[-1]['post_root_state']=state(.06)
    result=build(rows)
    cycle=result['cycles'][0]
    assert cycle['commanded_distance_m']==pytest.approx(.6)
    assert cycle['actual_distance_m']==pytest.approx(.06)
    assert cycle['best_progress_m']==pytest.approx(.06)
    assert cycle['linear_stall']['label']==0


def test_out_and_back_motion_does_not_erase_best_progress():
    rows=trace(40)
    for i in range(40):
        before=.06 if i==1 else 0.
        after=.06 if i==0 else 0.
        rows[i]['root_state']=state(before)
        rows[i]['post_root_state']=state(after)
    cycle=build(rows)['cycles'][0]
    assert cycle['actual_distance_m']==pytest.approx(.12)
    assert cycle['best_progress_m']==pytest.approx(.06)
    assert cycle['linear_stall']['label']==0


def test_stall_thresholds_include_exact_boundaries():
    rows=trace(40,vx=.1)
    for row in rows:row['post_root_state']=state(.05)
    # Keep physically contiguous positions: move only in the last interval.
    for row in rows[:-1]:row['post_root_state']=state()
    cycle=build(rows)['cycles'][0]
    assert cycle['commanded_distance_m']==pytest.approx(.2)
    assert cycle['actual_distance_m']==pytest.approx(.05)
    assert cycle['linear_stall']['label']==1


@pytest.mark.parametrize('missing', ['follower_command','post_root_state'])
def test_missing_linear_signal_is_unknown_not_zero_filled(missing):
    rows=trace(40)
    del rows[7][missing]
    assert build(rows)['cycles'][0]['linear_stall']['label'] is None


def test_turn_stall_can_coexist_with_translation_and_requires_continuous_response():
    rows=trace(40,distance_per_step=.01,body_wz=.001)
    result=build(rows)
    assert result['cycles'][0]['turn_stall']['label']==1
    assert result['cycles'][0]['linear_stall']['label']==0
    rows[10]['follower_command'][2]=0.
    assert build(rows)['cycles'][0]['turn_stall']['label']==0
    rows[10]['follower_command'][2]=.02
    rows[10]['body_angular_velocity_radps'][2]=.011
    assert build(rows)['cycles'][0]['turn_stall']['label']==0


def test_missing_angular_signal_and_partial_cycle_are_unknown():
    rows=trace(40)
    rows[10]['body_angular_velocity_radps']=None
    assert build(rows)['cycles'][0]['turn_stall']['label'] is None
    assert build(trace(39))['cycles'][0]['linear_stall']['label'] is None


def test_replanning_without_motion_does_not_create_progress_or_recovery():
    result=build(trace())
    assert result['horizons']['10s']['progress']['value_m']==pytest.approx(0.)
    assert result['recoveries'][0]['label']==0


def test_near_fall_recovery_requires_five_second_followup_and_progress():
    rows=trace(distance_per_step=.005)
    for i in range(5):rows[i]['post_root_state']=state((i+1)*.005,tilt=65.)
    recovery=next(r for r in build(rows)['recoveries'] if r['event']=='near_fall')
    assert recovery['label']==1
    assert recovery['progress_gain_m']>=.1
    recovery=next(r for r in build(rows[:90])['recoveries'] if r['event']=='near_fall')
    assert recovery['label'] is None


def test_progress_keeps_backward_motion_signed_across_cycles():
    rows=trace(distance_per_step=.005)
    rows[10]['post_root_state']=state(.04)
    rows[11]['root_state']=state(.04)
    diagnostics=build(rows)['diagnostics']
    assert diagnostics['progress_m'][10]<diagnostics['progress_m'][9]
    assert diagnostics['progress_m'][40]-diagnostics['progress_m'][39]==pytest.approx(.005)


@pytest.mark.parametrize('mutator',[
    lambda rows:rows[3].update(control_step=4),
    lambda rows:rows[3].update(logical_sim_time_ns=0),
    lambda rows:rows[40].update(candidate_cycle=0),
])
def test_broken_protocol_sequence_is_rejected(mutator):
    rows=trace();mutator(rows)
    with pytest.raises(ValueError):build(rows)


def test_builder_does_not_mutate_source_rows():
    rows=trace(); original=copy.deepcopy(rows)
    build(rows)
    assert rows==original


def test_large_finite_quaternion_is_normalized_without_overflow():
    rows=trace()
    rows[0]['post_root_state']=state(tilt=86.)
    rows[0]['post_root_state'][3:7]=[v*1e200 for v in rows[0]['post_root_state'][3:7]]
    assert target(build(rows),1,'tip_over')['label']==1


def test_zero_quaternion_is_unknown_and_invalid_path_does_not_produce_stall():
    rows=trace(40)
    rows[1]['post_root_state'][3:7]=[0.,0.,0.,0.]
    assert target(build(rows),2,'tip_over')['label'] is None
    result=build(trace(40),path=[[0.,0.],[0.,0.]])
    assert result['cycles'][0]['linear_stall']['label'] is None
    assert result['horizons']['2s']['progress']['target_mask'] is False


def test_severity_and_discrete_run_duration_are_explicit_metadata():
    rows=trace()
    for i in range(5):rows[i]['post_root_state']=state(tilt=65.)
    rows[2]['post_root_state']=state(tilt=70.)
    result=build(rows)
    assert result['max_tilt_deg']==pytest.approx(70.)
    event=next(e for e in result['events'] if e['event']=='near_fall')
    assert event['consecutive_samples']==5
    assert event['duration_control_sec']==pytest.approx(.25)
    assert event['sample_span_sec']==pytest.approx(.20)


def test_recovery_after_old_missing_sample_uses_known_followup_increments():
    rows=trace(distance_per_step=.005)
    rows[1]['post_root_state']=None
    for i in range(20,25):rows[i]['post_root_state']=state((i+1)*.005,tilt=65.)
    recovery=next(r for r in build(rows)['recoveries'] if r['event']=='near_fall')
    assert recovery['label']==1
