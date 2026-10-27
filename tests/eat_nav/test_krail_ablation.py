import numpy as np
import pytest
from importlib import import_module


def api():return import_module('eat_nav_krail_ablation')
def inputs():return np.zeros((4,4)),np.ones((4,4)),[4.,0.]

@pytest.mark.parametrize('mode',['A0','A1'])
def test_disabled_fall_heads_cannot_change_selection_stop_or_resume(mode):
    p,q,g=inputs();a=api().AblationGuard(mode);b=api().AblationGuard(mode)
    for step in (0,40,80,120):
        p[:,:2]=.99
        actual=a.decide(p,q,g,step,0.)
        expected=b.decide(np.zeros_like(p),q,g,step,0.)
        assert actual['selected_index']==expected['selected_index']
        assert actual['scores']==expected['scores']
        assert not actual['guard_stopped']


def test_a0_ignores_stall_and_resumes_after_common_tilt_stop():
    p,q,g=inputs();p[:]=1.;q[:]=0.;guard=api().AblationGuard('A0')
    assert guard.decide(p,q,g,0,0.)['selected_index']==0
    guard.observe_tilt(50.,1)
    assert guard.decide(p,q,g,40,0.)['selected_index'] is None
    assert guard.decide(p,q,g,80,0.)['selected_index'] is None
    assert guard.decide(p,q,g,120,0.)['selected_index']==0


def test_mixed_prediction_exclusions_and_tilt_preserve_episode_causes():
    p,q,g=inputs();q[:,3]=0.;p[:2,0]=.9;p[2:,2]=.99
    guard=api().AblationGuard('A2');d=guard.decide(p,q,g,0,0.)
    assert d['selected_index'] is None
    assert set(d['episode_causes'])=={'tip','stall'}
    guard.observe_tilt(50.,1)
    assert set(guard.episode_causes)=={'tip','stall','measured_tilt'}
    d=guard.decide(np.zeros_like(p),q,g,40,30.)
    assert 'no_release_candidate' not in d['release_blockers']
    restored=api().AblationGuard.from_state(guard.snapshot())
    assert restored.snapshot()==guard.snapshot()


def test_final_step_goal_entry_and_failure_are_separate():
    f=api().terminal_outcome
    assert f(.2,False,1800,1800)==dict(goal_entered=True,normal_success=True,reason='goal_reached')
    assert f(.2,True,1800,1800)==dict(goal_entered=True,normal_success=False,reason='natural_termination')


def test_exposure_partitions_do_not_double_count_stop():
    rows=[dict(guard_stopped=(i<3),follower_command=[0.,0.,0.]) for i in range(43)]
    totals=api().exposure_partition(rows,[None,False])
    assert totals==dict(stop_steps=3,evaluable_running_steps=3,unknown_running_steps=37,total_steps=43)


def test_complete_low_command_window_is_known_negative():
    rows=[dict(guard_stopped=False,follower_command=[0.,0.,0.],
        root_state=[0.,0.,0.,1.]+[0.]*9,post_root_state=[0.,0.,0.,1.]+[0.]*9,selected_candidate='forward_direct') for _ in range(40)]
    result=api().window_labels(rows,[[0.,0.],[1.,0.]])
    assert result==dict(linear=False,turn=False)


def test_suite_fixed_order_retry_and_wall_bounds():
    from eat_nav_krail_ablation_run import suite_manifest
    m=suite_manifest()
    assert len(m['runs'])==9 and m['maximum_attempts']==18
    assert m['steps']==1800 and m['container_timeout_s']==570
    assert [(x['seed'],x['condition']) for x in m['runs']]==api().ORDER
    assert m['reserved_test_opened'] is False


def test_score_metadata_names_only_active_weights():
    p,q,g=inputs();d=api().AblationGuard('A0').decide(p,q,g,0,0.)
    assert d['risk_weights']==[0.,0.,0.,0.]


def test_terminal_snapshot_is_used_for_last_interval_label():
    rows=[dict(guard_stopped=False,follower_command=[0.,0.,0.],root_state=[0.,0.,0.,1.]+[0.]*9,
        post_root_state=[0.,0.,0.,1.]+[0.]*9,selected_candidate='forward_direct') for _ in range(40)]
    rows[-1]['terminal']=dict(root_state=rows[-1]['post_root_state'],termination_reasons=['tilt_over'])
    assert api().window_labels(rows,[[0.,0.],[1.,0.]])==dict(linear=False,turn=False)


def test_release_blockers_update_between_decisions_without_resume_hits():
    p,q,g=inputs();guard=api().AblationGuard('A2');guard.decide(p,q,g,0,0.)
    guard.observe_tilt(50.,1)
    assert set(guard.release_blockers)=={'tilt','minimum_stop','confirmation'}
    guard.observe_tilt(30.,41)
    assert guard.release_blockers==['confirmation'] and guard.resume_hits==0


def test_suite_rejects_source_mutation(tmp_path):
    from eat_nav_krail_ablation_run import verify_suite_sources
    import hashlib
    p=tmp_path/'source.py';p.write_text('fixed')
    manifest={'source_sha256':{'source.py':hashlib.sha256(b'fixed').hexdigest()}}
    verify_suite_sources(manifest,tmp_path)
    p.write_text('changed')
    with pytest.raises(ValueError):verify_suite_sources(manifest,tmp_path)


def test_interrupted_or_short_horizon_is_not_eligible_comparison():
    from eat_nav_krail_ablation_audit import eligible_outcome
    assert not eligible_outcome(dict(stop_reason='window_closed',control_steps=20))
    assert not eligible_outcome(dict(stop_reason='horizon_complete',control_steps=1799))
    assert eligible_outcome(dict(stop_reason='goal_reached',control_steps=20))


def test_observed_event_time_uses_post_state_confirmation():
    from eat_nav_krail_ablation_audit import observed_events
    rows=[dict(control_step=i,post_root_state=[0.,0.,0.,2**-.5,2**-.5,0.,0.]+[0.]*6,terminal=None) for i in range(5)]
    events=observed_events(rows)
    near=next(e for e in events if e['event']=='near_fall')
    assert near['onset_time_s']==.05 and near['confirmation_time_s']==.25


def test_stop_diagnostics_are_json_serializable(tmp_path):
    import json
    from eat_nav_krail_ablation_audit import stop_diagnostics
    output=tmp_path/'sim-output';output.mkdir()
    (output/'selection-0000.json').write_text(json.dumps(dict(probabilities=[[0.]*4]*4,progress_m=[[0.]*4]*4)))
    row=dict(control_step=0,guard_stopped=True,episode_causes=['measured_tilt'],trip_causes=['measured_tilt'],
        release_blockers=['tilt'],root_state=[0.,0.,0.,1.]+[0.]*9,post_root_state=[0.,0.,0.,1.]+[0.]*9,
        follower_command=[0.,0.,0.],actor_command=[.1,0.,0.])
    json.dumps(stop_diagnostics([row],tmp_path,'A2'))


def test_pairing_compares_object_member_bytes_without_unpickling(tmp_path):
    import json
    from eat_nav_krail_ablation_audit import verify_initial_pair
    for name in ['a','b']:
        folder=tmp_path/name;folder.mkdir()
        (folder/'startup-ready.json').write_text(json.dumps(dict(initial_pose=[0.,0.,0.])))
        np.savez(folder/'input-0000.npz',dem=np.zeros(2))
        np.savez(folder/'initial-state.npz',actor_observation={'policy':np.ones((1,2))},root_state=np.zeros(13))
    result=verify_initial_pair(tmp_path/'a',tmp_path/'b')
    assert result['matches'] and result['object_members']['actor_observation.npy']['method']=='exact_uncompressed_bytes_no_deserialization'
    np.savez(tmp_path/'b/initial-state.npz',actor_observation={'policy':np.zeros((1,2))},root_state=np.zeros(13))
    assert not verify_initial_pair(tmp_path/'a',tmp_path/'b')['matches']


def test_goal_does_not_hide_pending_near_fall():
    from eat_nav_krail_ablation_audit import goal_finish_diagnostics
    import math
    rows=[]
    for angle in [61.,67.,74.,83.56]:
        q=[math.cos(math.radians(angle)/2),math.sin(math.radians(angle)/2),0.,0.]
        rows.append(dict(post_root_state=[0.,0.,0.]+q+[0.]*6,guard_stopped=True))
    d=goal_finish_diagnostics(rows,dict(goal_entered=True,stop_reason='goal_reached'))
    assert d['near_fall_pending_samples']==4 and d['high_tilt_at_goal']
    assert d['post_goal_physics_observed'] is False
