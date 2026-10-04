import numpy as np
import pytest

from eat_nav_e1_coordinate_addendum import (
    derive_origin, transform_base_path, map_world_distance_equivalence, geometry_metrics,
    pairwise_rms, first_divergence, signal_distance, audit_inputs, extract_signals,
    corrected_geometry, recompute_pcr, corrected_outcomes,
    stagewise_analysis,
    summarize_outcomes, make_decision, generate_addendum,
)


def test_origin_is_derived_from_independent_world_and_map_anchors():
    assert derive_origin([-19.3,-20.1],[.7,-.1])==[-20.,-20.]
    assert derive_origin([19.8,28.8],[-.2,.8])==[20.,28.]


def test_robot_local_path_transform_uses_anchor_yaw_and_no_extra_origin_rotation():
    base=np.array([[0.,0.],[1.,0.],[1.,1.]])
    actual=transform_base_path(base,[2.,3.,np.pi/2])
    assert np.allclose(actual,[[2.,3.],[2.,4.],[1.,4.]],atol=1e-12)


def test_map_world_geometry_is_translation_equivalent():
    path=np.array([[0.,0.],[1.,0.]])
    root_map=np.array([[0.,0.],[.5,.1],[1.,.2]])
    result=map_world_distance_equivalence(path,root_map+[-20.,-28.],[-20.,-28.])
    assert result['cross_track_max_abs_m']<1e-12
    assert result['endpoint_max_abs_m']<1e-12


def test_two_second_metrics_do_not_call_later_motion_path_tracking():
    path=np.array([[0.,0.],[1.,0.]])
    root=np.zeros((120,2));root[:40,0]=np.linspace(0,1,40);root[40:,0]=2.
    result=geometry_metrics(path,root,[0.,0.],dt=.05)
    # The frozen E1 cross-track formula uses nearest saved waypoint.
    assert result['cross_track_0_2s_p95_m']==pytest.approx(.4628205128205128)
    assert result['reach_within_0_2s_0_2m'] is True
    assert result['first_reach_step']==32
    assert result['post_reach_departure_m']==pytest.approx(1.)
    assert result['endpoint_error_at_2s_m']==0.
    assert result['cross_track_0_6s_diagnostic_p95_m']==1.


def test_pairwise_rms_is_invariant_under_common_translation():
    a=np.array([[0.,0.],[1.,0.]])
    b=np.array([[0.,0.],[1.,.2]])
    assert pairwise_rms(a,b)==pytest.approx(pairwise_rms(a+[20.,28.],b+[20.,28.]))


def test_first_divergence_reports_preceding_step_and_frozen_crossing():
    a=np.zeros((4,2));b=a.copy();b[2,0]=.2
    result=first_divergence(a,b,threshold=.1)
    assert result['first_nonzero_step']==2
    assert result['first_frozen_threshold_step']==2
    assert result['preceding_step_max_abs']==0.


def test_signal_distance_marks_missing_values_na():
    assert signal_distance(None,np.zeros((2,3)))['status']=='NA'


def test_audit_recovers_exact_frozen_18_run_identity():
    from pathlib import Path
    root=Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004')
    result=audit_inputs(root)
    assert len(result['rows'])==18
    assert result['authoritative_origins']['g0e-train-000']['xy_world_m']==[-20.,-20.]
    assert result['authoritative_origins']['g0e-train-001']['xy_world_m']==[20.,28.]
    assert result['source_provenance']['old_result_sha256']['final-gate-decision.json']


def test_actor_command_slice_preserves_raw_stage_separation():
    row={'follower_command':[.2,0.,.1], 'command':[.2,0.,.1],
        'pre_noise':[0.]*6+[.2,0.,.1]+[0.]*13,
        'post_noise':[0.]*6+[.2,0.,.1]+[0.]*13,
        'actor_input':[0.]*6+[.1,0.,0.]+[0.]*13,
        'actor_latent':[0.]*256,'actor_action':[0.]*6,
        'processed_action':{'flipper_pos':[0.]*4,'right_track_vel':[0.],'left_track_vel':[0.]},
        'joint_target':[0.]*52,'joint_position':[0.]*52,'joint_velocity':[0.]*52,
        'raw_joint_effort':[0.]*52,'root_state':[0.]*13,'post_root_state':[0.]*13,
        'contact_force':[[0.]*3]*60,'dem':{'transformed':[[[0.]*20]*20]*2},
        'critic_value':1.,'reward':0.,'terminal':None}
    series=extract_signals([row])
    assert series['command_manager'][0].tolist()==[.2,0.,.1]
    assert series['actor_pre_noise_command'][0].tolist()==[.2,0.,.1]
    assert series['actor_post_delay_command'][0].tolist()==[.1,0.,0.]
    assert series['actor_action'].shape==(1,6)


def test_corrected_geometry_and_pcr_preserve_frozen_modes():
    audit=audit_inputs()
    geometry=corrected_geometry(audit)
    pcr=recompute_pcr(audit)
    assert len(geometry['runs'])==18
    assert [pcr[x]['command_modes'] for x in ('g0e-train-000','g0e-train-001')]==[7,7]
    assert [pcr[x]['realized_modes'] for x in ('g0e-train-000','g0e-train-001')]==[6,1]
    assert all(x['map_world_distance_equivalence']['cross_track_max_abs_m']<1e-10
               for x in geometry['runs'].values())


def test_corrected_outcome_preserves_non_geometric_values_exactly():
    audit=audit_inputs()
    geometry=corrected_geometry(audit)
    outcomes=corrected_outcomes(audit,geometry)
    for row in outcomes['runs'].values():
        assert row['old']['delta_v']==row['corrected']['delta_v']
        assert row['old']['H_step_advantage']==row['corrected']['H_step_advantage']
        assert row['old']['maximum_contact_n']==row['corrected']['maximum_contact_n']


def test_stagewise_classification_uses_recorded_actor_and_body_signals():
    audit=audit_inputs()
    pcr=recompute_pcr(audit)
    result=stagewise_analysis(audit,pcr)
    first=result['groups']['g0e-train-000']
    second=result['groups']['g0e-train-001']
    assert first['collapse_stage']=='no_observed_collapse'
    assert second['collapse_stage']=='xy_collapse_with_embodiment_difference'
    for stage in ('follower_command','command_manager','actor_post_delay_command',
                  'actor_action','processed_action','root_linear_angular_velocity'):
        assert second['candidate_exceeds_duplicate'][stage]
    assert second['realized_modes']==1


def test_outcome_diversity_uses_state_groups_and_frozen_critic_threshold():
    audit=audit_inputs();geometry=corrected_geometry(audit)
    outcomes=corrected_outcomes(audit,geometry)
    pcr=recompute_pcr(audit);stage=stagewise_analysis(audit,pcr)
    summary=summarize_outcomes(audit,geometry,pcr,outcomes)
    decision=make_decision(audit,geometry,pcr,stage,summary)
    assert summary['statistical_unit']=='StateGroup'
    assert all(summary['groups'][sid]['delta_v_candidate_exceeds_duplicate']
               for sid in ('g0e-train-000','g0e-train-001'))
    assert decision['original_frozen_gate']=='inconclusive'
    assert decision['coordinate_frame']=='pass'
    assert decision['g0e-train-001_collapse_stage']=='xy_collapse_with_embodiment_difference'
    assert decision['evaluator_collection']=='not_ready'


def test_addendum_writes_only_new_output_root(tmp_path):
    from pathlib import Path
    old=Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004')
    before=(old/'final-gate-decision.json').read_bytes()
    result=generate_addendum(tmp_path/'new-addendum',render_figures=False)
    assert result['original-vs-corrected-decision.json']['original_frozen_gate']=='inconclusive'
    assert (tmp_path/'new-addendum/source-provenance.json').is_file()
    assert (tmp_path/'new-addendum/artifact-sha-ledger.json').is_file()
    assert before==(old/'final-gate-decision.json').read_bytes()


def test_artifact_ledger_excludes_mutable_runner_log(tmp_path):
    from pathlib import Path
    import hashlib, json
    output=tmp_path/'new-addendum'
    output.mkdir()
    (output/'runner.log').write_text('started\n')
    generate_addendum(output,render_figures=False)
    (output/'runner.log').write_text('started\ncompleted\n')
    ledger=json.loads((output/'artifact-sha-ledger.json').read_text())
    assert 'runner.log' in ledger['excluded_mutable']
    assert all(item['path']!='runner.log' for item in ledger['entries'])
    assert all(hashlib.sha256((output/item['path']).read_bytes()).hexdigest()==item['sha256']
               for item in ledger['entries'])


def test_all_requested_figures_render_from_recorded_signals(tmp_path):
    generate_addendum(tmp_path/'figured',render_figures=True)
    images=list((tmp_path/'figured/figures').glob('*.png'))
    assert len(images)==12
    assert all(path.stat().st_size>1000 for path in images)
