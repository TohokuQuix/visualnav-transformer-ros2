"""Collection contracts are CPU-only; no Isaac import during planning."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import eat_nav_moving_pose2s_collect as collect


def test_frozen_deadline_parents_are_independent_and_split_before_outcomes():
    p = collect.prepare_collection(parents_per_state=10, excluded_seeds={123})
    assert p['planned_branches'] == 980
    assert len({g['seed'] for g in p['parents']}) == 70
    for state in collect.STATES:
        groups = [g for g in p['parents'] if g['state'] == state]
        assert [g['role'] for g in groups].count('train') == 6
        assert [g['role'] for g in groups].count('validation') == 2
        assert [g['role'] for g in groups].count('audit-eval') == 2
    assert collect.prepare_collection(parents_per_state=10) == p
    assert p['physics_cache_restorable'] is False


def state(speed=.15, wz=0., vz=.03, rise=.03, boundary=.35, tilt=0.):
    return dict(speed_mps=speed, body_wz_radps=wz, mean_world_vz_mps=vz,
                tilt_rad=tilt, front_rise_m=rise, forward_boundary_distance_m=boundary,
                terrain_local_slope=rise/.25, low_gradient_dem=True,
                valid_dem_fraction=1., dem_history_seconds=1., command=[.3,0,0])


def test_qualifications_use_measured_terrain_and_velocity_not_elapsed_time():
    assert collect.qualify_state('moving_flat', state())['qualified']
    assert not collect.qualify_state('turning', state())['qualified']
    assert collect.qualify_state('turning', state(wz=.3))['qualified']
    assert collect.qualify_state('uphill_before', state())['qualified']
    assert not collect.qualify_state('uphill_before', state(boundary=1.))['qualified']
    assert collect.qualify_state('uphill_mid', state())['qualified']
    assert not collect.qualify_state('uphill_mid', state(vz=-.03))['qualified']
    assert collect.qualify_state('descent', state(vz=-.03,rise=-.03))['qualified']
    assert collect.qualify_state('step_before', state(rise=.12))['qualified']
    assert not collect.qualify_state('step_before', state(rise=.01))['qualified']
    assert collect.qualify_state('slow', state(speed=.02))['slow_kind']=='commanded_slow'
    assert collect.qualify_state('slow', dict(state(speed=.02),command=[0,0,0]))['slow_kind']=='stopped'


def test_persistence_does_not_count_nonconsecutive_or_incomplete_history():
    tracker = collect.AnchorQualification('turning')
    for tick in range(3): assert not tracker.observe(state(wz=.3),tick*50_000_000)['ready']
    assert tracker.observe(state(wz=.3),150_000_000)['ready']
    assert not tracker.observe(state(),200_000_000)['ready']
    assert not collect.qualify_state('moving_flat',dict(state(),dem_history_seconds=.5))['qualified']


def test_duplicate_gate_checks_endpoint_rotation_events_and_visible_buffers():
    reference = dict(root=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0.]), actor_input=np.zeros(25), joint=np.zeros(4))
    assert collect.compare_visible_state(reference,reference)['passed']
    altered={**reference,'actor_input':np.ones(25)}
    assert not collect.compare_visible_state(reference,altered)['passed']
    root=reference['root'].copy(); root[0]=.019
    assert collect.duplicate_gate(reference['root'],root,[False]*4,[False]*4)['passed']
    assert not collect.duplicate_gate(reference['root'],root,[False]*4,[True]*4)['passed']
    root[0]=.021
    assert not collect.duplicate_gate(reference['root'],root,[False]*4,[False]*4)['passed']


def test_tile_identity_survives_capacity_change_and_bank_has_no_append():
    p=collect.prepare_collection(parents_per_state=10)
    assert collect.jobs_for(p,10)[0]['parents'][0]['seed']==collect.jobs_for(p,2)[0]['parents'][0]['seed']
    assert p['candidate_count']==14 and p['horizon_steps']==40
    assert not any(p['interventions'].values())
    assert collect.prefix_command('turning',1)[2] != 0
    assert collect.prefix_command('slow',1)==[0.,0.,0.]


def test_causal_dem_profile_detects_terrain_boundary_and_rejects_invalid():
    frame=np.zeros((2,20,20));frame[1]=1.;frame[0,:,13:]=.2
    metrics=collect.dem_metrics(frame,.1)
    assert .2 <= metrics['forward_boundary_distance_m'] <= .5
    assert metrics['front_rise_m'] > .05
    frame[1]=0
    assert collect.dem_metrics(frame,.1)['valid_dem_fraction']==0


def test_spawn_height_uses_actual_mesh_triangle_not_center_origin_height():
    v=np.array([[0,0,0],[2,0,1],[0,2,0],[2,2,1.]])
    assert collect.mesh_height(v,np.array([[0,1,2],[1,3,2]]),[1,1]) == pytest.approx(.5)
    with pytest.raises(ValueError):collect.mesh_height(v,np.array([[0,1,2],[1,3,2]]),[4,4])


def test_policy_input_accepts_actual_dictionary_policy_observation():
    policy=np.array([[1.,2.]])
    assert collect.policy_input({'policy':policy,'critic':np.zeros((1,5))}) is policy
    assert collect.policy_input(policy) is policy


def test_measured_body_velocity_accepts_numpy_simulator_roots():
    root=np.array([0,0,0,1,0,0,0,.2,0,.03,0,0,.4])
    m=collect.physical_metrics(root,[.03]*5)
    assert m['body_wz_radps']==pytest.approx(.4)
    assert m['mean_world_vz_mps']==pytest.approx(.03)
    assert m['tilt_rad']==pytest.approx(0)


def test_visible_scalar_gate_handles_large_integer_manager_sentinels():
    assert collect.compare_visible_state({'sentinel':10**30},{'sentinel':10**30})['passed']


def test_policy_input_accepts_tensordict_style_mapping_from_env_step():
    class Observation:
        def keys(self):return ('policy','critic')
        def __getitem__(self,key):return np.ones((2,22)) if key=='policy' else None
    assert collect.policy_input(Observation()).shape==(2,22)


def test_restore_failure_gets_matched_prefix_fallback_but_boot_failure_does_not(tmp_path):
    out=tmp_path/'sim-output';out.mkdir()
    collect.write(out/'failure.json',dict(type='ValueError',message='visible state restoration failed: controller'))
    assert collect.should_replay({'state':'failed'},tmp_path)
    collect.write(out/'failure.json',dict(type='ModuleNotFoundError',message='isaaclab'))
    assert not collect.should_replay({'state':'failed'},tmp_path)
    assert not collect.should_replay({'state':'completed','restoration_gate_passed':False,'complete':{'anchor_available_by_env':[True,False]}},tmp_path)
    assert collect.should_replay({'state':'completed','restoration_gate_passed':False,'complete':{'anchor_available_by_env':[True,True]}},tmp_path)


def test_lazy_robot_cache_timestamp_restores_with_data_so_next_physics_refreshes():
    class Cache:
        def __init__(self):self.data=np.array([1.]);self.timestamp=1.
    class RobotData:
        def __init__(self):self._sim_timestamp=1.;self._root=Cache()
        def read(self,physical):
            if self._root.timestamp<self._sim_timestamp:
                self._root.data=np.array([physical]);self._root.timestamp=self._sim_timestamp
            return self._root.data.copy()
    robot=RobotData();saved=collect.capture_lazy_caches(robot)
    robot._sim_timestamp=3.;assert robot.read(3.)[0]==3.
    robot._sim_timestamp=1.;collect.restore_lazy_caches(robot,saved)
    robot._sim_timestamp=1.05
    assert robot.read(1.05)[0]==pytest.approx(1.05)
    assert saved['_root']['timestamp']==1.


def test_gate_binding_rejects_changed_plan_or_runtime_bytes():
    p=collect.prepare_collection()
    binding={'image_id':'image','source_sha256':{'collector.py':'abc','actor.pt':'actor'}}
    token=collect.gate_contract(p,binding)
    collect.validate_gate_contract(token,p,binding)
    with pytest.raises(ValueError,match='binding'):
        collect.validate_gate_contract(token,{**p,'parents_per_state':20},binding)
    with pytest.raises(ValueError,match='binding'):
        collect.validate_gate_contract(token,p,{**binding,'source_sha256':{'collector.py':'changed','actor.pt':'actor'}})


def test_common_world_anchor_selection_never_merges_different_clocks():
    selection=collect.CommonWorldAnchor(3)
    assert selection.consider([True,False,False],5)=='capture'
    assert selection.consider([False,True,False],6)=='keep'
    assert selection.consider([True,True,False],7)=='capture'
    assert selection.step==7 and selection.mask==[True,True,False]
    assert not selection.complete
    assert selection.consider([True,True,True],8)=='capture'
    assert selection.complete and selection.step==8


def test_flat_qualification_rejects_transverse_dem_slope():
    frame=np.zeros((2,20,20));frame[1]=1
    frame[0]=((np.arange(20)-9.5)*.1*.4)[:,None]
    assert not collect.dem_metrics(frame)['low_gradient_dem']


def test_fresh_candidate_sequence_repeats_probe_then_reverses_for_order_gate():
    seq=collect.fresh_sequence(True)
    assert seq[:2]==[0,0]
    assert seq[2:15]==list(range(1,14))
    assert seq[15:29]==list(reversed(range(14))) and seq[-1]==0
    assert collect.fresh_sequence(False)==[0,0]+list(range(1,14))


def test_fresh_parent_visible_state_slices_only_parent_and_preserves_shared_clock():
    snapshot={'root':np.ones((10,13)),'latency':np.arange(220).reshape(10,22),'clock':np.array(4.5),'rng':np.arange(32,dtype=np.uint8)}
    v=collect.parent_visible(snapshot,6,10)
    assert v['root'].shape==(13,) and v['latency'][0]==132
    assert v['clock']==4.5 and v['rng'].shape==(32,)
    assert snapshot['root'].shape==(10,13)


def test_counterfactual_outcomes_distinguish_natural_and_censor():
    assert collect.outcome_observed({'control_steps':4,'stop_reason':'natural_termination'})
    assert collect.outcome_observed({'control_steps':40,'stop_reason':'horizon_complete'})
    assert not collect.outcome_observed({'control_steps':12,'stop_reason':'boundary_censor'})
    assert not collect.outcome_observed({'control_steps':0,'stop_reason':'natural_termination'})


def test_duplicate_probe_requires_two_distinct_completed_processes():
    checks=[{'passed':True},{'passed':True}]
    assert collect.fresh_probe_passed(checks,'/first','/second')
    assert not collect.fresh_probe_passed(checks,'/same','/same')
    assert not collect.fresh_probe_passed([{'passed':False},{'passed':True}],'/first','/second')


def test_reduced_schedule_keeps_fourteen_candidates_and_post_forward_zero():
    seq=collect.fresh_schedule('forward_postzero')
    assert seq==[0,0]+list(range(1,14))+[0]
    assert set(seq)==set(range(14)) and len(seq)==16
    assert collect.fresh_schedule('full_reverse')==collect.fresh_sequence(True)
    with pytest.raises(ValueError):collect.fresh_schedule('silent_skip')


def test_persistent_budget_resume_keeps_original_start_and_counts_sibling_raw(tmp_path):
    plan=collect.prepare_collection();prod=tmp_path/'production';raw=tmp_path/'fresh-process-runtime'
    prod.mkdir();raw.mkdir();(raw/'branch').write_bytes(b'abcde')
    first=collect.persistent_budget(prod,plan,now=100.)
    assert first['deadline_unix']==10900.
    resumed=collect.persistent_budget(prod,plan,now=999.)
    assert resumed==first
    assert collect.budget_status(prod,first,now=10900.,free_bytes=100*2**30)['reason']=='persistent_deadline'
    status=collect.budget_status(prod,{**first,'raw_limit_bytes':1},now=101.,free_bytes=100*2**30)
    assert status['reason']=='raw_limit' and status['raw_bytes']>=5


def test_geometry_amendment_uses_inward_positive_edge_and_inverted_stair_descent():
    assert collect.reset_recipe('uphill_before')=={'x_m':7.95,'yaw_rad':np.pi}
    assert collect.reset_recipe('descent')=={'x_m':-6.,'yaw_rad':0.}
    assert collect.reset_recipe('uphill_mid')=={'x_m':-6.,'yaw_rad':0.}


def test_historical_turning_capacity_survives_smaller_new_benchmark():
    plan=collect.prepare_collection()
    amendment={'states':['turning','uphill_before'],'capacity_by_state':{'turning':10}}
    jobs=collect.amendment_jobs(plan,5,amendment)
    turning=[j for j in jobs if j['parents'][0]['state']=='turning']
    other=[j for j in jobs if j['parents'][0]['state']=='uphill_before']
    assert len(turning)==1 and len(turning[0]['parents'])==10
    assert turning[0]['job_id']=='turning-tiles0-1-2-3-4-5-6-7-8-9'
    assert len(other)==2 and all(len(j['parents'])==5 for j in other)
    assert all(j['terrain_cols']==10 for j in jobs)


def test_anchor_world_guard_rejects_outside_t0_before_first_action_reentry():
    assert collect.anchor_inside_guard([7.95,0],[0,0])
    assert not collect.anchor_inside_guard([7.95001,0],[0,0])
    assert not collect.anchor_inside_guard([0,float('nan')],[0,0])
    assert collect.anchor_inside_guard([107.9,200],[100,200])
