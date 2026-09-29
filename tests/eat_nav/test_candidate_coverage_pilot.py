"""Pilot order and deadline checks are frozen before candidate outcomes."""

from visualnav_transformer.eat_nav.candidate_coverage_pilot import (
    build_schedule, may_start_next,
)


def test_schedule_counts_include_one_anchor_freeze_per_terrain():
    schedule = build_schedule(["boxes", "rail"], seeds=list(range(1001, 1009)))
    assert len(schedule) == 34
    assert [row["kind"] for row in schedule[:4]] == ["anchor", "duplicate", "duplicate", "duplicate"]
    assert [row["kind"] for row in schedule[4:12]] == ["nomad"] * 8
    assert [row["kind"] for row in schedule[12:17]] == ["bank"] * 5
    assert schedule[0]["purpose"] == "freeze anchor state and map Path before candidate execution"
    assert schedule[4]["seed"] == 1001
    assert schedule[11]["seed"] == 1008
    assert schedule[17]["terrain_type"] == "rail"
    assert len({row["run_id"] for row in schedule}) == 34


def test_budget_gate_stops_before_next_candidate_when_forecast_exceeds_four_hours():
    assert may_start_next(elapsed_sec=100, predicted_remaining_sec=100,
                          approved_budget_sec=14400)
    assert not may_start_next(elapsed_sec=11000, predicted_remaining_sec=3500,
                              approved_budget_sec=14400)
    assert not may_start_next(elapsed_sec=14400, predicted_remaining_sec=0,
                              approved_budget_sec=14400)


def test_frozen_config_expands_to_exact_119_run_order():
    from pathlib import Path
    from collections import Counter
    import yaml
    path = Path(__file__).parents[2] / 'docs/quality-identifiability-gate/candidate-coverage/pilot-config.yaml'
    spec = yaml.safe_load(path.read_text())
    rows = build_schedule(spec['terrain_order'], seeds=spec['nomad_seeds_pilot'])
    assert len(rows) == spec['run_count']['total'] == 119
    assert Counter(row['kind'] for row in rows) == {
        'anchor': 7, 'duplicate': 21, 'nomad': 56, 'bank': 35}
    assert spec['approved_wall_clock_sec'] == 14400
    assert spec['max_full_prefix_steps'] == 40
    assert [row['seed'] for row in rows if row['kind'] == 'nomad'][:8] == spec['nomad_seeds_k32'][:8]


def test_bind_schedule_attaches_frozen_state_without_reordering():
    from visualnav_transformer.eat_nav.candidate_coverage_pilot import bind_schedule
    order = build_schedule(['boxes', 'rail'], seeds=list(range(1001, 1009)))
    selected = [{'terrain_type': 'boxes', 'state_id': 'coverage-000', 'view_index': 4,
                 'episode_key': [0, 1, 0], 'step': 25, 'full_prefix_steps': 26},
                {'terrain_type': 'rail', 'state_id': 'coverage-001', 'view_index': 8,
                 'episode_key': [0, 2, 0], 'step': 15, 'full_prefix_steps': 16}]
    bound = bind_schedule(order, selected)
    assert bound[0]['state_id'] == 'coverage-000'
    assert bound[16]['state_id'] == 'coverage-000'
    assert bound[17]['state_id'] == 'coverage-001'
    assert [r['run_id'] for r in bound] == [r['run_id'] for r in order]


def test_duplicate_gate_measures_relative_xy_and_rejects_changed_anchor():
    import pytest
    from visualnav_transformer.eat_nav.candidate_coverage_pilot import duplicate_gate
    critical=('root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
              'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
              'critic_observation','critic_value','action_manager_action','latency_substeps',
              'latency_seconds','latency_policy_obs_prev','action_delay_buffers','dem_history_sha256')
    anchor={name:0 for name in critical}
    summaries=[{'fixed_map_path_sha256':'same','anchor':dict(anchor)} for _ in range(3)]
    def trace(offset):
        return [{'root_state':[10.,20.], 'post_root_state':[10.+.01*k,20.+offset],
                 'command':[.1,0.,0.], 'terminal':None} for k in range(40)]
    result=duplicate_gate(summaries,[trace(0.),trace(0.),trace(0.)],xy_rms_limit_m=.0105)
    assert result['duplicate_xy_rms_m']==0
    with pytest.raises(ValueError,match='XY RMS'):
        duplicate_gate(summaries,[trace(0.),trace(.03),trace(0.)],xy_rms_limit_m=.0105)
    summaries[1]['anchor']['actor_action']=1
    with pytest.raises(ValueError,match='anchor'):
        duplicate_gate(summaries,[trace(0.)]*3,xy_rms_limit_m=.0105)
