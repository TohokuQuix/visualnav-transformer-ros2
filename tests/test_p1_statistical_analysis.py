import copy
import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_statistical_protocol import build_assignments
from visualnav_transformer.eat_nav.p1_statistical_analysis import extract_outcome, analyze


ROOT = Path(__file__).parents[1]
SOURCE = json.loads((ROOT/'config/eat_nav_p1_batch_source_contract.json').read_text())
CONFIG = json.loads((ROOT/'config/eat_nav_p1_statistical_v1.json').read_text())


def trace(reward=1.0, terminal_step=None):
    rows = []
    for step in range(120):
        root = [step*.001, 0, .5, 1, 0, 0, 0] + [0]*6
        post = [(step+1)*.001, 0, .5, 1, 0, 0, 0] + [0]*6
        rows.append({'control_step':step, 'reward':reward,
                     'root_state':root, 'post_root_state':post,
                     'terminal': {'root_state':post,'termination_reason':['fallen']}
                     if step == terminal_step else None})
    return rows


def test_primary_outcomes_are_40_steps_and_never_use_post_reset():
    result = extract_outcome(trace(), (.75, 0), gamma=.99)
    assert result['valid_primary']
    assert result['discounted_reward_2s'] == pytest.approx(sum(.99**i for i in range(40)))
    assert result['global_goal_progress_2s_m'] == pytest.approx(.04)
    invalid = extract_outcome(trace(terminal_step=12), (.75, 0), gamma=.99)
    assert not invalid['valid_primary']
    assert invalid['discounted_reward_2s'] is None
    assert invalid['global_goal_progress_2s_m'] is None
    assert invalid['first_terminal_step'] == 12
    assert invalid['termination_reason'] == ['fallen']


def test_short_terminal_trace_is_censored_without_post_reset_values():
    result = extract_outcome(trace(terminal_step=0)[:1], (.75, 0), gamma=.99)
    assert result['valid_primary'] is False
    assert result['first_terminal_step'] == 0
    assert result['discounted_reward_2s'] is None


def test_pilot_uses_batch_contrasts_and_marks_two_batch_intervals_descriptive():
    manifest = build_assignments(SOURCE, CONFIG['batch_seeds'], 123)
    outcomes = []
    for trial in manifest['trials']:
        reward = 10.0 + trial['batch_index']
        progress = .3 + .02*trial['batch_index']
        if trial['path_identity'] != 'direct_reference':
            reward += .5 if trial['batch_index'] == 0 else 1.5
            progress += .02 if trial['batch_index'] == 0 else .08
        outcomes.append({'trial_id': trial['trial_id'], 'valid_primary': True,
                         'discounted_reward_2s': reward,
                         'global_goal_progress_2s_m': progress,
                         'first_terminal_step': None})
    result = analyze(outcomes, manifest, CONFIG)
    first = result['comparisons'][0]
    assert first['independent_batch_count'] == 2
    assert first['confidence_interval_kind'] == 'pilot_descriptive_underpowered_cluster_bootstrap'
    assert first['discounted_reward_2s']['mean_difference'] == pytest.approx(1.0)
    assert first['global_goal_progress_2s_m']['mean_difference'] == pytest.approx(.05)
    assert first['discounted_reward_2s']['independent_contrasts'] == pytest.approx([.5,1.5])
    assert first['discounted_reward_2s']['recommended_total_batches'] >= 2
    assert result['confirmatory_significance_claim'] is False


def test_missing_trial_or_fabricated_pairing_is_rejected():
    manifest = build_assignments(SOURCE, CONFIG['batch_seeds'], 123)
    with pytest.raises(ValueError, match='trial coverage'):
        analyze([], manifest, CONFIG)
    paired = copy.deepcopy(manifest)
    paired['paired_policy_noise'] = True
    with pytest.raises(ValueError, match='pairing'):
        analyze([], paired, CONFIG)
