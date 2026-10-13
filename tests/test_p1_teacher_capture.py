import math
from types import SimpleNamespace

import pytest
import torch

from visualnav_transformer.eat_nav.p1_teacher_capture import (
    RewardTermCapture, SubstepContactAccumulator, compute_pain, joint_body_maps,
)


class FakeManager:
    def __init__(self):
        self.active_terms = ['move', 'unused', 'termination_penalty']
        self._step_reward = torch.tensor([[2.0, 0.0, -1.0], [4.0, 0.0, 0.0]])
        self.calls = 0
        self.cfgs = {'move': SimpleNamespace(weight=2.0),
                     'unused': SimpleNamespace(weight=0.0),
                     'termination_penalty': SimpleNamespace(weight=-1.0)}

    def get_term_cfg(self, name):
        return self.cfgs[name]

    def compute(self, dt):
        self.calls += 1
        raise AssertionError('must not recompute reward')


def test_reward_capture_reads_existing_manager_without_recompute():
    manager = FakeManager()
    capture = RewardTermCapture.from_manager(manager, 0.05)
    row = capture.capture_step(manager, torch.tensor([0.05, 0.2]), 7, 350_000_000)
    assert manager.calls == 0
    assert row['weighted_contribution'][0] == pytest.approx([0.1, 0.0, -0.05])
    assert row['raw_value'][0][0] == pytest.approx(1.0)
    assert row['raw_value'][0][2] == pytest.approx(1.0)
    assert row['raw_value'][0][1] is None
    assert row['termination_contribution'][0] == pytest.approx(-0.05)
    assert row['reward_compute_phase'] == 'post_physics_pre_command_update'


def test_reward_capture_fails_on_mismatched_total_and_changed_weights():
    manager = FakeManager()
    capture = RewardTermCapture.from_manager(manager, 0.05)
    with pytest.raises(ValueError, match='reward_sum'):
        capture.capture_step(manager, torch.tensor([0.0, 0.2]), 0, 0)
    manager.cfgs['move'].weight = 3.0
    with pytest.raises(ValueError, match='weight_changed'):
        capture.capture_step(manager, torch.tensor([0.05, 0.2]), 0, 0)


def test_substep_contact_collects_true_per_substep_peak_and_integral():
    acc = SubstepContactAccumulator(body_count=2, physics_dt=0.005, substeps=3)
    for i, force in enumerate(([[1, 0, 0], [0, 0, 0]],
                                [[3, 0, 0], [0, 4, 0]],
                                [[2, 0, 0], [0, 0, 0]])):
        acc.accept(torch.tensor(force, dtype=torch.float32), i, i * 5_000_000)
    result = acc.finish()
    assert result['peak_magnitude_N'] == [3.0, 4.0]
    assert result['peak_substep'] == [1, 1]
    assert result['integral_force_Ns'][0][0] == pytest.approx(0.03)
    assert result['contact_substep_count'] == [3, 1]
    assert result['sample_count'] == 3
    with pytest.raises(ValueError, match='already_finalized'):
        acc.accept(torch.zeros(2, 3), 3, 15_000_000)


def test_substep_missing_or_stale_is_fail_closed():
    acc = SubstepContactAccumulator(body_count=1, physics_dt=0.005, substeps=2)
    acc.accept(torch.zeros(1, 3), 0, 0)
    with pytest.raises(ValueError, match='incomplete'):
        acc.finish()
    with pytest.raises(ValueError, match='substep_order'):
        acc.accept(torch.zeros(1, 3), 0, 0)


def test_mapping_and_pain_missing_allowables():
    maps = joint_body_maps(['j0', 'j1'], ['b0'], {'drive': ['j0']})
    assert maps['joints'][0]['actuator_terms'] == ['drive']
    assert maps['joints'][1]['actuator_terms'] == []
    assert maps['bodies'][0]['name'] == 'b0'
    loads = {'torque_Nm': [[2., 4.], [3., 6.]],
             'body_force_world_N': [[[3., 4., 0.]], [[0., 0., 10.]]]}
    assert compute_pain(loads, {'schema': 'eatnav_pain_allowables_v1',
                                'joint_torque_ref_Nm': {}, 'body_force_ref_N': {}})['status'] == 'missing_allowables'


def test_synthetic_pain_keeps_components_and_peak():
    loads = {'torque_Nm': [[2., 4.], [3., 6.]],
             'body_force_world_N': [[[3., 4., 0.]], [[0., 0., 10.]]]}
    allow = {'schema': 'eatnav_pain_allowables_v1',
             'joint_torque_ref_Nm': {'j0': 2., 'j1': 4.},
             'body_force_ref_N': {'b0': 5.},
             'joint_weights': {'j0': 1., 'j1': 1.},
             'body_weights': {'b0': 1.}}
    result = compute_pain(loads, allow, joint_names=['j0', 'j1'], body_names=['b0'], dt=0.05)
    assert result['status'] == 'computed_synthetic_only'
    assert result['joint_component'][0] == pytest.approx(2.0)
    assert result['body_component'] == pytest.approx([1.0, 4.0])
    assert result['joint_peak_utilization'] == pytest.approx(1.5)
    assert result['body_peak_utilization'] == pytest.approx(2.0)
    assert result['exceedance_duration_s'] == pytest.approx(0.1)
