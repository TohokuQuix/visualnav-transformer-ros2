import torch
import pytest

from visualnav_transformer.eat_nav.p1_teacher_runtime import PhysicsSubstepTap, validate_teacher_step


class Sim:
    def __init__(self):
        self.calls = 0
    def step(self, render=False):
        self.calls += 1


class View:
    def __init__(self, sim):
        self.sim = sim
    def get_net_contact_forces(self, dt):
        return torch.tensor([[float(self.sim.calls), 0, 0]])


def test_tap_reads_after_each_real_physics_step_without_extra_step():
    sim = Sim(); view = View(sim)
    original = sim.step.__func__
    with PhysicsSubstepTap(sim, view, num_envs=1, body_count=1,
                           physics_dt=0.005, substeps=3, stamp_ns=100_000_000) as tap:
        for _ in range(3): sim.step(render=False)
        result = tap.finish()[0]
    assert sim.calls == 3
    assert sim.step.__func__ is original
    assert result['peak_magnitude_N'] == [3.0]
    assert result['peak_substep'] == [2]
    assert result['integral_force_Ns'][0][0] == pytest.approx(0.03)
    assert result['physics_stamp_ns'] == [105_000_000, 110_000_000, 115_000_000]


def test_tap_fails_if_physics_substep_missing():
    sim = Sim()
    with PhysicsSubstepTap(sim, View(sim), num_envs=1, body_count=1,
                           physics_dt=0.005, substeps=3, stamp_ns=0) as tap:
        sim.step()
        with pytest.raises(ValueError, match='incomplete'):
            tap.finish()


def test_teacher_step_timestamp_and_reward_alignment():
    good = {'step': 2, 'logical_stamp_ns': 100_000_000,
            'reward': {'step': 2, 'logical_stamp_ns': 100_000_000,
                       'reward_compute_phase': 'post_physics_pre_command_update'},
            'substep': {'sample_count': 10, 'physics_stamp_ns':
                        [100_000_000 + 5_000_000 * (i + 1) for i in range(10)]},
            'control_dt_s': 0.05}
    validate_teacher_step(good)
    bad = dict(good, reward=dict(good['reward'], step=1))
    with pytest.raises(ValueError, match='reward_alignment'):
        validate_teacher_step(bad)
