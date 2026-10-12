import pytest
from dataclasses import replace

from visualnav_transformer.eat_nav.p1_g3_rosfree_control import (
    RosFreeG3Control, RosFreeSameEpisodeGate, command_term_slice,
    reconcile_command_observation,
)


def paths():
    return [[[0, 0], [1, 0]] for _ in range(4)]


def poses():
    return [(0, 0, 0) for _ in range(4)]


class FakeFollower:
    created = []

    def __init__(self, path, *, library):
        self.calls = []
        FakeFollower.created.append(self)

    def step_diagnostic(self, x, y, yaw, stamp):
        self.calls.append(stamp)
        return ((0, 0, 0) if x >= 1 else (.3, 0, 0)), {'goal_reached': x >= 1}


def test_four_independent_cores_and_exact_step_order():
    FakeFollower.created.clear()
    controller = RosFreeG3Control(paths(), library='unused', follower_type=FakeFollower)
    assert len({id(x) for x in FakeFollower.created}) == 4
    events = []
    initial = controller.issue(2_000_000_000, poses(), frame='tile_local')
    assert [x.command for x in initial] == [(.3, 0, 0)] * 4
    controller.inject(initial, lambda commands: (events.append(('inject', commands)), commands)[1], stamp=2_000_000_000)
    controller.mark_actor(2_000_000_000, lambda: events.append(('actor', None)))
    controller.mark_physics(2_000_000_000, lambda: events.append(('physics', None)))
    controller.issue(2_050_000_000, poses(), frame='tile_local')
    assert [name for name, _ in events] == ['inject', 'actor', 'physics']
    assert all(f.calls == [2_000_000_000, 2_050_000_000] for f in FakeFollower.created)


def test_duplicate_stamp_wrong_order_and_frame_fail_closed():
    controller = RosFreeG3Control(paths(), library='unused', follower_type=FakeFollower)
    with pytest.raises(ValueError, match='frame'):
        controller.issue(2_000_000_000, poses(), frame='base')
    rows = controller.issue(2_000_000_000, poses(), frame='tile_local')
    with pytest.raises(ValueError, match='phase'):
        controller.mark_actor(2_000_000_000, lambda: None)
    controller.inject(rows, lambda commands: commands, stamp=2_000_000_000)
    with pytest.raises(ValueError, match='phase'):
        controller.mark_physics(2_000_000_000, lambda: None)
    controller.mark_actor(2_000_000_000, lambda: None)
    controller.mark_physics(2_000_000_000, lambda: None)
    with pytest.raises(ValueError, match='duplicate'):
        controller.issue(2_000_000_000, poses(), frame='tile_local')
    with pytest.raises(ValueError, match='50 ms'):
        controller.issue(2_100_000_000, poses(), frame='tile_local')


def test_goal_zero_hold_and_episode_reset():
    controller = RosFreeG3Control(paths(), library='unused', follower_type=FakeFollower)
    rows = controller.issue(2_000_000_000, [(1, 0, 0)] * 4, frame='tile_local')
    assert all(row.command == (0, 0, 0) and row.diagnostic['goal_reached'] for row in rows)
    controller.inject(rows, lambda commands: commands, stamp=2_000_000_000)
    controller.mark_actor(2_000_000_000, lambda: None)
    controller.mark_physics(2_000_000_000, lambda: None)
    assert all(row.command == (0, 0, 0) for row in controller.issue(
        2_050_000_000, [(1, 0, 0)] * 4, frame='tile_local'))
    controller.reset_episode(paths())
    assert controller.issue(12_000_000_000, poses(), frame='tile_local')[0].command == (.3, 0, 0)


def test_same_episode_gate_rejects_reset_and_step_gap():
    controller = RosFreeG3Control(paths(), library='unused', follower_type=FakeFollower)
    rows = controller.issue(2_000_000_000, poses(), frame='tile_local')
    gate = RosFreeSameEpisodeGate([20] * 4, 800)
    gate.accept(0, 2_000_000_000, rows, [20] * 4, [21] * 4, [False] * 4, [])
    with pytest.raises(ValueError, match='noncontiguous'):
        gate.accept(2, 2_100_000_000, rows, [22] * 4, [23] * 4, [False] * 4, [])
    next_rows = tuple(replace(row, stamp_ns=2_050_000_000) for row in rows)
    with pytest.raises(ValueError, match='reset'):
        gate.accept(1, 2_050_000_000, next_rows, [21] * 4, [22] * 4, [False] * 4, [0])


def test_four_env_640_step_unreached_control_has_no_stamp_or_episode_gap():
    controller = RosFreeG3Control(paths(), library='unused', follower_type=FakeFollower)
    gate = RosFreeSameEpisodeGate([20] * 4, 800)
    for step in range(640):
        stamp = 2_000_000_000 + step * 50_000_000
        rows = controller.issue(stamp, poses(), frame='tile_local')
        controller.inject(rows, lambda commands: commands, stamp=stamp)
        controller.mark_actor(stamp, lambda: None)
        controller.mark_physics(stamp, lambda: None)
        gate.accept(step, stamp, rows, [20 + step] * 4, [21 + step] * 4,
                    [False] * 4, [])
    assert gate.next_step == 640
    assert all(len(core.calls) == 640 for core in controller.followers)


def test_command_term_is_resolved_from_observation_contract():
    class Cfg:
        noise = None
        scale = 1.0

    class Manager:
        _group_obs_term_names = {'policy': ['gravity', 'velocity_commands', 'joints']}
        _group_obs_term_dim = {'policy': [(3,), (3,), (4,)]}
        _group_obs_term_cfgs = {'policy': [Cfg(), Cfg(), Cfg()]}

    assert command_term_slice(Manager(), 'policy') == slice(3, 6)
    Manager._group_obs_term_cfgs['policy'][1].noise = object()
    with pytest.raises(ValueError, match='noise'):
        command_term_slice(Manager(), 'policy')


def test_command_reconciliation_uses_prior_step_not_overwritten_buffer():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace

    class Cfg:
        noise = None
        scale = 1.0

    manager = SimpleNamespace(
        _group_obs_term_names={'policy': ['gravity', 'velocity_commands'],
                               'critic': ['velocity_commands', 'joints']},
        _group_obs_term_dim={'policy': [(3,), (3,)], 'critic': [(3,), (2,)]},
        _group_obs_term_cfgs={'policy': [Cfg(), Cfg()], 'critic': [Cfg(), Cfg()]},
    )
    target = torch.tensor([[.3, 0., .1]] * 4)
    old = torch.tensor([[.1, 0., -.2]] * 4)
    raw_after_env_step = torch.tensor([[.8, .8, .8]] * 4)
    policy = torch.zeros((4, 6))
    critic = torch.zeros((4, 5))
    observed = {key: policy.clone() for key in
                ('policy_pre_noise', 'policy_post_noise_pre_delay', 'policy_actor_input')}
    observed['critic_observation'] = critic.clone()
    bare = SimpleNamespace(
        observation_manager=manager,
        command_manager=SimpleNamespace(_terms={
            'base_velocity': SimpleNamespace(vel_command_b=target.clone())}),
        _latency_policy_obs_prev=torch.cat([torch.zeros((4, 3)), raw_after_env_step], dim=1),
        _onix_latency_seconds=torch.full((4, 1), .025), step_dt=.05,
    )
    previous = torch.cat([torch.zeros((4, 3)), old], dim=1)
    corrected, diagnostics, result = reconcile_command_observation(
        {'policy': policy, 'critic': critic}, observed, bare, target.tolist(), previous)
    expected = target + .5 * (old - target)
    assert torch.allclose(corrected['policy'][:, 3:6], expected)
    assert torch.allclose(corrected['critic'][:, :3], target)
    assert torch.equal(policy, torch.zeros_like(policy))
    assert torch.equal(critic, torch.zeros_like(critic))
    assert torch.allclose(diagnostics['policy_actor_input'][:, 3:6], expected)
    assert torch.allclose(bare._latency_policy_obs_prev[:, 3:6], target)
    assert result['observation_sample_count_added'] == 0
    assert result['previous_raw'] == old.tolist()


def test_reconciliation_owns_inference_observations_and_latency_across_steps():
    """A model-owned inference tensor must stay immutable through multi-env updates."""
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace

    class Cfg:
        noise = None
        scale = 1.0

    manager = SimpleNamespace(
        _group_obs_term_names={'policy': ['body', 'velocity_commands', 'joints'],
                               'critic': ['velocity_commands', 'body']},
        _group_obs_term_dim={'policy': [(2,), (3,), (2,)], 'critic': [(3,), (2,)]},
        _group_obs_term_cfgs={'policy': [Cfg(), Cfg(), Cfg()],
                              'critic': [Cfg(), Cfg()]},
    )
    commands = torch.tensor([[.2, .0, .1], [.4, .1, -.1],
                             [.6, -.1, .2], [.0, .0, .0]], dtype=torch.float64)
    bare = SimpleNamespace(
        observation_manager=manager,
        command_manager=SimpleNamespace(_terms={
            'base_velocity': SimpleNamespace(vel_command_b=commands.clone())}),
        _onix_latency_seconds=torch.tensor([[.0], [.025], [.05], [.025]], dtype=torch.float64),
        step_dt=.05,
    )
    with torch.inference_mode():
        bare._latency_policy_obs_prev = torch.full((4, 7), .7, dtype=torch.float64)
    previous_raw = torch.zeros((4, 7), dtype=torch.float64)
    previous_raw[:, 2:5] = torch.tensor([[.8, .0, .0], [.1, .0, .1],
                                         [-.2, .2, .0], [.3, .0, .0]], dtype=torch.float64)

    for step in range(3):
        with torch.inference_mode():
            original_policy = torch.arange(28, dtype=torch.float64).reshape(4, 7) + step
            original_critic = torch.arange(20, dtype=torch.float64).reshape(4, 5) + step
            obs = {'policy': original_policy, 'critic': original_critic}
            observed = {'policy_pre_noise': original_policy.clone(),
                        'policy_post_noise_pre_delay': original_policy.clone(),
                        'policy_actor_input': original_policy.clone(),
                        'critic_observation': original_critic.clone()}
        saved_policy = original_policy.clone()
        saved_critic = original_critic.clone()
        saved_diagnostics = {key: value.clone() for key, value in observed.items()}
        expected_alpha = torch.tensor([[0.], [.5], [1.], [.5]], dtype=torch.float64)
        expected_actor = commands + expected_alpha * (previous_raw[:, 2:5] - commands)
        buffer_before = bare._latency_policy_obs_prev
        version_before = None if buffer_before.is_inference() else buffer_before._version
        corrected, captured, detail = reconcile_command_observation(
            obs, observed, bare, commands.tolist(), previous_raw)

        assert corrected['policy'].dtype == original_policy.dtype
        assert corrected['policy'].device == original_policy.device
        assert corrected['policy'].data_ptr() != original_policy.data_ptr()
        assert torch.equal(original_policy, saved_policy)
        assert torch.equal(original_critic, saved_critic)
        assert all(torch.equal(observed[key], value) for key, value in saved_diagnostics.items())
        assert torch.equal(corrected['policy'][:, :2], saved_policy[:, :2])
        assert torch.equal(corrected['policy'][:, 5:], saved_policy[:, 5:])
        assert torch.equal(corrected['critic'][:, 3:], saved_critic[:, 3:])
        assert torch.equal(corrected['policy'][:, 2:5], expected_actor)
        assert torch.equal(corrected['critic'][:, :3], commands)
        assert torch.equal(captured['policy_pre_noise'][:, 2:5], commands)
        assert torch.equal(captured['policy_post_noise_pre_delay'][:, 2:5], commands)
        assert torch.equal(captured['policy_actor_input'][:, 2:5], expected_actor)
        assert detail['actor_seen'] == expected_actor.tolist()
        assert torch.equal(bare._latency_policy_obs_prev[:, 2:5], commands)
        assert torch.equal(bare._latency_policy_obs_prev[:, :2],
                           torch.full((4, 2), .7, dtype=torch.float64))
        assert not bare._latency_policy_obs_prev.is_inference()
        if version_before is None:
            assert bare._latency_policy_obs_prev.data_ptr() != buffer_before.data_ptr()
        else:
            assert bare._latency_policy_obs_prev._version == version_before + 1
        previous_raw = bare._latency_policy_obs_prev.clone()
        commands = torch.tensor([[.0, .0, .0], [.5, .0, .0],
                                 [.25, .0, -.25], [.0, .0, .0]], dtype=torch.float64)
        bare.command_manager._terms['base_velocity'].vel_command_b = commands.clone()
