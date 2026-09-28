import torch
import pytest
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, ObservationCapture, capture_actor_critic


class Term:
    def __init__(self):
        self.vel_command_b = torch.zeros(8, 3)


class Commands:
    def __init__(self):
        self._terms = {'base_velocity': Term()}
        self.calls = 0
    def compute(self, dt):
        self.calls += 1
        self._terms['base_velocity'].vel_command_b.fill_(-1)


def test_command_override_survives_command_manager_compute():
    manager = Commands()
    bridge = CommandOverride(manager, 8)
    command = torch.full((8, 3), 0.3)
    bridge.set(command)
    manager.compute(0.05)
    assert manager.calls == 1
    assert torch.allclose(manager._terms['base_velocity'].vel_command_b, command)
    bridge.close()


class Noise:
    def __init__(self):
        self.func = lambda x, cfg: x + 2


class Cfg:
    def __init__(self):
        self.noise = Noise(); self.clip = None; self.scale = 3


class Observations:
    _group_obs_term_names = {'policy': ['x'], 'critic': ['v']}
    _group_obs_term_cfgs = {'policy': [Cfg()], 'critic': [Cfg()]}
    _group_obs_term_dim = {'policy': [(1,)], 'critic': [(1,)]}
    def compute_group(self, group, update_history=False):
        cfg = self._group_obs_term_cfgs[group][0]
        x = torch.ones(8, 1)
        return cfg.noise.func(x, cfg.noise) * cfg.scale


def test_observation_capture_records_one_pass_and_noise_delay_stages():
    manager = Observations()
    capture = ObservationCapture(manager)
    capture.begin_step()
    post = manager.compute_group('policy')
    critic = manager.compute_group('critic')
    captured = capture.finish_step({'policy': post + 1, 'critic': critic})
    assert torch.allclose(captured['policy_pre_noise'], torch.full((8, 1), 3.))
    assert torch.allclose(captured['policy_post_noise_pre_delay'], torch.full((8, 1), 9.))
    assert torch.allclose(captured['policy_actor_input'], torch.full((8, 1), 10.))
    assert torch.allclose(captured['critic_observation'], critic)
    with pytest.raises(ValueError, match='once'):
        capture.begin_step(); manager.compute_group('policy'); manager.compute_group('policy'); capture.finish_step({'policy': post, 'critic': critic})
    capture.close()


class Policy:
    def __init__(self):
        self.actor = torch.nn.Sequential(torch.nn.Linear(22, 256), torch.nn.ELU(), torch.nn.Linear(256, 6))
        self.critic = torch.nn.Linear(267, 1)
    def act_inference(self, obs):
        return self.actor(obs["policy"])
    def evaluate(self, obs):
        return self.critic(obs["critic"])


def test_actor_capture_uses_actual_policy_forward():
    action, latent, value = capture_actor_critic(Policy(), torch.zeros(8,22), torch.zeros(8,267), latent_layer=2)
    assert action.shape == (8,6)
    assert latent.shape == (8,256)
    assert value.shape == (8,1)


def test_copy_batch_state_preserves_nonbatch_rows():
    from visualnav_transformer.eat_nav.phase_c_sim import copy_batch_rows
    x = torch.tensor([[1.,2.],[4.,5.],[7.,8.]])
    copy_batch_rows(x, 0, [1,2], batch_axis=0)
    assert torch.equal(x, torch.tensor([[1.,2.],[1.,2.],[1.,2.]]))
    ring = torch.arange(24.).reshape(2,3,4)
    copy_batch_rows(ring, 0, [1,2], batch_axis=1)
    assert torch.equal(ring[:,0], ring[:,1]) and torch.equal(ring[:,0], ring[:,2])
