"""Exact one-pass Actor/observation capture and external follower command injection."""
from __future__ import annotations

import math
import torch


class CommandOverride:
    def __init__(self, command_manager, num_envs, name='base_velocity', provider=None):
        self.manager = command_manager
        self.term = command_manager._terms[name]
        self.num_envs = num_envs
        self._original = command_manager.compute
        self._command = None
        self.provider = provider

        def compute_with_override(*args, **kwargs):
            result = self._original(*args, **kwargs)
            if self.provider is not None:
                self.set(self.provider())
            if self._command is not None:
                self.term.vel_command_b.copy_(self._command)
            return result

        command_manager.compute = compute_with_override

    def set(self, command):
        target = self.term.vel_command_b
        command = torch.as_tensor(command, dtype=target.dtype, device=target.device)
        if command.shape != (self.num_envs, 3) or not torch.isfinite(command).all():
            raise ValueError('follower command shape/nonfinite mismatch')
        self._command = command.clone()
        target.copy_(self._command)

    def close(self):
        self.manager.compute = self._original


class ObservationCapture:
    def __init__(self, manager):
        self.manager = manager
        self._original_compute_group = manager.compute_group
        self._noise_originals = []
        self.active = False
        self.policy_calls = 0
        self.post_noise = None
        self.critic = None
        self.pre_terms = {}
        for group, cfgs in manager._group_obs_term_cfgs.items():
            if group != 'policy':
                continue
            names = manager._group_obs_term_names[group]
            for name, cfg in zip(names, cfgs):
                noise = getattr(cfg, 'noise', None)
                func = getattr(noise, 'func', None)
                if not callable(func):
                    continue
                def capture_noise(obs, *args, _name=name, _func=func, **kwargs):
                    if self.active:
                        self.pre_terms[_name] = obs.detach().clone()
                    return _func(obs, *args, **kwargs)
                noise.func = capture_noise
                self._noise_originals.append((noise, func))

        def compute_group(group, *args, **kwargs):
            result = self._original_compute_group(group, *args, **kwargs)
            if self.active and group == 'policy':
                self.policy_calls += 1
                self.post_noise = result.detach().clone()
            elif self.active and group == 'critic':
                self.critic = result.detach().clone()
            return result
        manager.compute_group = compute_group

    def begin_step(self):
        self.active = True
        self.policy_calls = 0
        self.pre_terms = {}
        self.post_noise = None
        self.critic = None

    def finish_step(self, returned_observations):
        if self.policy_calls != 1:
            raise ValueError(f'policy observation must be generated once per control step, got {self.policy_calls}')
        actor_input = returned_observations['policy'].detach().clone()
        if self.post_noise.shape != actor_input.shape:
            raise ValueError('post-noise/post-delay shape mismatch')
        pre = self.post_noise.clone()
        names = self.manager._group_obs_term_names['policy']
        cfgs = self.manager._group_obs_term_cfgs['policy']
        dims = self.manager._group_obs_term_dim['policy']
        offset = 0
        for name, cfg, dim in zip(names, cfgs, dims):
            width = math.prod(dim)
            raw = self.pre_terms.get(name)
            if raw is not None:
                clean = raw.clone()
                if cfg.clip:
                    clean = clean.clamp(min=cfg.clip[0], max=cfg.clip[1])
                if cfg.scale is not None:
                    clean = clean * cfg.scale
                pre[:, offset:offset+width] = clean.reshape(len(clean), width)
            offset += width
        if offset != actor_input.shape[1]:
            raise ValueError('policy term dimensions do not match Actor input')
        self.active = False
        return {'policy_pre_noise': pre, 'policy_post_noise_pre_delay': self.post_noise,
                'policy_actor_input': actor_input,
                'critic_observation': (self.critic if self.critic is not None else
                                       returned_observations['critic'].detach().clone()),
                'policy_term_names': tuple(names), 'policy_term_dims': tuple(dims)}

    def close(self):
        self.manager.compute_group = self._original_compute_group
        for noise, func in self._noise_originals:
            noise.func = func
        self.active = False


@torch.no_grad()
def capture_actor_critic(policy, actor_obs, critic_obs, latent_layer=6):
    if actor_obs.ndim != 2 or critic_obs.ndim != 2 or len(actor_obs) != len(critic_obs):
        raise ValueError('Actor/Critic batch shape mismatch')
    latent = []
    def hook(_module, inputs):
        latent.append(inputs[0].detach().clone())
    handle = policy.actor[latent_layer].register_forward_pre_hook(hook)
    try:
        observations = {"policy": actor_obs, "critic": critic_obs}
        action = policy.act_inference(observations)
    finally:
        handle.remove()
    if len(latent) != 1:
        raise ValueError('Actor latent hook did not fire exactly once')
    value = policy.evaluate(observations)
    return action.detach().clone(), latent[0], value.detach().clone()


def copy_batch_rows(tensor, source: int, destinations, batch_axis=0):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('state field is not a tensor')
    if source < 0 or any(i < 0 or i >= tensor.shape[batch_axis] for i in destinations):
        raise IndexError('branch index out of range')
    source_value = tensor.select(batch_axis, source).clone()
    for index in destinations:
        tensor.select(batch_axis, index).copy_(source_value)


def clone_internal_branch_state(env, source=0, destinations=None):
    """Copy visible per-env state; enumerate uncopyable global/PhysX state explicitly."""
    import copy
    if destinations is None:
        destinations = list(range(1, env.num_envs))
    destinations = list(destinations)
    report = {'copied': [], 'unreplicated': []}

    def field(owner, name, label=None, batch_axis=0):
        value = getattr(owner, name, None)
        key = label or name
        if isinstance(value, torch.Tensor):
            copy_batch_rows(value, source, destinations, batch_axis)
            report['copied'].append(key)
        else:
            report['unreplicated'].append(key)

    robot = env.scene['robot']
    origins = env.scene.env_origins
    root = robot.data.root_state_w[source].clone().repeat(len(destinations), 1)
    root[:, :3] += origins[destinations] - origins[source]
    ids = torch.as_tensor(destinations, device=origins.device, dtype=torch.long)
    robot.write_root_state_to_sim(root, env_ids=ids)
    joint_pos = robot.data.joint_pos[source].clone().repeat(len(destinations), 1)
    joint_vel = robot.data.joint_vel[source].clone().repeat(len(destinations), 1)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    report['copied'] += ['robot.root_state_relative_to_origin', 'robot.joint_position', 'robot.joint_velocity']
    for name in ('episode_length_buf', '_latency_substeps', '_onix_latency_seconds',
                 '_latency_policy_obs_prev', '_latency_policy_obs_reset_mask',
                 '_terrain_path_length_xy', '_terrain_path_initialized',
                 '_onix_flipper_fold_mode'):
        field(env, name)
    previous_xy = getattr(env, '_terrain_prev_root_pos_xy', None)
    if isinstance(previous_xy, torch.Tensor):
        for dest in destinations:
            previous_xy[dest] = previous_xy[source] + (origins[dest, :2] - origins[source, :2])
        report['copied'].append('_terrain_prev_root_pos_xy_relative_to_origin')
    else:
        report['unreplicated'].append('_terrain_prev_root_pos_xy')

    command = env.command_manager._terms['base_velocity']
    for name in ('vel_command_b', 'heading_target', 'is_heading_env', 'is_standing_env',
                 'time_left', 'command_counter'):
        field(command, name, 'command.' + name)
    sampler = getattr(command, 'sampler', None)
    if sampler is not None and hasattr(sampler, 'rngs'):
        for dest in destinations:
            sampler.rngs[dest].bit_generator.state = copy.deepcopy(sampler.rngs[source].bit_generator.state)
            if source in sampler._states:
                sampler._states[dest] = copy.deepcopy(sampler._states[source])
            if source in sampler._last_times:
                sampler._last_times[dest] = sampler._last_times[source]
        report['copied'].append('command.sampler.rng_and_held_state')
    else:
        report['unreplicated'].append('command.sampler.rng_and_held_state')

    action_manager = env.action_manager
    for name in ('_action', '_prev_action'):
        field(action_manager, name, 'action_manager.' + name)
    for term_name, term in action_manager._terms.items():
        for name in ('_raw_actions', '_processed_actions', '_pre_clip_processed_actions',
                     '_previous_targets', '_travel_direction'):
            value = getattr(term, name, None)
            if isinstance(value, torch.Tensor):
                field(term, name, f'action.{term_name}.{name}')

    for name, buffer in getattr(env, '_action_latency_buffers', {}).items():
        field(buffer, '_time_lags', f'delay.{name}.time_lags')
        ring = buffer._circular_buffer
        field(ring, '_num_pushes', f'delay.{name}.num_pushes')
        if isinstance(ring._buffer, torch.Tensor):
            field(ring, '_buffer', f'delay.{name}.ring', batch_axis=1)
        else:
            report['unreplicated'].append(f'delay.{name}.ring_uninitialized')
        report['copied'].append(f'delay.{name}.global_pointer={ring._pointer}')

    history = getattr(env.observation_manager, '_group_obs_term_history_buffer', {})
    for group, terms in history.items():
        for name, ring in terms.items():
            field(ring, '_num_pushes', f'obs_history.{group}.{name}.num_pushes')
            if isinstance(ring._buffer, torch.Tensor):
                field(ring, '_buffer', f'obs_history.{group}.{name}.ring', batch_axis=1)
            else:
                report['unreplicated'].append(f'obs_history.{group}.{name}.ring_uninitialized')
    report['unreplicated'] += [
        'PhysX_contact_warm_start_cache', 'contact_sensor_history', 'raycaster_history',
        'global_Python_NumPy_CPU_CUDA_RNG_not_per_env', 'follower_controller_state_requires_fresh_process']
    env.scene.write_data_to_sim()
    env.sim.forward()
    root_now = robot.data.root_state_w.clone()
    root_now[:, :3] -= origins
    fields = {'root_local_state': root_now, 'joint_position': robot.data.joint_pos,
              'joint_velocity': robot.data.joint_vel,
              'command': command.vel_command_b}
    report['initial_state_max_abs'] = {name: float((value[destinations]-value[source]).abs().max())
                                       for name, value in fields.items()}
    if max(report['initial_state_max_abs'].values()) > 1e-4:
        raise ValueError('branch state clone consistency mismatch')
    return report
