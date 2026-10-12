"""Synchronous four-env Path-to-command control, independent of ROS and Isaac."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .p1_follower_core import PureFollower


STEP_NS = 50_000_000
FRAME = 'tile_local'


@dataclass(frozen=True)
class FollowerRow:
    env_id: int
    stamp_ns: int
    pose: tuple[float, float, float]
    command: tuple[float, float, float]
    diagnostic: dict
    frame: str = FRAME


class RosFreeG3Control:
    """Issue one command per env/stamp, then enforce injection → Actor → physics."""

    def __init__(self, paths, *, library, follower_type=PureFollower):
        self._library = library
        self._follower_type = follower_type
        self.reset_episode(paths)

    def reset_episode(self, paths):
        if len(paths) != 4 or any(len(path) < 2 for path in paths):
            raise ValueError('four nonempty Paths required')
        for follower in getattr(self, 'followers', ()):
            if hasattr(follower, 'close'):
                follower.close()
        self.followers = [self._follower_type(path, library=self._library) for path in paths]
        if len({id(follower) for follower in self.followers}) != 4:
            raise ValueError('core instances must be env-independent')
        handles = [getattr(follower, '_instance', None) for follower in self.followers]
        if all(handle is not None for handle in handles) and len(set(handles)) != 4:
            raise ValueError('C++ core handles must be env-independent')
        self.last_stamp_ns = None
        self.phase = 'ready'
        self.rows = None

    def issue(self, stamp_ns, poses, *, frame):
        if frame != FRAME:
            raise ValueError('pose frame must be tile_local')
        if self.phase not in ('ready', 'physics_done'):
            raise ValueError('control phase requires completed prior physics')
        if not isinstance(stamp_ns, int) or stamp_ns < 0:
            raise ValueError('logical stamp must be a nonnegative integer')
        if self.last_stamp_ns is not None:
            delta = stamp_ns - self.last_stamp_ns
            if delta == 0:
                raise ValueError('duplicate core call stamp')
            if delta != STEP_NS:
                raise ValueError('core calls require exactly 50 ms increments')
        if len(poses) != 4 or any(len(pose) != 3 or
                                  not all(math.isfinite(float(x)) for x in pose)
                                  for pose in poses):
            raise ValueError('four finite tile-local poses required')
        rows = []
        for env_id, (follower, pose) in enumerate(zip(self.followers, poses)):
            pose = tuple(float(x) for x in pose)
            command, diagnostic = follower.step_diagnostic(*pose, stamp_ns)
            if len(command) != 3 or not all(math.isfinite(float(x)) for x in command):
                raise ValueError('core returned invalid command')
            command = tuple(float(x) for x in command)
            if diagnostic['goal_reached'] and command != (0., 0., 0.):
                raise ValueError('Goal must hold zero command')
            rows.append(FollowerRow(env_id, stamp_ns, pose, command, diagnostic))
        self.last_stamp_ns = stamp_ns
        self.rows = tuple(rows)
        self.phase = 'issued'
        return self.rows

    def inject(self, rows, callback, *, stamp, suppressed_envs=()):
        self._require('issued', stamp)
        if rows is not self.rows:
            raise ValueError('injection rows differ from current core call')
        suppressed = set(suppressed_envs)
        if not suppressed.issubset({0, 1, 2, 3}):
            raise ValueError('invalid suppressed env id')
        commands = tuple((0., 0., 0.) if i in suppressed else row.command
                         for i, row in enumerate(rows))
        actual = callback(commands)
        if actual is None or len(actual) != 4 or any(len(row) != 3 for row in actual):
            raise ValueError('command injection evidence missing')
        if any(not math.isfinite(float(actual[i][j])) or
               abs(float(actual[i][j]) - commands[i][j]) > 1e-6
               for i in range(4) for j in range(3)):
            raise ValueError('command injection mismatch')
        self.phase = 'injected'
        return actual

    def mark_actor(self, stamp, callback):
        self._require('injected', stamp)
        result = callback()
        self.phase = 'actor_done'
        return result

    def mark_physics(self, stamp, callback):
        self._require('actor_done', stamp)
        result = callback()
        self.phase = 'physics_done'
        return result

    def _require(self, phase, stamp):
        if self.phase != phase or self.last_stamp_ns != stamp:
            raise ValueError(f'control phase/stamp mismatch: expected {phase} at {self.last_stamp_ns}')


def command_term_slice(manager, group):
    """Locate the generated-command term from the resolved observation contract."""
    names = manager._group_obs_term_names[group]
    dimensions = manager._group_obs_term_dim[group]
    if names.count('velocity_commands') != 1:
        raise ValueError(f'{group} command observation term missing/duplicate')
    index = names.index('velocity_commands')
    widths = [math.prod(dim) for dim in dimensions]
    if widths[index] != 3:
        raise ValueError(f'{group} command observation must have width three')
    cfg = manager._group_obs_term_cfgs[group][index]
    if getattr(cfg, 'noise', None) is not None or getattr(cfg, 'scale', None) not in (None, 1.0):
        raise ValueError(f'{group} command observation has unsupported noise/scale')
    return slice(sum(widths[:index]), sum(widths[:index+1]))


def reconcile_command_observation(obs, observed, bare, commands, previous_raw):
    """Replace only generated-command terms after injection, without a second obs sample.

    The previous env.step produced obs before the next pose-dependent Path command
    existed. Actor and Critic must see this stamp's command. Other observation
    terms, including their paired noise, remain the original one-pass sample.
    """
    import torch

    manager = bare.observation_manager
    actor = command_term_slice(manager, 'policy')
    critic = command_term_slice(manager, 'critic')
    target = torch.as_tensor(commands, dtype=obs['policy'].dtype, device=obs['policy'].device)
    if tuple(target.shape) != (4, 3) or not bool(torch.isfinite(target).all()):
        raise ValueError('invalid commands for Actor observation')
    if not bool(torch.allclose(
            bare.command_manager._terms['base_velocity'].vel_command_b[:4], target,
            atol=1e-6, rtol=0)):
        raise ValueError('Actor command term differs from injected command')
    previous = bare._latency_policy_obs_prev
    if previous is not None and previous_raw is not None:
        prior_raw = previous_raw[:4, actor].clone()
        alpha = (bare._onix_latency_seconds[:4] / max(float(bare.step_dt), 1e-9)).clamp(0., 1.)
        delayed = target + alpha * (prior_raw - target)
        previous[:4, actor] = target
    else:
        prior_raw = target.clone()
        alpha = torch.zeros((4, 1), dtype=target.dtype, device=target.device)
        delayed = target
    obs['policy'][:4, actor] = delayed
    obs['critic'][:4, critic] = target
    for key in ('policy_pre_noise', 'policy_post_noise_pre_delay'):
        observed[key][:4, actor] = target
    observed['policy_actor_input'][:4, actor] = delayed
    observed['critic_observation'][:4, critic] = target
    return {'actor_slice': [actor.start, actor.stop],
            'critic_slice': [critic.start, critic.stop],
            'injected': target.detach().cpu().tolist(),
            'actor_seen': delayed.detach().cpu().tolist(),
            'previous_raw': prior_raw.detach().cpu().tolist(),
            'latency_alpha': alpha.detach().cpu().reshape(4).tolist(),
            'observation_sample_count_added': 0}


class RosFreeSameEpisodeGate:
    def __init__(self, initial_lengths, max_episode_length):
        if len(initial_lengths) != 4 or min(initial_lengths) < 0 or \
                max(initial_lengths) + 640 >= max_episode_length:
            raise ValueError('insufficient same-episode margin')
        self.initial = tuple(initial_lengths)
        self.next_step = 0
        self.last_stamp = None

    def accept(self, step, stamp, rows, before, after, done, reset_calls):
        if (step != self.next_step or step >= 640 or len(rows) != 4 or
                len(before) != 4 or len(after) != 4 or len(done) != 4):
            raise ValueError('noncontiguous four-env control step')
        if self.last_stamp is not None and stamp != self.last_stamp + STEP_NS:
            raise ValueError('noncontiguous 50 ms logical stamp')
        if any(row.env_id != i or row.stamp_ns != stamp for i, row in enumerate(rows)):
            raise ValueError('follower stamp/env mismatch')
        if reset_calls or any(done):
            raise ValueError('same-episode reset/termination')
        if any(before[i] != self.initial[i] + step or
               after[i] != self.initial[i] + step + 1 for i in range(4)):
            raise ValueError('episode length discontinuity')
        self.next_step += 1
        self.last_stamp = stamp
