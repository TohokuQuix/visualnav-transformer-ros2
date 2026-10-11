"""Explicitly pair the existing additive-uniform Actor observation noise.

One original Isaac Lab noise-function call is made for each logical step and
term using a key independent of condition and term call order.  Its sampled
noise is broadcast to every condition; state-dependent clean data still differ.
"""

from __future__ import annotations

import hashlib
import torch


EXPECTED = ('base_ang_vel', 'joint_pos', 'joint_vel', 'projected_gravity')


def key_seed(group: str, block_seed: int, step: int, term: str) -> int:
    payload = f'p1-paired-v1|{group}|{block_seed}|{step}|{term}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)


class PairedPolicyNoise:
    def __init__(self, manager, group: str, block_seed: int):
        self.group = group
        self.block_seed = block_seed
        self.step = None
        self.samples = {}
        self.originals = []
        names = manager._group_obs_term_names['policy']
        cfgs = manager._group_obs_term_cfgs['policy']
        active = []
        for name, cfg in zip(names, cfgs):
            noise = getattr(cfg, 'noise', None)
            func = getattr(noise, 'func', None)
            if not callable(func):
                continue
            active.append(name)
            if noise.operation != 'add' or not hasattr(noise, 'n_min') or not hasattr(noise, 'n_max'):
                raise ValueError(f'unsupported Actor noise semantics: {name}')
            self.originals.append((noise, func))
            noise.func = self._wrapper(name, func)
        if set(active) != set(EXPECTED):
            self.close()
            raise ValueError(f'Actor noise term drift: {active}')

    def set_step(self, step: int):
        if step < 0:
            raise ValueError('negative logical step')
        self.step = step
        self.samples = {}

    def _wrapper(self, name, original):
        def apply(data, cfg):
            if self.step is None:
                raise ValueError('paired noise logical step not set')
            if name in self.samples:
                raise ValueError(f'Actor noise term repeated in one step: {name}')
            device = data.device
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(key_seed(self.group, self.block_seed, self.step, name))
                clean = torch.zeros_like(data[:1])
                sampled = original(clean, cfg) - clean
            self.samples[name] = sampled.detach().clone()
            return data + sampled.expand_as(data)
        return apply

    def verify_step(self):
        if set(self.samples) != set(EXPECTED):
            raise ValueError(f'Actor noise missing or duplicated: {sorted(self.samples)}')
        return {name: tensor.clone() for name, tensor in self.samples.items()}

    def close(self):
        for noise, func in self.originals:
            noise.func = func
