"""Step-local PhysX sampling and time contract for P1 teacher logging."""

from __future__ import annotations

import math

from .p1_teacher_capture import SubstepContactAccumulator


class PhysicsSubstepTap:
    def __init__(self, sim, contact_view, *, num_envs, body_count, physics_dt,
                 substeps, stamp_ns):
        self.sim = sim
        self.view = contact_view
        self.num_envs = num_envs
        self.body_count = body_count
        self.physics_dt = physics_dt
        self.substeps = substeps
        self.stamp_ns = stamp_ns
        self.acc = [SubstepContactAccumulator(body_count, physics_dt, substeps)
                    for _ in range(num_envs)]
        self.original = None

    def __enter__(self):
        if self.original is not None or not callable(getattr(self.view, 'get_net_contact_forces', None)):
            raise ValueError('physx_contact_view_unavailable')
        self.original = self.sim.step

        def stepped(*args, **kwargs):
            self.original(*args, **kwargs)
            index = len(self.acc[0].samples)
            if index >= self.substeps:
                raise ValueError('too_many_physics_substeps')
            force = self.view.get_net_contact_forces(dt=self.physics_dt)
            if force.shape != (self.num_envs * self.body_count, 3):
                raise ValueError('physx_contact_shape')
            frame = force.reshape(self.num_envs, self.body_count, 3)
            stamp = self.stamp_ns + round((index + 1) * self.physics_dt * 1e9)
            for env, accumulator in enumerate(self.acc):
                accumulator.accept(frame[env], index, stamp)

        self.sim.step = stepped
        return self

    def finish(self):
        return [accumulator.finish() for accumulator in self.acc]

    def __exit__(self, *_):
        if self.original is not None:
            self.sim.step = self.original
            self.original = None


def validate_teacher_step(row):
    step, stamp = row['step'], row['logical_stamp_ns']
    reward = row['reward']
    sub = row['substep']
    if (reward['step'] != step or reward['logical_stamp_ns'] != stamp or
            reward['reward_compute_phase'] != 'post_physics_pre_command_update'):
        raise ValueError('reward_alignment')
    dt = row['control_dt_s']
    if not math.isclose(dt, 0.05, abs_tol=1e-9) or sub['sample_count'] != 10:
        raise ValueError('control_or_physics_count')
    expected = [stamp + 5_000_000 * (i + 1) for i in range(10)]
    if sub['physics_stamp_ns'] != expected:
        raise ValueError('substep_stamp_alignment')
