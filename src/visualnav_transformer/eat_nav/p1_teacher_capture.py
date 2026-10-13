"""Fail-closed, non-mutating captures for P1 teacher evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


SCHEMA = 'eatnav_p1_teacher_capture_v1'
PAIN_SCHEMA = 'eatnav_pain_allowables_v1'


@dataclass(frozen=True)
class RewardTermCapture:
    names: tuple[str, ...]
    weights: tuple[float, ...]
    dt: float
    atol: float = 1e-5
    rtol: float = 1e-5

    @classmethod
    def from_manager(cls, manager, dt: float):
        names = tuple(manager.active_terms)
        weights = tuple(float(manager.get_term_cfg(name).weight) for name in names)
        if not names or not math.isfinite(dt) or dt <= 0 or not all(map(math.isfinite, weights)):
            raise ValueError('invalid_reward_contract')
        return cls(names, weights, dt)

    def manifest(self):
        return {'schema': SCHEMA, 'terms': [{'index': i, 'name': name,
                 'resolved_weight': self.weights[i], 'raw_available': self.weights[i] != 0.0}
                 for i, name in enumerate(self.names)], 'dt_s': self.dt,
                'manager_step_reward_meaning': 'weighted_rate; contribution=step_reward*dt',
                'sum_atol': self.atol, 'sum_rtol': self.rtol}

    def capture_step(self, manager, returned_reward, step: int, stamp_ns: int):
        if tuple(manager.active_terms) != self.names or any(
            float(manager.get_term_cfg(name).weight) != weight
            for name, weight in zip(self.names, self.weights)
        ):
            raise ValueError('reward_weight_changed')
        rates = manager._step_reward.detach()
        total = returned_reward.detach()
        if rates.ndim != 2 or rates.shape[1] != len(self.names) or total.shape != (rates.shape[0],):
            raise ValueError('reward_shape')
        if not bool(torch.isfinite(rates).all()) or not bool(torch.isfinite(total).all()):
            raise ValueError('reward_nonfinite')
        weighted = rates * self.dt
        if not bool(torch.allclose(weighted.sum(dim=1), total, atol=self.atol, rtol=self.rtol)):
            raise ValueError('reward_sum_mismatch')
        weighted_cpu = weighted.cpu().tolist()
        rates_cpu = rates.cpu().tolist()
        raw = [[rate / weight if weight != 0.0 else None
                for rate, weight in zip(row, self.weights)] for row in rates_cpu]
        termination_indices = [i for i, name in enumerate(self.names)
                               if 'termination' in name or 'is_terminated' in name]
        return {'schema': SCHEMA, 'step': step, 'logical_stamp_ns': stamp_ns,
                'reward_compute_phase': 'post_physics_pre_command_update',
                'raw_value': raw, 'weighted_contribution': weighted_cpu,
                'termination_contribution': [sum(row[i] for i in termination_indices)
                                             for row in weighted_cpu],
                'total_reward': total.cpu().tolist()}


def joint_body_maps(joint_names, body_names, actuator_joint_names):
    if len(set(joint_names)) != len(joint_names) or len(set(body_names)) != len(body_names):
        raise ValueError('duplicate_index_name')
    unknown = {joint for names in actuator_joint_names.values() for joint in names} - set(joint_names)
    if unknown:
        raise ValueError('unresolved_actuator_joint')
    return {'joints': [{'index': i, 'name': name,
                       'actuator_terms': [term for term, names in actuator_joint_names.items()
                                          if name in names]}
                      for i, name in enumerate(joint_names)],
            'bodies': [{'index': i, 'name': name} for i, name in enumerate(body_names)]}


class SubstepContactAccumulator:
    """Accumulate actual physics-callback samples, never cached 50 ms sensor history."""

    def __init__(self, body_count: int, physics_dt: float, substeps: int):
        if body_count <= 0 or substeps <= 0 or physics_dt <= 0:
            raise ValueError('substep_contract')
        self.body_count = body_count
        self.physics_dt = physics_dt
        self.substeps = substeps
        self.samples = []
        self.stamps = []
        self.finalized = False

    def accept(self, force, substep: int, physics_stamp_ns: int):
        if self.finalized:
            raise ValueError('already_finalized')
        if substep != len(self.samples) or substep >= self.substeps or (
            self.stamps and physics_stamp_ns <= self.stamps[-1]
        ):
            raise ValueError('substep_order')
        tensor = torch.as_tensor(force).detach()
        if tensor.shape != (self.body_count, 3) or not bool(torch.isfinite(tensor).all()):
            raise ValueError('substep_force_shape_or_nonfinite')
        self.samples.append(tensor.to(device='cpu', dtype=torch.float32).clone())
        self.stamps.append(int(physics_stamp_ns))

    def finish(self):
        if len(self.samples) != self.substeps:
            raise ValueError('incomplete_substeps')
        if self.finalized:
            raise ValueError('already_finalized')
        self.finalized = True
        force = torch.stack(self.samples)
        magnitude = torch.linalg.vector_norm(force, dim=2)
        peak, indices = magnitude.max(dim=0)
        bodies = torch.arange(self.body_count)
        integral = force.sum(dim=0) * self.physics_dt
        return {'schema': SCHEMA, 'sample_count': self.substeps,
                'physics_stamp_ns': self.stamps,
                'peak_magnitude_N': peak.tolist(),
                'peak_force_world_N': force[indices, bodies].tolist(),
                'peak_substep': indices.tolist(),
                'integral_force_Ns': integral.tolist(),
                'contact_substep_count': (magnitude > 0).sum(dim=0).tolist(),
                'missing_substeps': 0, 'overflow': False,
                'source': 'physx_contact_view_after_each_sim_step'}


def _percentile(values, q):
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def compute_pain(loads, allowables, *, joint_names=None, body_names=None, dt=0.05):
    if allowables.get('schema') != PAIN_SCHEMA:
        raise ValueError('allowables_schema')
    joint_names = list(joint_names or [])
    body_names = list(body_names or [])
    tau_ref = allowables.get('joint_torque_ref_Nm', {})
    force_ref = allowables.get('body_force_ref_N', {})
    joint_w = allowables.get('joint_weights', {})
    body_w = allowables.get('body_weights', {})
    missing = ([f'joint:{name}' for name in joint_names
                if name not in tau_ref or name not in joint_w] +
               [f'body:{name}' for name in body_names
                if name not in force_ref or name not in body_w])
    if not joint_names or not body_names:
        missing += ['index_mapping']
    if missing:
        return {'status': 'missing_allowables', 'missing': missing, 'pain': None}
    for mapping in (tau_ref, force_ref, joint_w, body_w):
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in mapping.values()):
            raise ValueError('invalid_allowable_or_weight')
    torque = loads['torque_Nm']
    force = loads['body_force_world_N']
    if not torque or len(torque) != len(force) or dt <= 0:
        raise ValueError('load_window')
    joint_parts = []
    body_parts = []
    joint_peak = 0.0
    body_peak = 0.0
    exceed = 0
    for tau_row, force_row in zip(torque, force):
        if len(tau_row) != len(joint_names) or len(force_row) != len(body_names):
            raise ValueError('load_mapping_shape')
        ju = [abs(float(tau)) / tau_ref[name] for tau, name in zip(tau_row, joint_names)]
        fu = [math.dist(vec, (0, 0, 0)) / force_ref[name]
              for vec, name in zip(force_row, body_names)]
        if not all(map(math.isfinite, ju + fu)):
            raise ValueError('load_nonfinite')
        joint_parts.append(sum(joint_w[name] * u * u for name, u in zip(joint_names, ju)))
        body_parts.append(sum(body_w[name] * u * u for name, u in zip(body_names, fu)))
        joint_peak = max(joint_peak, *ju)
        body_peak = max(body_peak, *fu)
        exceed += max(ju + fu) >= 1.0
    return {'status': 'computed_synthetic_only',
            'joint_component': joint_parts, 'body_component': body_parts,
            'joint_mean': sum(joint_parts) / len(joint_parts),
            'body_mean': sum(body_parts) / len(body_parts),
            'joint_peak': max(joint_parts), 'body_peak': max(body_parts),
            'joint_p95': _percentile(joint_parts, 0.95),
            'body_p95': _percentile(body_parts, 0.95),
            'joint_peak_utilization': joint_peak, 'body_peak_utilization': body_peak,
            'exceedance_duration_s': exceed * dt,
            'combined_label': None,
            'note': 'synthetic test only until allowables have engineering provenance'}
