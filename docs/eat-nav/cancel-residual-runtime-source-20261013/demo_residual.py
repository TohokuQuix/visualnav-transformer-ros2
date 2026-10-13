"""Read-only Cancel-aligned measurements from sealed EAT demo step traces."""

from __future__ import annotations

import math
from statistics import mean


DT_S = 0.05
_BINS = (("0.0-0.5", 0, 10), ("0.5-1.0", 10, 20),
         ("1.0-2.0", 20, 40), ("2.0+", 40, None))


def _length(rows):
    return sum(math.dist(r['pose_pre'][:2], r['pose_post'][:2]) for r in rows)


def capture_physical_state(robot, contact_sensor=None) -> dict:
    """Read simulated state without advancing sensors or changing control."""
    import torch

    def values(tensor):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('nonfinite simulated physical state')
        return tensor.detach().cpu().tolist()

    data = robot.data
    return {
        'root_lin_vel_w': values(data.root_lin_vel_w[0]),
        'root_ang_vel_w': values(data.root_ang_vel_w[0]),
        'joint_names': list(robot.joint_names),
        'joint_pos': values(data.joint_pos[0]),
        'joint_vel': values(data.joint_vel[0]),
        'applied_torque': values(data.applied_torque[0]) if hasattr(data, 'applied_torque') else None,
        'torque_meaning': 'Isaac ArticulationData.applied_torque; simulated applied joint torque, not measured effort',
        'contact_net_forces_w': (
            values(contact_sensor.data.net_forces_w[0]) if contact_sensor is not None else None),
        'contact_meaning': 'Isaac ContactSensorData.net_forces_w by body; net simulated contact force',
    }


def summarize_cancel(records: list[dict]) -> dict:
    """Summarize the first Cancel without conflating path length and net motion."""
    events = [r for r in records if r.get('event') == 'cancel']
    if len(events) != 1:
        raise ValueError('expected exactly one Cancel event')
    cancel = events[0]
    steps = {r['step']: r for r in records if r.get('event') == 'step'}
    post = [steps[index] for index in sorted(steps) if index >= cancel['step']]
    if not post or post[0]['step'] != cancel['step'] or any(
            right['step'] != left['step'] + 1 or
            right['stamp_ns'] != left['stamp_ns'] + 50_000_000 or
            math.dist(left['pose_post'][:2], right['pose_pre'][:2]) > 1e-5
            for left, right in zip(post, post[1:])):
        raise ValueError('noncontiguous Cancel trace')
    start = post[0]['pose_pre']
    end = post[-1]['pose_post']
    lengths = {name: _length(post[begin:finish]) for name, begin, finish in _BINS}
    last = post[-1]
    measured = last.get('root_lin_vel_w_post')
    if measured is None:
        speed = math.dist(last['pose_pre'][:2], last['pose_post'][:2]) / DT_S
        source = 'finite_difference_pose'
    else:
        speed = math.hypot(*measured[:2])
        source = 'robot.data.root_lin_vel_w_post'
    final_window = post[-10:]
    return {
        'cancel_step': cancel['step'], 'cancel_time_s': cancel['stamp_ns'] / 1e9,
        'post_cancel_steps': len(post), 'cancel_pose_map': start,
        'last_pose_map': end, 'cumulative_xy_m': _length(post),
        'net_xy_m': math.dist(start[:2], end[:2]),
        'max_xy_displacement_m': max(math.dist(start[:2], r['pose_post'][:2]) for r in post),
        'yaw_change_rad': end[2] - start[2], 'segments_m': lengths,
        'last_0_5s_mean_xy_speed_mps': _length(final_window) / (len(final_window) * DT_S),
        'last_xy_speed_mps': speed, 'last_root_velocity_source': source,
        'post_cancel_command_nonzero_steps': sum(any(abs(x) > 1e-6 for x in r['command']) for r in post),
        'post_cancel_actor_input_nonzero_steps': sum(any(abs(x) > 1e-6 for x in r['actor_input_command']) for r in post),
        'post_cancel_done_steps': sum(bool(r['done']) for r in post),
        'action_mean_abs': [mean(abs(r['actor_action'][j]) for r in post) for j in range(6)],
        'source_step_count': len(steps),
    }
