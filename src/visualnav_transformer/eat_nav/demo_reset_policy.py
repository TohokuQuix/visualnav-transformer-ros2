"""Demo-only fall reset policy and drift-safe sensor time mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math


def assert_finite_sim_state(robot) -> None:
    """Stop a numerically invalid demo rather than hiding it through reset."""
    import torch

    for name in ('root_state_w', 'joint_pos', 'joint_vel'):
        if not bool(torch.isfinite(getattr(robot.data, name)).all()):
            raise ValueError(f'nonfinite {name}; demo stopped without reset')


def configure_demo_reset_policy(cfg) -> None:
    """Allow only the existing tilt termination; fail on unknown reset terms."""
    terms = {name: value for name, value in vars(cfg.terminations).items()
             if not name.startswith('_') and value is not None}
    unexpected = set(terms) - {'time_out', 'non_finite_state', 'tilt_over'}
    if unexpected or 'tilt_over' not in terms:
        raise ValueError(f'unexpected/missing demo termination terms: {sorted(unexpected)}')
    cfg.terminations.time_out = None
    # A nonfinite state is a fatal diagnostic error, never an automatic reset.
    cfg.terminations.non_finite_state = None
    event = cfg.events.reset_flipper_fold_mode
    if event is None:
        raise ValueError('reset flipper fold mode event missing')
    event.params = {'modes': (0.,), 'probabilities': (1.,)}
    # The reset event uses [rear-right, rear-left, front-right, front-left]
    # and chooses [neutral, front-folded, rear-folded]. Apply this both at
    # initial env.reset() and at each tilt-triggered reset.
    joints = cfg.events.reset_robot_joints
    if joints is None or tuple(joints.params.get('pose_probabilities', ())) != (0., .5, .5):
        raise ValueError('unexpected/missing demo flipper pose reset event')
    if len(joints.params.get('flipper_joint_names', ())) != 4 or not math.isclose(
            float(joints.params.get('front_rest_angle', float('nan'))), 0.):
        raise ValueError('unexpected demo flipper joint reset geometry')
    joints.params['pose_probabilities'] = (0., 1., 0.)


def assert_demo_flipper_pose(robot, cfg, env_ids) -> list[list[float]]:
    """Verify the actual post-reset joint state, rear extended/front folded."""
    import torch

    params = cfg.events.reset_robot_joints.params
    joint_ids, _ = robot.find_joints(list(params['flipper_joint_names']), preserve_order=True)
    if len(joint_ids) != 4:
        raise ValueError('demo flipper pose requires four joints')
    actual = robot.data.joint_pos[env_ids][:, joint_ids]
    expected = torch.tensor([0., 0., float(params['folded_angle']),
                             float(params['folded_angle'])],
                            device=actual.device, dtype=actual.dtype)
    if not bool(torch.isfinite(actual).all()) or not bool(torch.all(
            torch.abs(actual - expected) <= 1e-3)):
        raise ValueError(f'demo flipper pose mismatch: {actual.detach().cpu().tolist()}')
    return actual.detach().cpu().tolist()


@dataclass
class DemoSensorClock:
    """Map the raycaster's float32 sample age to the integer demo /clock."""

    step_ns: int
    episode_origin_ns: int = 0

    def __post_init__(self):
        if self.step_ns <= 0:
            raise ValueError('step duration must be positive')

    def sensor_update_on_logical_clock(
            self, logical_ns: int, sensor_now_s: float, sensor_update_s: float) -> int:
        """Use two readings of the same Isaac clock so accumulated drift cancels."""
        if not all(math.isfinite(value) and value >= 0 for value in
                   (sensor_now_s, sensor_update_s)) or sensor_update_s > sensor_now_s:
            raise ValueError('sensor clock values are invalid or reversed')
        age_ns = round((sensor_now_s - sensor_update_s) * 1e9)
        return int(logical_ns) - age_ns

    def after_fall(self, completed_step: int) -> None:
        new_origin = (completed_step + 1) * self.step_ns
        if new_origin <= self.episode_origin_ns:
            raise ValueError('reset origin must increase')
        self.episode_origin_ns = new_origin
