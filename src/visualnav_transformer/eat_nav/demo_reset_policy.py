"""Demo-only fall reset policy and mapping of episode-local sensor time."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class DemoSensorClock:
    """Place Isaac sensor timestamps on the uninterrupted demo /clock timeline."""

    step_ns: int
    episode_origin_ns: int = 0

    def __post_init__(self):
        if self.step_ns <= 0:
            raise ValueError('step duration must be positive')

    def global_sensor_stamp(self, episode_sensor_ns: int) -> int:
        if episode_sensor_ns < 0:
            raise ValueError('sensor timestamp must be nonnegative')
        return self.episode_origin_ns + episode_sensor_ns

    def after_fall(self, completed_step: int) -> None:
        new_origin = (completed_step + 1) * self.step_ns
        if new_origin <= self.episode_origin_ns:
            raise ValueError('reset origin must increase')
        self.episode_origin_ns = new_origin
