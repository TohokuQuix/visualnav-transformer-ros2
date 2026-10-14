"""Pre-noise 22-term policy observation from explicit body-frame quantities."""

import numpy as np


FLIPPERS = ('rr', 'rl', 'fr', 'fl')


def _values(mapping, keys):
    try:
        values = [float(mapping[key]) for key in keys]
    except (KeyError, TypeError) as error:
        raise ValueError(f'missing required Actor term: {error}') from error
    if not np.isfinite(values).all():
        raise ValueError('nonfinite Actor term')
    return values


def build_actor22(*, angular_velocity_body, gravity_body, command_body,
                  fold_mode, flipper_position, flipper_default,
                  track_velocity, flipper_effort, track_effort) -> np.ndarray:
    """Apply resolved OnixRoughEnvCfg policy term order/scales, without noise."""
    angular = np.asarray(angular_velocity_body, dtype=np.float64)
    gravity = np.asarray(gravity_body, dtype=np.float64)
    command = np.asarray(command_body, dtype=np.float64)
    if any(value.shape != (3,) or not np.isfinite(value).all()
           for value in (angular, gravity, command)) or fold_mode not in (-1, 0, 1):
        raise ValueError('Actor body vectors/fold mode')
    positions = np.asarray(_values(flipper_position, FLIPPERS)) - \
                np.asarray(_values(flipper_default, FLIPPERS))
    try:
        velocities = [np.mean(track_velocity[side]) for side in ('right', 'left')]
        efforts = _values(flipper_effort, FLIPPERS) + \
            [sum(track_effort[side]) for side in ('right', 'left')]
    except (KeyError, TypeError) as error:
        raise ValueError(f'missing track Actor term: {error}') from error
    result = np.concatenate((np.clip(angular, -100, 100)*.25,
                             np.clip(gravity, -1, 1),
                             np.clip(command, -100, 100),
                             [float(fold_mode)],
                             np.clip(positions, -100, 100),
                             np.clip(velocities, -100, 100)*.05,
                             np.clip(efforts, -100, 100)*.01)).astype(np.float32)
    if result.shape != (22,) or not np.isfinite(result).all():
        raise ValueError('Actor 22D shape/finite')
    return result
