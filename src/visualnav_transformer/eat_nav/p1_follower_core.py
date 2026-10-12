"""Python binding for the ROS-free C++ P1 follower calculation core."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess


SOURCE = Path(__file__).with_suffix('.cpp')
DIAGNOSTIC_NAMES = (
    'carrot_x', 'carrot_y', 'projection_x', 'projection_y', 'remaining_path_distance',
    'heading_error', 'travel_direction', 'goal_reached', 'linear_recovery',
    'angular_recovery', 'linear_recovery_started', 'linear_recovery_finished',
    'angular_recovery_started', 'recovery_cause', 'commanded_distance',
    'actual_distance', 'commanded_angular_distance', 'actual_angular_distance',
    'motion_ready', 'best_remaining_path_distance', 'progress_age_sec',
    'backtrack_points', 'backtrack_length', 'use_goal_extension', 'episode_elapsed_sec',
    'pre_motion_samples', 'pre_motion_oldest_ns', 'pre_motion_newest_ns',
    'pre_motion_ready', 'pre_commanded_distance', 'pre_actual_distance',
    'pre_motion_ratio', 'goal_distance', 'turn_in_place', 'projection_segment',
)


def compile_core(output_dir: Path) -> Path:
    """Build a content-addressed shared library; never modify the ROS follower."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    library = output_dir / f'libp1_follower_{name}.so'
    if not library.exists():
        temporary = library.with_suffix('.so.tmp')
        subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared',
                        str(SOURCE), '-o', str(temporary)], check=True)
        temporary.replace(library)
    return library


class PureFollower:
    def __init__(self, path_xy, *, library: Path):
        self._lib = ctypes.CDLL(str(library))
        self._lib.p1_follower_create.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        self._lib.p1_follower_create.restype = ctypes.c_void_p
        self._lib.p1_follower_step.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                               ctypes.c_double, ctypes.c_double,
                                               ctypes.c_longlong, ctypes.POINTER(ctypes.c_double)]
        self._lib.p1_follower_step.restype = ctypes.c_int
        self._lib.p1_follower_step_diagnostic.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_longlong, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ]
        self._lib.p1_follower_step_diagnostic.restype = ctypes.c_int
        self._lib.p1_follower_diagnostic_count.restype = ctypes.c_size_t
        self._lib.p1_follower_clone.argtypes = [ctypes.c_void_p]
        self._lib.p1_follower_clone.restype = ctypes.c_void_p
        if self._lib.p1_follower_diagnostic_count() != len(DIAGNOSTIC_NAMES):
            raise RuntimeError('follower diagnostic ABI mismatch')
        self._lib.p1_follower_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self._lib.p1_follower_get_state.restype = ctypes.c_int
        self._lib.p1_follower_set_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self._lib.p1_follower_set_state.restype = ctypes.c_int
        self._lib.p1_follower_destroy.argtypes = [ctypes.c_void_p]
        values = [float(v) for xy in path_xy for v in xy]
        self._path = (ctypes.c_double * len(values))(*values)
        self._instance = self._lib.p1_follower_create(self._path, len(path_xy))
        if not self._instance:
            raise ValueError('invalid Path')

    def step(self, x: float, y: float, yaw: float, stamp_ns: int):
        command = (ctypes.c_double * 3)()
        result = self._lib.p1_follower_step(self._instance, x, y, yaw,
                                            int(stamp_ns), command)
        if result:
            raise RuntimeError(f'follower unsupported/invalid state: {result}')
        return tuple(command)

    def step_diagnostic(self, x: float, y: float, yaw: float, stamp_ns: int):
        """One exact-stamp control step; no ROS clock or callbacks are involved."""
        command = (ctypes.c_double * 3)()
        values = (ctypes.c_double * len(DIAGNOSTIC_NAMES))()
        result = self._lib.p1_follower_step_diagnostic(
            self._instance, x, y, yaw, int(stamp_ns), command, values, len(values))
        if result:
            raise RuntimeError(f'follower unsupported/invalid state: {result}')
        diagnostic = dict(zip(DIAGNOSTIC_NAMES, values))
        for name in ('goal_reached', 'linear_recovery', 'angular_recovery',
                     'linear_recovery_started', 'linear_recovery_finished',
                     'angular_recovery_started', 'motion_ready', 'use_goal_extension',
                     'pre_motion_ready', 'turn_in_place'):
            diagnostic[name] = bool(diagnostic[name])
        diagnostic['backtrack_points'] = int(diagnostic['backtrack_points'])
        diagnostic['pre_motion_samples'] = int(diagnostic['pre_motion_samples'])
        diagnostic['projection_segment'] = int(diagnostic['projection_segment'])
        diagnostic['recovery_cause'] = {0: None, 1: 'progress_stall',
                                        2: 'linear_stall'}[int(diagnostic['recovery_cause'])]
        return tuple(command), diagnostic

    def state(self) -> tuple[float, float, float, float]:
        """Legacy four-field nominal state; use clone() for full recovery state."""
        values = (ctypes.c_double * 4)()
        if self._lib.p1_follower_get_state(self._instance, values):
            raise RuntimeError('follower state unavailable')
        return tuple(values)

    def restore(self, state):
        """Restore legacy nominal fields only; not a long-horizon state copy."""
        if len(state) != 4:
            raise ValueError('follower state has four components')
        values = (ctypes.c_double * 4)(*state)
        if self._lib.p1_follower_set_state(self._instance, values):
            raise ValueError('invalid follower state')

    def clone(self):
        """Deep-copy all path, clock, motion, progress and recovery history."""
        copied = self._lib.p1_follower_clone(self._instance)
        if not copied:
            raise RuntimeError('follower clone failed')
        other = object.__new__(PureFollower)
        other._lib = self._lib
        other._path = self._path
        other._instance = copied
        return other

    def close(self):
        if self._instance:
            self._lib.p1_follower_destroy(self._instance)
            self._instance = None

    def __del__(self):
        self.close()
