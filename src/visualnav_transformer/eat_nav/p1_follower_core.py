"""Python binding for the ROS-free C++ P1 follower calculation core."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess


SOURCE = Path(__file__).with_suffix('.cpp')


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

    def state(self) -> tuple[float, float, float, float]:
        values = (ctypes.c_double * 4)()
        if self._lib.p1_follower_get_state(self._instance, values):
            raise RuntimeError('follower state unavailable')
        return tuple(values)

    def restore(self, state):
        if len(state) != 4:
            raise ValueError('follower state has four components')
        values = (ctypes.c_double * 4)(*state)
        if self._lib.p1_follower_set_state(self._instance, values):
            raise ValueError('invalid follower state')

    def close(self):
        if self._instance:
            self._lib.p1_follower_destroy(self._instance)
            self._instance = None

    def __del__(self):
        self.close()
