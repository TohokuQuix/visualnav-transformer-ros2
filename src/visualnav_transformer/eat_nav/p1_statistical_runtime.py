"""Runtime-independent geometry reuse and fail-before-Path validation."""

from __future__ import annotations

import copy
import hashlib
import json

from .p1_statistical_protocol import initialization_checks, terrain_identity_checks


def _as_list(value):
    return value.tolist() if hasattr(value, 'tolist') else value


def jsonable_values(value):
    """Convert tensor/NumPy values at the simulator-to-JSON boundary."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


class CachedTerrainGeometry:
    """Generate one canonical mesh, then return a fresh copy for each tile."""

    def __init__(self, generate):
        self._generate = generate
        self._mesh = None
        self._origin = None
        self.canonical_sha256 = None

    def copy_for_slot(self):
        if self._mesh is None:
            self._mesh, self._origin = self._generate()
            representation = {'vertices': _as_list(self._mesh.vertices),
                              'faces': _as_list(getattr(self._mesh, 'faces', [])),
                              'origin': _as_list(self._origin)}
            raw = json.dumps(representation, sort_keys=True, separators=(',', ':')).encode()
            self.canonical_sha256 = hashlib.sha256(raw).hexdigest()
        return self._mesh.copy(), copy.deepcopy(self._origin)


def require_prepath_gate(records: list[dict], limits: dict) -> dict:
    terrain = terrain_identity_checks(records)
    if not terrain['pass']:
        raise ValueError('terrain_identity:' + ','.join(terrain['failures']))
    initialization = initialization_checks(records, limits)
    if not initialization['pass']:
        raise ValueError('controlled_initialization:' + ','.join(initialization['failures']))
    return {'pass': True, 'terrain': terrain, 'initialization': initialization,
            'Path_publications_before_gate': 0}
