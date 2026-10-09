"""Runtime-independent geometry reuse and fail-before-Path validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math

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


def validate_candidate_free_trace(rows: list[dict]) -> dict:
    """Check the 120-step logging path without treating it as a Path Outcome."""
    if len(rows) != 120 or [row.get('control_step') for row in rows] != list(range(120)):
        raise ValueError('candidate-free trace must cover steps 0..119')
    fields={'actor_input':22,'critic_observation':267,'actor_action':6,
            'command':3,'root_state':13,'post_root_state':13}
    for row in rows:
        for name,width in fields.items():
            values=row.get(name)
            if not isinstance(values,list) or len(values)!=width or not all(
                    isinstance(value,(int,float)) and math.isfinite(value) for value in values):
                raise ValueError(f'candidate-free {name} shape/nonfinite')
        reward=row.get('reward')
        if not isinstance(reward,(int,float)) or not math.isfinite(reward):
            raise ValueError('candidate-free reward nonfinite')
    return {'control_steps':120,'finite':True,'shape':{'actor':22,'critic':267,'action':6}}
