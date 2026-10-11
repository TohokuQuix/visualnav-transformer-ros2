"""Fail-closed, Isaac-independent material evidence contract for G3 crossover v2."""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

from .p1_g3_material_diagnostic import map_effective_materials

SCHEMA = 'eatnav_p1_g3_path_crossover_v2_material_audit'
MATERIAL_SCHEMA = 'eatnav_p1_live_material_audit_v2'
RANGE_NAMES = ('static_friction_range', 'dynamic_friction_range',
               'restitution_range')


def prepare_shared_output_dir(path: Path) -> None:
    """Allow the host gate to create approval/rejection markers in Docker output.

    The directory belongs to this fresh, single-purpose attempt. Its files keep
    their own original ownership and the host never rewrites sensor evidence.
    """
    path.mkdir(parents=True, exist_ok=False)
    path.chmod(0o777)


def assess_material_gate(terrain_rows: list[dict], collision_rows: list[dict],
                         link_rows: list[list[dict]], tensor, startup: dict,
                         frozen_startup: dict, declared_terrain_sha: str) -> dict:
    """Evaluate composed terrain and complete effective robot material coverage.

    Startup ranges are supplied from the resolved collection configuration. The
    frozen copy prevents a later config change from silently redefining equality.
    """
    failures = []
    ranges = []
    for name in RANGE_NAMES:
        value = startup.get(name)
        if (not isinstance(value, (tuple, list)) or len(value) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) for v in value) or value[0] > value[1]):
            failures.append('resolved_startup_range_invalid:' + name)
        elif list(value) != frozen_startup.get(name):
            failures.append('resolved_startup_range_drift:' + name)
        else:
            ranges.append(tuple(value))
    for name in ('num_buckets', 'make_consistent'):
        if startup.get(name) != frozen_startup.get(name):
            failures.append('resolved_startup_event_drift:' + name)
    if (not terrain_rows or any(row.get('binding_status') != 'bound' or
            row.get('attribute_error') is not None or
            row.get('composed_canonical_sha256') != declared_terrain_sha
            for row in terrain_rows)):
        failures.append('terrain_composed_material_mismatch')
    if not isinstance(collision_rows, list) or any(
            row.get('binding_status') not in ('bound', 'unbound')
            for row in collision_rows):
        failures.append('robot_usd_binding_unresolved')
    effective = None
    if len(ranges) == 3 and isinstance(collision_rows, list):
        try:
            effective = map_effective_materials(link_rows, collision_rows, tensor, ranges)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            failures.append('effective_PhysX_material_invalid:' + str(exc))
        else:
            effective['canonical_sha256'] = hashlib.sha256(json.dumps(
                effective['canonical_fingerprint_by_env'], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if not effective['coverage_pass']:
                failures.append('robot_collision_shape_coverage_failed')
            if not effective['startup_ranges_pass']:
                failures.append('robot_effective_material_outside_resolved_event')
            fingerprints = effective['canonical_fingerprint_by_env']
            if len(fingerprints) != 4 or len(set(fingerprints.values())) != 1:
                failures.append('robot_material_fingerprint_varies_by_env')
    else:
        failures.append('effective_PhysX_material_not_assessable')
    return {'schema': MATERIAL_SCHEMA, 'pass': not failures, 'failures': failures,
            'terrain_collision_bindings': terrain_rows,
            'robot_collision_bindings': collision_rows,
            'terrain_declared_sha256': declared_terrain_sha,
            'resolved_startup_randomization': startup,
            'frozen_startup_randomization': frozen_startup,
            'robot_effective_physx': effective}
