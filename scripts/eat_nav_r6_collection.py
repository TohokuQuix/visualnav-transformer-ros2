"""CPU-only contracts used by the R6 Isaac collector.

Output layout: group-<env>/initial-input.npz (legacy leading batch dimension),
anchor-state.npz, group.json, candidate-00..13/trace.jsonl and complete.json.
Terrain/manifest JSON and actual-physics-properties.npz/physics-audit.json live at
the batch root. Fixed --terrain-cols plus --tile-indices retain parent geometry
when reducing a batch; startup domain randomization is separately recorded and
is not guaranteed identical across runtime batch sizes.
A group is the indivisible split unit. Contact onset is deliberately unavailable
when only net body forces exist: ordinary ground support is not obstacle contact.
"""
from __future__ import annotations
import numpy as np

STEP_NS = 50_000_000
SCENES = ('field', 'wall', 'narrow', 'one_sided', 'diagonal', 'approach')


def validate_groups(groups, num_envs):
    if len(groups) != num_envs or len({g['group_id'] for g in groups}) != num_envs:
        raise ValueError('one unique manifest group required per environment')
    for i, group in enumerate(groups):
        if group['role'] not in ('train', 'validation', 'test', 'benchmark'):
            raise ValueError('unknown split role')
        if group.get('env_index', i) != i:
            raise ValueError('group env_index ordering differs')
    return groups


def verify_anchor(reference, current, atol=1e-5):
    errors = {}
    for key, expected in reference.items():
        value = np.asarray(current.get(key, [])); expected = np.asarray(expected)
        if value.shape != expected.shape:
            errors[key] = 'shape'
        elif not np.allclose(expected, value, atol=atol, rtol=0, equal_nan=False):
            errors[key] = float(np.max(np.abs(expected - value)))
    return {'passed': not errors, 'absolute_tolerance': atol, 'errors': errors}


def step_records(pre, post, commands, done, terminals, active, step, candidate, start_ns):
    rows = []
    for i in range(len(pre)):
        terminal = terminals.get(i)
        physical_post = terminal['root_state'] if terminal else np.asarray(post[i]).tolist()
        rows.append(dict(env_index=i, control_step=step, candidate=candidate,
                         logical_sim_time_ns=int(start_ns + step * STEP_NS),
                         post_timestamp_ns=int(start_ns + (step + 1) * STEP_NS),
                         root_state=np.asarray(pre[i]).tolist(), post_root_state=physical_post,
                         follower_command=np.asarray(commands[i]).tolist(), valid_mask=bool(active[i]),
                         terminal=None if terminal is None else {'pre_reset_state': terminal},
                         reset_mask=bool(done[i]), stop_reason='natural_termination' if done[i] else None,
                         contact_observation={'available': False, 'onset': None,
                           'reason': 'net body forces do not identify fixture contact'}))
    return rows


def fixture_boxes(scene):
    if scene not in SCENES:
        raise ValueError('unknown supplemental scene')
    wall = dict(size_m=[.20, 3., .8], center_m=[1.1, 0., .4], yaw_rad=0.)
    if scene == 'field': return []
    if scene == 'wall': return [wall]
    if scene == 'approach': return [dict(wall, center_m=[1.7, 0., .4])]
    if scene == 'diagonal': return [dict(wall, yaw_rad=float(np.pi / 4))]
    if scene == 'one_sided': return [dict(size_m=[1., .5, .3], center_m=[1., .45, .15], yaw_rad=0.)]
    return [dict(size_m=[2., .20, .8], center_m=[1., y, .4], yaw_rad=0.) for y in (-.65, .65)]


def fixture_proximity(body_positions, origin, scene):
    points = np.asarray(body_positions) - np.asarray(origin)
    distances = []
    for box in fixture_boxes(scene):
        yaw = box['yaw_rad']; c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray([[c,-s,0],[s,c,0],[0,0,1]])
        local = (points - box['center_m']) @ rotation
        q = np.abs(local) - np.asarray(box['size_m']) / 2
        distances.extend((np.linalg.norm(np.maximum(q,0),axis=1) + np.minimum(np.max(q,axis=1),0)).tolist())
    return dict(minimum_body_origin_distance_m=min(distances) if distances else None,
                true_contact_available=False,
                definition='body-frame origins to fixture oriented boxes; not occupied-body clearance')


def selected_tiles(num_envs, terrain_cols, indices=None):
    tiles = list(range(num_envs)) if indices is None else [int(x) for x in indices.split(',')]
    if len(tiles) != num_envs or len(set(tiles)) != num_envs or any(t < 0 or t >= terrain_cols for t in tiles):
        raise ValueError('unique parent terrain tile column required per runtime environment')
    return tiles


def fixture_contact_index(point_world, origin, scene, tolerance_m=.025):
    """Classify an actual PhysX contact point, never a geometric proximity alone.

    Ground/box-bottom points below 3cm are deliberately ambiguous/excluded.
    A tolerance permits PhysX contact offsets at an analytic fixture surface.
    """
    point=np.asarray(point_world)-np.asarray(origin)
    if point[2] <= .03:return None
    for i,box in enumerate(fixture_boxes(scene)):
        c,s=np.cos(box['yaw_rad']),np.sin(box['yaw_rad'])
        rotation=np.asarray([[c,-s,0],[s,c,0],[0,0,1]])
        local=(point-box['center_m'])@rotation
        q=np.abs(local)-np.asarray(box['size_m'])/2
        signed=np.linalg.norm(np.maximum(q,0))+min(float(np.max(q)),0.)
        if abs(signed)<=tolerance_m:return i
    return None


def audit_physics_properties(masses,inertias):
    masses=np.asarray(masses);inertias=np.asarray(inertias).reshape(*masses.shape,3,3)
    finite=bool(np.isfinite(masses).all() and np.isfinite(inertias).all())
    eigenvalues=np.linalg.eigvalsh(inertias) if finite else np.full((*masses.shape,3),np.nan)
    return dict(valid=finite and bool((masses>0).all() and (eigenvalues>0).all()),
                all_finite=finite,minimum_mass_kg=float(np.nanmin(masses)),
                nonpositive_mass_count=int((masses<=0).sum()),
                minimum_inertia_eigenvalue=float(np.nanmin(eigenvalues)) if finite else None,
                nonpositive_inertia_eigenvalue_count=int((eigenvalues<=0).sum()))


def update_anchor_valid(previous, terminated):
    """Only the environment which reset loses its proposed anchor."""
    previous=np.asarray(previous,dtype=bool);terminated=np.asarray(terminated,dtype=bool)
    if previous.ndim!=1 or previous.shape!=terminated.shape:raise ValueError('per-env validity/reset shape mismatch')
    return previous & ~terminated
