import math

from visualnav_transformer.eat_nav.radial_inventory import (
    TERRAIN_ORDER, build_pose_schedule, pose_world, validate_record,
)


def test_schedule_has_two_seeds_seven_terrains_sixteen_angles_without_outcome():
    rows = build_pose_schedule((20261007, 20261008))
    assert len(rows) == 224
    assert len({r['pose_id'] for r in rows}) == 224
    assert set(r['terrain'] for r in rows) == set(TERRAIN_ORDER)
    for seed in (20261007, 20261008):
        for col, terrain in enumerate(TERRAIN_ORDER):
            group = [r for r in rows if r['batch_seed'] == seed and r['terrain'] == terrain]
            assert len(group) == 16
            assert {r['angle_index'] for r in group} == set(range(16))
            assert {r['column'] for r in group} == {col}
            assert {r['source_family'] for r in group} == {f'{seed}:{terrain}'}
            assert all(r['row'] == r['angle_index'] for r in group)
            assert all(r['terrain_class'] == ('source_flat_radial' if terrain in TERRAIN_ORDER[:4] else 'unguaranteed_center_radial') for r in group)
    assert all('outcome' not in r and 'path' not in r for r in rows)


def test_pose_world_uses_runtime_origin_and_preserves_yaw():
    p = pose_world((24.0, -12.0, 1.5), 2)
    assert math.isclose(p['world_x'], 24.0 + .3 / math.sqrt(2))
    assert math.isclose(p['world_y'], -12.0 + .3 / math.sqrt(2))
    assert math.isclose(p['world_yaw'], math.pi / 4)
    assert p['world_z_reference'] == 1.5


def test_record_validation_rejects_mask_missing_and_origin_or_actor_mismatch():
    row = build_pose_schedule((20261007,))[0]
    record = {
        'pose_id': row['pose_id'], 'batch_seed': row['batch_seed'], 'terrain': row['terrain'],
        'row': row['row'], 'column': row['column'], 'source_family': row['source_family'],
        'env_origin': [0, 0, 0], 'requested_pose': pose_world((0, 0, 0), 0),
        'settled_error_xy_m': 0.01, 'settled_yaw_error_rad': .01,
        'settled_speed_mps': .01, 'settled_angular_speed_radps': .01,
        'penetration': False, 'unexpected_reset': False, 'termination': False,
        'dem_history_shape': [4, 2, 20, 20], 'dem_valid_fraction': .8,
        'dem_finite': True, 'timestamps_monotone': True, 'actor_finite': True,
        'cleanup_ok': True,
    }
    assert validate_record(record, row)['valid']
    assert not validate_record({**record, 'dem_valid_fraction': 0}, row)['valid']
    assert not validate_record({**record, 'actor_finite': False}, row)['valid']
    assert not validate_record({**record, 'column': 4}, row)['valid']


def test_candidate_free_simulator_imports_no_bridge_follower_or_ros():
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / 'scripts/eat_nav_radial_inventory_sim.py').read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    assert not any('bridge' in name or 'follower' in name or 'rclpy' in name or 'nav_msgs' in name for name in imports)
    assert 'publish' not in source
    assert '--base-path' not in source
    assert 'capture_actor_critic' not in source
    assert 'critic_value' not in source
