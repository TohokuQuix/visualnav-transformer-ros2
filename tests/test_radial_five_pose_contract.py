import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from visualnav_transformer.eat_nav.radial_contact_audit import decode_contact_buffer, orientation_metrics
from visualnav_transformer.eat_nav.contact_preflight_protocol import normalize_view_filter_paths


def test_five_fixed_poses_are_unique_and_match_preserved_schedule():
    root=Path('/tmp/eatnav-radial-five-pose-smoke-20261007')
    if not (root/'pose-manifest.json').exists():
        return
    manifest=json.loads((root/'pose-manifest.json').read_text())['poses']
    schedule=json.loads(Path('/tmp/eatnav-radial-terrain-inventory-20261007/pose-schedule.json').read_text())
    assert len(manifest)==len({p['pose_id'] for p in manifest})==5
    by_id={p['pose_id']:p for p in schedule}
    for p in manifest:
        assert p==by_id[p['pose_id']]
        assert p['batch_seed']==20261007


def test_contact_decoder_only_uses_count_start_and_fails_on_invalid():
    data=dict(sensor_paths=['/World/envs/env_0/onix_chassis/chassis_body'],
              filter_paths=['/World/ground/terrain'], forces=[3.0,9999.0],
              points=[[0,0,0],[float('nan'),0,0]], normals=[[0,0,1],[0,0,1]],
              separations=[-.002,float('nan')], counts=[[1]], starts=[[0]], capacity=2)
    out=decode_contact_buffer(**data)
    assert out['contact_count']==1
    assert out['pairs'][0]['max_penetration_m']==.002
    assert out['pairs'][0]['pair_kind']=='robot_terrain'
    assert not out['buffer_overflow_or_full']
    data['counts']=[[2]]
    import pytest
    with pytest.raises(ValueError):
        decode_contact_buffer(**data)


def test_filter_layout_and_h1():
    expected=['/World/ground/terrain']
    assert normalize_view_filter_paths([expected,expected],expected,2)==expected
    import pytest
    with pytest.raises(ValueError):
        normalize_view_filter_paths([['/World/ground/wrong']],expected,1)
    assert abs(orientation_metrics([1,0,0,0],0)['h1_signed_rad'])<1e-10


def test_terrain_collider_resolves_mesh_child_not_xform_parent():
    from visualnav_transformer.eat_nav.radial_five_pose_contract import terrain_mesh_path
    assert terrain_mesh_path('/World/ground/terrain')=='/World/ground/terrain/mesh'
