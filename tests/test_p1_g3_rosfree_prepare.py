"""Offline staging and process contract; never starts Isaac or ROS."""

import argparse
import json
from pathlib import Path

import pytest

import eat_nav_p1_g3_rosfree_host as host
import eat_nav_p1_g3_rosfree_prepare as prepare


REPO = Path(__file__).resolve().parents[1]


def test_direct_only_manifest_stage_and_sha(tmp_path):
    reference = prepare.REFERENCE
    source = json.loads((reference/'batch-first-differential-0-repeat-0/input/source-contract.json').read_text())
    assignment = prepare.build_assignment(source, 1000)
    assignment_file = tmp_path/'assignment.json'
    prepare.write_new(assignment_file, assignment)
    manifest = prepare.build_manifest(REPO, reference, prepare.sha(assignment_file), assignment)
    root = tmp_path/'stage'
    report = prepare.stage_one(REPO, reference, root, assignment, manifest)
    batch = root/prepare.PROCESS_ID
    assert report['manifest_sha256'] == prepare.sha(root/'manifest.json')
    assert manifest['runtime_authorized'] is False
    assert manifest['external_router_required'] is False
    assert [x['path_role'] for x in assignment['processes'][0]['trials']] == ['direct']*4
    ledger = json.loads((batch/'input-sha-ledger.json').read_text())
    assert all(prepare.sha(batch/name) == expected for name,expected in ledger.items())
    assert len([name for name in ledger if name.startswith('src/')]) >= 20
    with pytest.raises(ValueError, match='not authorized'):
        host.validate_staged(batch, argparse.Namespace(
            anchor_id='first-differential',batch_index=0,execution_index=0))


def test_rosfree_launch_has_no_transport_or_router_process():
    command = host.build_wrapper_command(
        remote='/persistent/evidence',container_id='test-container',token='token',
        source_sha='source',config_sha='config',anchor_id='first-differential',
        batch_index=0,execution_index=0)
    rendered = ' '.join(command).lower()
    for forbidden in ('zenoh','rclpy','bridge','follower_node','ros2 run','router'):
        assert forbidden not in rendered
    assert 'source.py' in rendered and 'owner-wrapper.py' in rendered


@pytest.mark.parametrize('source', [
    'scripts/eat_nav_p1_g3_rosfree_sim.py',
    'scripts/eat_nav_p1_g3_rosfree_host.py',
    'src/visualnav_transformer/eat_nav/p1_g3_rosfree_control.py',
])
def test_control_sources_have_no_ros_runtime_import(source):
    import ast
    tree = ast.parse((REPO/source).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    assert not any(name.startswith(('rclpy','tf2_ros','rmw_zenoh',
                                    'geometry_msgs','nav_msgs')) for name in imports)
