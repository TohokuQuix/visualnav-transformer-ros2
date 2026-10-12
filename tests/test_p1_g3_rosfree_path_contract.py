"""Offline tests for the only authorized persistent ROS-free output path."""

from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_g3_rosfree_path_contract import (
    validate_simulator_paths, validate_host_bind, mount_points_from_proc,
)


PROCESS = 'batch-first-differential-0-period-0'


def fixture(tmp_path):
    host_bind = tmp_path/'host-bind'
    container_bind = tmp_path/'container-bind'
    host_bind.mkdir()
    container_bind.mkdir()
    attempt = '.eatnav-p1-g3-rosfree-direct-core-v2-attempt-1'
    root = container_bind/attempt
    batch = root/PROCESS
    (batch/'input').mkdir(parents=True)
    (batch/'source.py').write_text('frozen source')
    host_root = host_bind/attempt
    (host_root/PROCESS/'input').mkdir(parents=True)
    manifest = {
        'schema': 'eatnav_p1_g3_rosfree_direct_core_v2',
        'runtime_authorized': True,
        'automatic_continuation': False,
        'authorized_process_ids': [PROCESS],
        'attempt_id': attempt,
        'attempt_host_root': str(host_root),
        'attempt_container_root': str(root),
        'host_bind_source': str(host_bind),
        'container_bind_destination': str(container_bind),
    }
    return manifest, host_root, root, batch


def check(manifest, batch, *, output=None, input_dir=None, mount_present=True):
    return validate_simulator_paths(
        manifest, input_dir or batch/'input', output or batch/'output',
        batch/'source.py', {str(batch.parents[1])} if mount_present else set(),
        expected_bind=(manifest['host_bind_source'], manifest['container_bind_destination']))


def test_correct_new_attempt_root_and_host_bind(tmp_path):
    manifest, host_root, root, batch = fixture(tmp_path)
    result = check(manifest, batch)
    assert result['attempt_id'] == manifest['attempt_id']
    inspect = {'Mounts': [{'Type': 'bind', 'Source': manifest['host_bind_source'],
                           'Destination': manifest['container_bind_destination'],
                           'RW': True}]}
    assert validate_host_bind(inspect, manifest, host_root/PROCESS,
                              root/PROCESS, expected_bind=(
                                  manifest['host_bind_source'],
                                  manifest['container_bind_destination']))['bind_mapping_verified'] is True


def test_old_g3_root_rejected(tmp_path):
    manifest, _, root, batch = fixture(tmp_path)
    old = root.parent/'.eatnav-p1-long-horizon-v1'/PROCESS
    (old/'input').mkdir(parents=True)
    (old/'source.py').write_text('old')
    with pytest.raises(ValueError):
        check(manifest, old)


def test_manifest_mismatched_root_rejected(tmp_path):
    manifest, _, _, batch = fixture(tmp_path)
    manifest['attempt_container_root'] += '-other'
    with pytest.raises(ValueError):
        check(manifest, batch)


def test_traversal_rejected_even_if_normalized_destination_matches(tmp_path):
    manifest, _, _, batch = fixture(tmp_path)
    path = Path(str(batch/'extra'/'..'/'output'))
    with pytest.raises(ValueError, match='traversal'):
        check(manifest, batch, output=path)


def test_symlink_escape_rejected(tmp_path):
    manifest, _, _, batch = fixture(tmp_path)
    outside = tmp_path/'sealed-outside'
    outside.mkdir()
    (outside/'host-seal.json').write_text('{}')
    (batch/'output').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        check(manifest, batch)


def test_missing_bind_mount_rejected(tmp_path):
    manifest, _, _, batch = fixture(tmp_path)
    with pytest.raises(ValueError, match='bind'):
        check(manifest, batch, mount_present=False)


def test_existing_sealed_evidence_root_rejected(tmp_path):
    manifest, _, _, batch = fixture(tmp_path)
    (batch/'host-seal.json').write_text('{}')
    with pytest.raises(ValueError, match='sealed'):
        check(manifest, batch)


def test_runtime_authorization_and_attempt_identity_required(tmp_path):
    manifest, _, _, batch = fixture(tmp_path)
    manifest['runtime_authorized'] = False
    with pytest.raises(ValueError, match='authorization'):
        check(manifest, batch)
    manifest['runtime_authorized'] = True
    manifest['attempt_id'] += '-wrong'
    with pytest.raises(ValueError, match='identity'):
        check(manifest, batch)


def test_host_rejects_wrong_or_missing_bind_mapping(tmp_path):
    manifest, host_root, root, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='bind'):
        validate_host_bind({'Mounts': []}, manifest, host_root/PROCESS,
                           root/PROCESS, expected_bind=(manifest['host_bind_source'],
                                                       manifest['container_bind_destination']))
    with pytest.raises(ValueError, match='bind'):
        validate_host_bind({'Mounts': [{'Type': 'bind', 'Source': '/wrong',
                          'Destination': manifest['container_bind_destination'], 'RW': True}]},
                           manifest, host_root/PROCESS, root/PROCESS,
                           expected_bind=(manifest['host_bind_source'],
                                          manifest['container_bind_destination']))


def test_proc_mountinfo_parses_exact_bind_destination():
    text = ('5468 5401 259:2 /home/kojima/lib/onix_isaac_lab '
            '/workspace/isaaclab_onix rw,relatime - ext4 /dev/nvme0n1p2 rw\n')
    assert mount_points_from_proc(text) == {'/workspace/isaaclab_onix'}
