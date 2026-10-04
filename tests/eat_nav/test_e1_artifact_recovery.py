import json
from pathlib import Path
import shutil

import pytest

from eat_nav_e1_monitor import verify_complete
from eat_nav_e1_artifact_recovery import verify_recovered_registry,seal_tree,verify_seal


ATTEMPT3=Path('/tmp/eatnav-e1-attempt3-20261004')
SOURCE=ATTEMPT3/'runs/g0e-train-000-nominal_duplicate-00'


def test_original_verifier_accepts_read_only_recovery_path_mapping(tmp_path):
    needed=['started.json','docker-exec-return.json','final-status.json',
        'run-complete.json','bridge-shutdown.json','live-0.marker.json',
        'live-0.trace.json.gz','live-0.clock.json.gz','live-0.json','anchor-gate.json']
    for name in needed:shutil.copy2(SOURCE/name,tmp_path/name)
    (tmp_path/'follower-logs').mkdir()
    shutil.copy2(SOURCE/'follower-logs/bridge-events.jsonl',
                 tmp_path/'follower-logs/bridge-events.jsonl')
    with pytest.raises(ValueError,match='marker identity'):
        verify_complete(tmp_path)
    assert verify_complete(tmp_path,source_run=SOURCE)


def test_recovered_registry_requires_saved_verifier_sha_and_valid_json(tmp_path):
    source=ATTEMPT3/'recovered-attempt3-00-registry-original.json'
    verified=json.loads((SOURCE/'finalization-verifier-result.json').read_text())
    good=tmp_path/'registry.json';shutil.copy2(source,good)
    result=verify_recovered_registry(good,verified['process_registry_sha256'],
        'g0e-train-000-nominal_duplicate-00')
    assert result['status']=='pass' and result['role_count']>=15
    good.write_text(good.read_text()+' ')
    with pytest.raises(ValueError,match='SHA'):
        verify_recovered_registry(good,verified['process_registry_sha256'],
            'g0e-train-000-nominal_duplicate-00')
    good.write_text('{bad')
    with pytest.raises(ValueError):
        verify_recovered_registry(good,verified['process_registry_sha256'],
            'g0e-train-000-nominal_duplicate-00')


def test_seal_detects_new_file_and_registry_append(tmp_path):
    registry=tmp_path/'process-registry.json';registry.write_text('{}\n')
    original=seal_tree(tmp_path)
    assert verify_seal(tmp_path)==original
    with registry.open('a') as stream:stream.write('{"resource":1}\n')
    with pytest.raises(ValueError,match='sealed tree changed'):verify_seal(tmp_path)
    registry.write_text('{}\n')
    (tmp_path/'unexpected.log').write_text('late write\n')
    with pytest.raises(ValueError,match='sealed tree changed'):verify_seal(tmp_path)


def test_current_run_output_rejects_previous_or_external_path(tmp_path):
    from eat_nav_e1_artifact_recovery import assert_current_run_path
    current=tmp_path/'runs/current/process-registry.json'
    assert assert_current_run_path(tmp_path,'current',current)==current
    with pytest.raises(ValueError,match='outside current run'):
        assert_current_run_path(tmp_path,'current',tmp_path/'runs/previous/process-registry.json')
    with pytest.raises(ValueError,match='invalid current run'):
        assert_current_run_path(tmp_path,'../previous',current)
    (tmp_path/'runs').mkdir()
    (tmp_path/'runs/alias').symlink_to(tmp_path/'runs/current')
    with pytest.raises(ValueError,match='symlink'):
        assert_current_run_path(tmp_path,'alias',tmp_path/'runs/alias/process-registry.json')


def test_recovery_set_is_separate_and_source_unchanged(tmp_path):
    from eat_nav_e1_artifact_recovery import recover_first_run,sha
    saved=ATTEMPT3/'recovered-attempt3-00-registry-original.json'
    corrupted=SOURCE/'process-registry.json';before=sha(corrupted)
    identity=json.loads((SOURCE/'run-complete.json').read_text())['identity']
    recovery=tmp_path/'artifact-recovery/attempt3-direct-repeat-00'
    result=recover_first_run(SOURCE,saved,recovery,identity)
    assert result['status']=='pass' and sha(corrupted)==before
    assert sha(recovery/'process-registry.json')==result['process_registry_sha256']
    assert verify_seal(recovery)==result['sealed_tree_sha256']


def test_recovery_rejects_wrong_saved_registry_without_touching_source(tmp_path):
    from eat_nav_e1_artifact_recovery import recover_first_run,sha
    wrong=tmp_path/'wrong.json';wrong.write_text('{}\n')
    corrupted=SOURCE/'process-registry.json';before=sha(corrupted)
    identity=json.loads((SOURCE/'run-complete.json').read_text())['identity']
    with pytest.raises(ValueError,match='SHA'):
        recover_first_run(SOURCE,wrong,tmp_path/'recovery',identity)
    assert sha(corrupted)==before and not (tmp_path/'recovery').exists()
