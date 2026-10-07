import json


def marker(path, value):
    path.write_text(json.dumps(value))


def test_diagnostic_invalid_index_is_not_promoted_to_full_smoke(tmp_path):
    from visualnav_transformer.eat_nav.contact_host_gate import evaluate_run_markers
    marker(tmp_path/'failure.json',{'error':'index'})
    marker(tmp_path/'contact-index-invalid.json',{'first_invalid':{'count':300},'raw_sha256':'abc'})
    assert evaluate_run_markers(tmp_path,'diagnostic',0)=='diagnostic_index_invalid'
    assert evaluate_run_markers(tmp_path,'full',0)=='invalid'


def test_complete_five_pose_requires_200_steps_and_no_failure(tmp_path):
    from visualnav_transformer.eat_nav.contact_host_gate import evaluate_run_markers
    marker(tmp_path/'sim-finished.json',{'physics_steps':199,'pose_count':5})
    assert evaluate_run_markers(tmp_path,'full',0)=='invalid'


def test_full_shard_marker_requires_its_exact_pose_count(tmp_path):
    from visualnav_transformer.eat_nav.contact_host_gate import evaluate_run_markers
    marker(tmp_path/'sim-finished.json',{'physics_steps':200,'pose_count':2})
    assert evaluate_run_markers(tmp_path,'full',0,expected_pose_count=2)=='full_complete'
    assert evaluate_run_markers(tmp_path,'full',0,expected_pose_count=5)=='invalid'
    marker(tmp_path/'sim-finished.json',{'physics_steps':200,'pose_count':5})
    assert evaluate_run_markers(tmp_path,'full',0)=='full_complete'
    marker(tmp_path/'failure.json',{'error':'late'})
    assert evaluate_run_markers(tmp_path,'full',0)=='invalid'
