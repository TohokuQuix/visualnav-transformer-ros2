from scripts.eat_nav_seven_state_diagnose import classify_command_delivery, summarize_follower_log


def test_near_goal_without_message_is_not_normal_stop():
    result = classify_command_delivery(
        goal_distance_m=0.10, tolerance_m=0.20,
        before=[0, 0], after=[0, 0], commands=[None, None], goal_status=[False, False],
    )
    assert result['delivery'] == 'no_message'
    assert result['classification'] == 'pipeline_unresolved'


def test_fresh_zero_with_goal_status_is_normal_stop():
    result = classify_command_delivery(
        goal_distance_m=0.10, tolerance_m=0.20,
        before=[0, 0], after=[1, 1], commands=[[0, 0, 0], [0, 0, 0]],
        goal_status=[True, True],
    )
    assert result['delivery'] == 'fresh_zero'
    assert result['classification'] == 'no_action_required'


def test_far_goal_with_no_message_is_infrastructure_candidate():
    result = classify_command_delivery(
        goal_distance_m=0.80, tolerance_m=0.20,
        before=[0, 0], after=[0, 0], commands=[None, None], goal_status=[False, False],
    )
    assert result['delivery'] == 'no_message'
    assert result['classification'] == 'infrastructure_diagnosis_required'


def test_partial_freshness_is_not_full_command_delivery():
    result = classify_command_delivery(
        goal_distance_m=0.80, tolerance_m=0.20,
        before=[0, 0], after=[1, 0], commands=[[0.2, 0, 0], None],
        goal_status=[False, False],
    )
    assert result['delivery'] == 'partial_fresh'
    assert result['classification'] == 'infrastructure_diagnosis_required'


def test_log_summary_separates_path_tf_and_control_evidence():
    lines = '''[INFO] [1.0] [follower]: Started: path=/path
[INFO] [1.2] [follower]: Activated default path with 9 waypoints.
[WARN] [1.3] [follower]: Failed to lookup robot pose from body to map
[INFO] [2.4] [follower]: control: cmd=(vx=0.2, wz=0.1)
'''
    report = summarize_follower_log(lines)
    assert report['path_callback_count_lower_bound'] == 1
    assert report['tf_failure_count'] == 1
    assert report['control_callback_count_lower_bound'] == 2
    assert report['first_path_callback_s'] == 1.2
    assert report['first_control_debug_s'] == 2.4
    assert report['command_publish_count'] is None


def test_map_path_goal_geometry_uses_published_endpoint():
    from scripts.eat_nav_seven_state_diagnose import map_path_goal_geometry
    out = map_path_goal_geometry([[1.0, 2.0], [1.0, 2.8]], [0.8, 0.0], 0.2)
    assert abs(out['published_goal_distance_m'] - 0.8) < 1e-12
    assert out['within_goal_tolerance'] is False
    assert out['relative_goal_distance_m'] == 0.8
