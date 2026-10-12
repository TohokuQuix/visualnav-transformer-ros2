import json
import math
from pathlib import Path

import hashlib

import pytest

from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, SOURCE, compile_core


def test_independent_core_compiles_and_first_golden_command(tmp_path):
    library = compile_core(tmp_path)
    root = Path('/home/kojima/eatnav-backups/2026-10-09-p1-radial-28-analysis')
    run = root / 'runs/p1-00-first-differential-b-20-r00/run'
    summary = json.loads((run / 'candidate-summary.json').read_text())
    trace = json.loads((run / 'candidate-trace.jsonl').open().readline())
    contract = json.loads((root / 'provenance/continuation/radial-candidate-contract.json').read_text())
    base = next(item['waypoints'] for item in contract['paths'] if item['path_sha256'] == summary['candidate_path_sha256'])
    root_state = trace['root_state']
    w, x, y, z = root_state[3:7]
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    bx, by = base[0]
    mx, my = summary['map_path'][0]
    px = mx - math.cos(yaw)*bx + math.sin(yaw)*by
    py = my - math.sin(yaw)*bx - math.cos(yaw)*by
    follower = PureFollower(summary['map_path'], library=library)
    actual = follower.step(px, py, yaw, trace['logical_sim_time_ns'])
    assert max(abs(a-b) for a,b in zip(actual, trace['follower_command'])) < 1e-6


def test_all_3360_golden_commands_and_direct_clone_state_isolation(tmp_path):
    library = compile_core(tmp_path)
    root = Path('/home/kojima/eatnav-backups/2026-10-09-p1-radial-28-analysis')
    rows = json.loads((root / 'integrated-manifest.json').read_text())['rows']
    contract = json.loads((root / 'provenance/continuation/radial-candidate-contract.json').read_text())
    paths = {item['path_sha256']: item['waypoints'] for item in contract['paths']}
    worst = 0.0
    total = 0
    for row in rows:
        run = root / 'runs' / row['physical_run_id'] / 'run'
        summary = json.loads((run / 'candidate-summary.json').read_text())
        traces = [json.loads(line) for line in (run / 'candidate-trace.jsonl').read_text().splitlines()]
        def yaw(state):
            w, x, y, z = state[3:7]
            return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        first = traces[0]['root_state']
        angle = yaw(first)
        bx, by = paths[row['candidate_path_sha256']][0]
        mx, my = summary['map_path'][0]
        px = mx - math.cos(angle)*bx + math.sin(angle)*by
        py = my - math.sin(angle)*bx - math.cos(angle)*by
        origin = first[0]-px, first[1]-py
        follower = PureFollower(summary['map_path'], library=library)
        for trace in traces:
            state = trace['root_state']
            actual = follower.step(state[0]-origin[0], state[1]-origin[1],
                                   yaw(state), trace['logical_sim_time_ns'])
            error = max(abs(a-b) for a, b in zip(actual, trace['follower_command']))
            worst = max(worst, error)
            assert error < 1e-6, (row['physical_run_id'], trace['control_step'], error)
            total += 1
        follower.close()
    assert total == 3360
    assert worst < 1e-6


def test_core_is_ros_free_and_accepts_episode_local_long_time(tmp_path):
    source = SOURCE.read_text()
    assert '#include <rclcpp' not in source
    assert '#include <geometry_msgs' not in source
    production = Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src/simple_trajectory_follower.cpp')
    assert hashlib.sha256(production.read_bytes()).hexdigest() == '75249a834a90feb25fc2967e2e253eeaeffa498491c68692434bd08099a8d2be'
    library = compile_core(tmp_path)
    path = [[0.1, 0.0], [0.8, 0.0]]
    left = PureFollower(path, library=library)
    right = PureFollower(path, library=library)
    assert left.step(0.0, 0.0, 0.0, 2_000_000_000) == right.step(0.0, 0.0, 0.0, 2_000_000_000)
    assert left.step(0.8, 0.0, 0.0, 2_050_000_000) == (0.0, 0.0, 0.0)
    assert right.step(0.0, 0.0, 0.0, 2_050_000_000)[0] > 0
    assert right.step(0.0, 0.0, 0.0, 9_000_000_001)[0] > 0


def test_explicit_follower_state_can_be_cloned_per_env(tmp_path):
    library = compile_core(tmp_path)
    a = PureFollower([[0.1, 0.0], [0.8, 0.0]], library=library)
    b = PureFollower([[0.1, 0.0], [0.8, 0.0]], library=library)
    a.step(0.0, 0.0, 0.0, 2_000_000_000)
    snapshot = a.state()
    b.restore(snapshot)
    assert b.state() == snapshot
    assert a.step(0.1, 0.0, 0.0, 2_050_000_000) == b.step(0.1, 0.0, 0.0, 2_050_000_000)
    a.step(0.8, 0.0, 0.0, 2_100_000_000)
    assert a.state() != b.state()


def test_long_horizon_progress_stall_and_goal_hold(tmp_path):
    follower = PureFollower([[0, 0], [10, 0]], library=compile_core(tmp_path))
    events = []
    for step in range(500):
        command, diagnostic = follower.step_diagnostic(0, 0, 0, 12_000_000_000 + step * 50_000_000)
        assert all(math.isfinite(value) for value in command)
        if diagnostic['linear_recovery_started']:
            events.append((step, diagnostic['recovery_cause']))
    assert events[0] == (400, 'progress_stall')
    assert follower.step_diagnostic(10, 0, 0, 37_000_000_000)[0] == (0, 0, 0)
    assert follower.step_diagnostic(10, 0, 0, 37_050_000_000)[1]['goal_reached']


def test_time_and_recovery_boundaries(tmp_path):
    follower = PureFollower([[0, 0], [10, 0]], library=compile_core(tmp_path))
    follower.step_diagnostic(0, 0, 0, 2_000_000_000)
    with pytest.raises(RuntimeError):
        follower.step_diagnostic(0, 0, 0, 2_000_000_000)
    for step in range(1, 400):
        assert not follower.step_diagnostic(0, 0, 0, 2_000_000_000 + step * 50_000_000)[1]['linear_recovery_started']
    assert follower.step_diagnostic(0, 0, 0, 22_000_000_000)[1]['linear_recovery_started']


def test_linear_stall_backtrack_completion_and_full_state_clone(tmp_path):
    library = compile_core(tmp_path)
    follower = PureFollower([[0, 0], [10, 0]], library=library)
    for step in range(403):
        x = min(step, 200) * .005
        command, diagnostic = follower.step_diagnostic(x, 0, 0, 2_000_000_000 + step * 50_000_000)
    assert diagnostic['recovery_cause'] == 'linear_stall'
    assert diagnostic['linear_recovery']
    assert diagnostic['backtrack_points'] >= 2
    assert command[0] < 0
    clone = follower.clone()
    for step in range(403, 463):
        x = 1 - (step - 402) * .005
        stamp = 2_000_000_000 + step * 50_000_000
        assert follower.step_diagnostic(x, 0, 0, stamp) == clone.step_diagnostic(x, 0, 0, stamp)
    assert diagnostic['linear_recovery']  # capture at trigger
    assert follower.step_diagnostic(.695, 0, 0, 25_150_000_000)[1]['linear_recovery'] is False


def test_angular_boost_starts_at_motion_window_boundary(tmp_path):
    follower = PureFollower([[0, 0], [10, 0]], library=compile_core(tmp_path))
    started = []
    for step in range(410):
        command, diagnostic = follower.step_diagnostic(
            step * .002, 0, .7, 2_000_000_000 + step * 50_000_000)
        if diagnostic['angular_recovery_started']:
            started.append(step)
            assert command[0] == pytest.approx(.1)
    assert started == [402]


def test_unreached_640_step_episode_is_finite_and_never_claims_goal(tmp_path):
    follower = PureFollower([[0, 0], [100, 0]], library=compile_core(tmp_path))
    for step in range(640):
        command, diagnostic = follower.step_diagnostic(
            step * .01, 0, 0, 12_000_000_000 + step * 50_000_000)
        assert not diagnostic['goal_reached']
        assert all(math.isfinite(x) for x in command)
    assert diagnostic['episode_elapsed_sec'] == pytest.approx(31.95)


def test_offline_replay_g2_exact_stamp_trace(tmp_path):
    from scripts.eat_nav_p1_g3_rosfree_offline_parity import CONFIG, replay_output

    output = Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1/g2-attempt-0/output')
    result = replay_output(output, compile_core(tmp_path), json.loads(CONFIG.read_text()))
    assert result['rows'] == 2400
    assert result['pass']
    assert result['recovery_ros_steps'] == 0
