"""No router process or network is used in these lifecycle tests."""
from pathlib import Path
import json

from eat_nav_p1_g3_router_observer_v4 import sample_router, run_observer
from eat_nav_p1_g3_runtime_evidence_v4 import check_router_series


def _proc(root: Path, pid: int, ppid: int, tick: int, command: str):
    path = root / str(pid)
    path.mkdir(parents=True)
    # The suffix starts with state; starttime is suffix index 19.
    fields = ['S', str(ppid), '500', '500', '0'] + ['0'] * 14 + [str(tick)]
    (path / 'stat').write_text(f'{pid} (process name) ' + ' '.join(fields) + '\n')
    (path / 'cmdline').write_bytes(command.encode() + b'\0')
    (path / 'cgroup').write_text('0::/user.slice/router.scope\n')
    (path / 'fd').mkdir()


def fixture(tmp_path):
    proc = tmp_path / 'proc'; proc.mkdir()
    _proc(proc, 10, 1, 100, 'ros2 run rmw_zenoh_cpp rmw_zenohd')
    _proc(proc, 11, 10, 101, '/opt/ros/bin/rmw_zenohd')
    router = tmp_path / 'router'; router.mkdir()
    (router / 'router-launch.pid').write_text('10\n')
    (router / 'router.pid').write_text('11\n')
    (router / 'router.stdout.log').write_text('Using ZID: abc123\nZenoh can be reached at: tcp/127.0.0.1:7447\nStarted Zenoh router with id abc123\n')
    (router / 'router.stderr.log').write_text('')
    return proc, router


def test_observer_captures_parent_session_cgroup_cmdline_and_logs(tmp_path):
    proc, router = fixture(tmp_path)
    row = sample_router(router, proc_root=proc, now_ns=lambda: 1000,
                        listener_probe=lambda pid, port: True,
                        endpoint_probe=lambda endpoint: True)
    assert row['pid'] == 11 and row['start_tick'] == 101
    assert row['launcher_pid'] == 10 and row['launcher_start_tick'] == 100
    assert row['router_parent_pid'] == 10 and row['router_session_id'] == 500
    assert row['router_cgroup'] == '0::/user.slice/router.scope'
    assert row['router_cmdline'] == '/opt/ros/bin/rmw_zenohd'
    assert row['zid'] == 'abc123' and row['listener_owned']
    assert row['stdout_sha256'] and row['stderr_sha256']
    assert row['misses'] == []


def test_deleted_log_path_uses_launcher_fd_without_mutating_router(tmp_path):
    proc, router = fixture(tmp_path)
    output = (router / 'router.stdout.log').read_bytes()
    (proc / '10/fd/1').write_bytes(output)
    (proc / '10/fd/2').write_bytes(b'')
    (router / 'router.stdout.log').unlink()
    (router / 'router.stderr.log').unlink()
    row = sample_router(router, proc_root=proc, now_ns=lambda: 1000,
                        listener_probe=lambda pid, port: True,
                        endpoint_probe=lambda endpoint: True)
    assert row['stdout_path_deleted'] and row['stderr_path_deleted']
    assert row['stdout_available'] and row['stderr_available']
    assert row['stdout_bytes'] == output


def test_deleted_log_does_not_block_on_pipe_fd(tmp_path):
    import os
    proc, router = fixture(tmp_path)
    (router/'router.stdout.log').unlink()
    os.mkfifo(proc/'10/fd/1')
    row = sample_router(router, proc_root=proc, now_ns=lambda: 1000,
                        listener_probe=lambda pid, port: True,
                        endpoint_probe=lambda endpoint: True)
    assert not row['stdout_available']
    assert 'log_unavailable' in row['misses']


def test_observer_writes_fsynced_series_and_preserved_log_copy(tmp_path):
    proc, router = fixture(tmp_path)
    output = tmp_path / 'evidence'; stop = tmp_path / 'stop'
    ticks = iter([0, 1_000_000, 500_000_000, 501_000_000, 1_000_000_000, 1_001_000_000])
    def clock():
        return next(ticks)
    def sleep(_):
        if (output / 'router-series.jsonl').exists() and len((output / 'router-series.jsonl').read_text().splitlines()) >= 2:
            stop.write_text('stop')
    run_observer(router, output, stop, proc_root=proc, now_ns=clock, sleeper=sleep,
                 listener_probe=lambda pid, port: True,
                 endpoint_probe=lambda endpoint: True)
    rows = [json.loads(line) for line in (output / 'router-series.jsonl').read_text().splitlines()]
    assert len(rows) >= 2 and (output / 'router-stdout-copy.bin').read_bytes() == (router / 'router.stdout.log').read_bytes()
    assert (output / 'observer-complete.json').is_file()
    assert all(row['observer_heartbeat_seq'] == i for i, row in enumerate(rows))
    check_router_series(rows, run_start_ns=rows[0]['sample_finished_wall_ns'],
                        run_end_ns=rows[-1]['sample_started_wall_ns'])
