"""The student GUI launcher must not depend on the author's home directory."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('portable_gui', ROOT/'scripts/eat_nav_phase3_gui_launch.py')
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


def test_router_discovery_and_identity_without_pid_file(tmp_path):
    proc = tmp_path/'proc'
    router = proc/'42'
    router.mkdir(parents=True)
    (router/'cmdline').write_bytes(b'/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd\0')
    (router/'stat').write_text('42 (rmw_zenohd) S ' + '0 '*18 + '12345 ' + '0 '*20)
    identity = gui.router_identity(None, proc_root=proc)
    assert identity['pid'] == 42
    assert identity['start_tick'] == 12345
    assert identity['stdout_sha256'] is None


def test_router_discovery_rejects_ambiguity(tmp_path):
    proc = tmp_path/'proc'
    for pid in (42, 43):
        router = proc/str(pid)
        router.mkdir(parents=True)
        (router/'cmdline').write_bytes(b'/opt/ros/rmw_zenohd\0')
        (router/'stat').write_text(f'{pid} (rmw_zenohd) S ' + '0 '*18 + '12345 ')
    with pytest.raises(ValueError, match='multiple'):
        gui.router_identity(None, proc_root=proc)


def test_onix_mount_is_derived_from_container(tmp_path):
    source = tmp_path/'onix'
    source.mkdir()
    mounts = [{'Type': 'bind', 'Source': str(source), 'Destination': '/workspace/isaaclab_onix'}]
    assert gui.resolve_onix_mount(mounts) == (source, Path('/workspace/isaaclab_onix'))
    with pytest.raises(ValueError, match='bind mount'):
        gui.resolve_onix_mount([])


def test_worker_python_is_explicit_or_discovered(tmp_path):
    worker = tmp_path/'miniconda3/envs/tinyvla/bin/python'
    worker.parent.mkdir(parents=True)
    worker.touch()
    assert gui.resolve_worker_python(None, home=tmp_path) == worker
    with pytest.raises(FileNotFoundError, match='worker-python'):
        gui.resolve_worker_python(None, home=tmp_path/'other')


def test_no_author_specific_runtime_path():
    source = (ROOT/'scripts/eat_nav_phase3_gui_launch.py').read_text()
    assert '/home/kojima' not in source
    assert '/home/kojima' not in (ROOT/'scripts/eat_nav_phase3_gui_isaac.py').read_text()
