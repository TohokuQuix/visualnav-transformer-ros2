"""The expanded probe, X2 and continuation preserve the ROS subprocess environment."""
import os
from pathlib import Path
import subprocess

import pytest


REPO = Path('/tmp/eatnav-vnt-expanded-feasibility')


def test_old_x2_environment_replaces_ros_pythonpath():
    base = {'PYTHONPATH':'/opt/ros/jazzy/lib/python3.12/site-packages'}
    old = {**base, 'PYTHONPATH':f'{REPO}/src:{REPO}/scripts'}
    assert '/opt/ros/jazzy/lib/python3.12/site-packages' not in old['PYTHONPATH'].split(':')


def test_builder_preserves_ros_paths_and_other_variables():
    from eat_nav_ros_subprocess_env import build_env
    base={'PYTHONPATH':'/opt/ros/jazzy/lib/python3.12/site-packages:/tmp/old',
          'RMW_IMPLEMENTATION':'rmw_zenoh_cpp','ROS_DOMAIN_ID':'7',
          'AMENT_PREFIX_PATH':'/opt/ros/jazzy','LD_LIBRARY_PATH':'/opt/ros/jazzy/lib',
          'ZENOH_CONFIG_OVERRIDE':'test'}
    built=build_env(base, repo=REPO)
    paths=built['PYTHONPATH'].split(':')
    assert paths[:2]==[str(REPO/'src'),str(REPO/'scripts')]
    assert paths[2:]==base['PYTHONPATH'].split(':')
    assert all(built[key]==value for key,value in base.items() if key!='PYTHONPATH')
    assert build_env(built,repo=REPO)==built


def test_builder_handles_missing_path_and_duplicates():
    from eat_nav_ros_subprocess_env import build_env
    assert build_env({'OTHER':'keep'},repo=REPO)['PYTHONPATH']==f'{REPO}/src:{REPO}/scripts'
    base={'PYTHONPATH':f'{REPO}/scripts:/opt/ros/jazzy/lib/python3.12/site-packages:{REPO}/src'}
    assert build_env(base,repo=REPO)['PYTHONPATH']==f'{REPO}/src:{REPO}/scripts:/opt/ros/jazzy/lib/python3.12/site-packages'


def test_probe_x2_and_continuation_share_builder():
    from eat_nav_ros_subprocess_env import build_env
    from eat_nav_expanded_ros_x2 import x2_command
    base={'PYTHONPATH':'/opt/ros/jazzy/lib/python3.12/site-packages'}
    assert build_env(base,repo=REPO)['PYTHONPATH'].startswith(f'{REPO}/src:{REPO}/scripts:')
    assert x2_command(Path('/tmp/test-x2'))[0]=='/usr/bin/python3'
    source=(REPO/'scripts/eat_nav_e1_run_one.py').read_text()
    probe=(REPO/'scripts/eat_nav_expanded_gpu_startup_probe.py').read_text()
    x2=(REPO/'scripts/eat_nav_expanded_ros_x2.py').read_text()
    expanded=(REPO/'scripts/eat_nav_expanded_feasibility_run.py').read_text()
    assert 'build_ros_host_env' in source
    assert 'build_container_env' in expanded
    assert 'build_container_env' in probe
    assert 'build_ros_host_env' in x2
    assert 'PYTHONPATH=' not in source and 'PYTHONPATH=' not in probe and 'PYTHONPATH=' not in x2


def test_continuation_manifest_starts_at_frozen_row_16():
    from eat_nav_expanded_ros_continuation import row_split
    rows=[{'order':i,'process_id':'expanded-train-001-left_2' if i==15 else f'row-{i}'}
          for i in range(50)]
    reused,pending=row_split(rows)
    assert len(reused)==15 and len(pending)==35
    assert pending[0]['order']==15
    rows[15]['process_id']='changed'
    with pytest.raises(ValueError,match='row 16'):
        row_split(rows)


def test_reused_storage_is_counted_without_copying_raw_run(tmp_path):
    import json
    from eat_nav_expanded_feasibility_run import measured_storage_bytes
    from eat_nav_e1_monitor import measured_disk_bytes
    old=tmp_path/'old';run=old/'runs/row-0';run.mkdir(parents=True)
    (run/'trace.gz').write_bytes(b'raw'*100)
    new=tmp_path/'new';(new/'runs').mkdir(parents=True)
    (new/'runs/row-0').symlink_to(run,target_is_directory=True)
    (new/'continuation-reuse-ledger.json').write_text(json.dumps({'reused_run_bytes':300}))
    assert measured_storage_bytes(new)>=300
    assert measured_disk_bytes(new)>=300
    assert (new/'runs/row-0').is_symlink()
