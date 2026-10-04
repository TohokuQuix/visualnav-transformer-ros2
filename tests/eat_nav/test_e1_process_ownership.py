from pathlib import Path

import pytest

from eat_nav_e1_process_ownership import ProcessRegistry, stop_and_join_monitor, register_followers, container_owned_roots, registered_alive, ownership_spec
import json


def info(pid, tick, *, state='S', parent=1):
    return {'host_pid':pid,'start_ticks':tick,'parent_host_pid':parent,
            'container_pid':pid,'process_group':pid,'session_id':pid,
            'cgroup':'0::/test','state':state,'command_line':'/same/simulator.py'}


def registry():
    r=ProcessRegistry('run-a',100)
    for role,pid in [('runner',10),('runtime_monitor',11),('finalization_verifier',10),
                     ('zenoh_router',12),('docker_exec',20),('simulator',21),
                     ('bridge',22)]+[(f'follower_{i}',30+i) for i in range(8)]:
        owner='external' if role=='zenoh_router' else ('measurement' if role in
            {'docker_exec','simulator','bridge'} or role.startswith('follower_') else 'control')
        r.register(role,info(pid,pid*10),ownership=owner)
    return r


@pytest.mark.parametrize('role', ['runtime_monitor','runner','finalization_verifier','zenoh_router'])
def test_control_or_external_with_simulator_path_is_not_measurement_residual(role):
    r=registry();entry=r.by_role(role)[0]
    assert r.residuals(lambda pid: info(pid,entry['start_ticks']) if pid==entry['host_pid'] else None)==[]


@pytest.mark.parametrize('role', ['simulator','bridge','follower_3','docker_exec'])
def test_owned_measurement_survivor_is_residual(role):
    r=registry();entry=r.by_role(role)[0]
    found=r.residuals(lambda pid: info(pid,entry['start_ticks']) if pid==entry['host_pid'] else None)
    assert len(found)==1 and found[0]['role']==role


def test_pid_reuse_and_missing_proc_are_exited_not_residual():
    r=registry();entry=r.by_role('simulator')[0]
    assert r.residuals(lambda pid: info(pid,999) if pid==entry['host_pid'] else None)==[]
    assert r.residuals(lambda pid: None)==[]


def test_owned_zombie_is_residual():
    r=registry();entry=r.by_role('simulator')[0]
    found=r.residuals(lambda pid: info(pid,entry['start_ticks'],state='Z') if pid==entry['host_pid'] else None)
    assert found[0]['state']=='Z'


def test_unregistered_matching_command_is_not_residual():
    r=registry()
    assert r.residuals(lambda pid: None)==[]


def test_registry_does_not_accept_stale_run_or_start():
    r=registry()
    with pytest.raises(ValueError,match='run identity'):
        r.assert_run('run-b',100)
    with pytest.raises(ValueError,match='run identity'):
        r.assert_run('run-a',101)


def test_saved_registry_preserves_run_identity(tmp_path):
    r=registry();p=tmp_path/'registry.json';r.save(p)
    loaded=ProcessRegistry.load(p)
    loaded.assert_run('run-a',100)
    assert loaded.by_role('simulator')==r.by_role('simulator')


def test_follower_spawn_events_register_all_eight_without_command_matching(tmp_path):
    r=ProcessRegistry('run-a',100);p=tmp_path/'events.jsonl'
    p.write_text(''.join(json.dumps({'event':'follower_spawn','branch':i,
        'steady_ns':i+1,'process':info(30+i,300+i)})+'\n' for i in range(8)))
    register_followers(r,p)
    assert [r.by_role(f'follower_{i}')[0]['host_pid'] for i in range(8)]==list(range(30,38))


def test_container_processes_are_selected_by_cgroup_and_prelaunch_identity():
    old=info(40,400);new=info(41,410);other=info(42,420)
    old['cgroup']=new['cgroup']='0::/docker/abc';other['cgroup']='0::/docker/xyz'
    assert container_owned_roots({40:old,41:new,42:other},'abc',[[40,400]])=={41}


def test_liveness_uses_pid_and_start_tick_not_command_text():
    entry=registry().by_role('bridge')[0]
    assert registered_alive(entry,lambda pid:info(pid,entry['start_ticks']))
    assert not registered_alive(entry,lambda pid:info(pid,entry['start_ticks']+1))
    assert not registered_alive(entry,lambda pid:None)


def test_preflight_role_spec_separates_external_router_and_controls():
    spec=ownership_spec()
    assert spec['zenoh_router']['ownership']=='external'
    assert spec['runtime_monitor']['ownership']=='control'
    assert spec['finalization_verifier']['ownership']=='control'
    assert all(spec[f'follower_{i}']['ownership']=='measurement' for i in range(8))


def test_attempt_two_monitor_command_is_not_formal_residual(monkeypatch):
    import eat_nav_e1_run_one as runner
    source=runner.REMOTE_CODE
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:
        f'PID COMMAND\n2219087 /usr/bin/python3 eat_nav_e1_monitor.py --sim-source {source}\n')
    assert runner.active_processes()==[]


def test_forced_cleanup_targets_only_registered_live_simulator_identity(monkeypatch):
    import eat_nav_e1_run_one as runner
    r=registry();entry=r.by_role('simulator')[0]
    monkeypatch.setattr(runner,'process_record',lambda pid:
        info(pid,entry['start_ticks']) if pid==entry['host_pid'] else None)
    assert runner._docker_sim_pids(r)==[entry['container_pid']]
    monkeypatch.setattr(runner,'process_record',lambda pid:
        info(pid,999) if pid==entry['host_pid'] else None)
    assert runner._docker_sim_pids(r)==[]


def test_all_eight_followers_and_roles_required():
    r=registry();r.require_complete()
    r.entries=[entry for entry in r.entries if entry['role']!='follower_7']
    with pytest.raises(ValueError,match='follower_7'):
        r.require_complete()


class Monitor:
    def __init__(self,code=0):self.code=code;self.events=[];self.returncode=None
    def wait(self,timeout):
        self.events.append('join');self.returncode=self.code;return self.code


def test_monitor_is_joined_before_measurement_scan(tmp_path):
    monitor=Monitor();events=[]
    stop_and_join_monitor(monitor,tmp_path/'monitor.stop',lambda:events.append('sample'),timeout=30)
    events+=monitor.events;events.append('scan')
    assert events==['sample','join','scan']
    assert (tmp_path/'monitor.stop').is_file()


def test_monitor_join_timeout_is_control_cleanup_failure(tmp_path):
    class Hung(Monitor):
        def wait(self,timeout):raise TimeoutError('join')
    with pytest.raises(RuntimeError,match='control_cleanup_failure'):
        stop_and_join_monitor(Hung(),tmp_path/'monitor.stop',lambda:None,timeout=30)


def test_monitor_nonzero_exit_is_control_cleanup_failure(tmp_path):
    with pytest.raises(RuntimeError,match='control_cleanup_failure'):
        stop_and_join_monitor(Monitor(1),tmp_path/'monitor.stop',lambda:None,timeout=30)
