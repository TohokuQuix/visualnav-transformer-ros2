"""Ownership and GPU accounting tests required before the one-shot probe."""
import json
import signal
import subprocess

from eat_nav_g0e_gpu_probe import (MIB, Nvml, account, atomic, deduplicate_gpu,
    descendants, proc_info, safe_signal)


UUID='GPU-test'


def process(pid,parent,start,container_pid=None):
    return {'host_pid':pid,'parent_host_pid':parent,'start_ticks':start,
            'container_pid':container_pid or pid,'namespace_pid_chain':[pid,container_pid or pid],
            'cgroup':'0::/system.slice/docker-abc.scope' if container_pid else '0::/user.slice/test'}


def gpu(pid,mib,kind='compute'):
    return {'host_pid':pid,'gpu_uuid':UUID,'gpu_index':0,
            'used_gpu_memory_bytes':mib*MIB,'kind':kind}


def test_simulator_descendant_and_external_classification():
    table={100:process(100,1,1000,10),101:process(101,100,1001,11),
           102:process(102,101,1002,12),200:process(200,1,2000)}
    assert descendants(table,{100})=={100,101,102}
    result=account([{'used_bytes':1000*MIB}], [gpu(102,600),gpu(200,300)],table,{100},set(),'abc')
    assert result['simulator_tree_used_bytes']==600*MIB
    assert result['other_processes_used_bytes']==300*MIB
    assert result['unattributed_bytes']==100*MIB
    assert result['processes'][0]['proc']['container_pid']==12
    assert result['processes'][0]['container_id']=='abc'


def test_preexisting_and_start_time_reuse():
    table={100:process(100,1,2000,10),200:process(200,1,3000)}
    result=account([{'used_bytes':100*MIB}],[gpu(100,80)],table,{100},{(100,1000),(200,3000)},'abc')
    assert result['processes'][0]['created_after_monitor'] is True
    assert result['processes'][0]['existed_before_monitor'] is False


def test_compute_graphics_overlap_is_not_double_counted():
    table={100:process(100,1,1000,10)}
    rows=deduplicate_gpu([gpu(100,80,'compute'),gpu(100,80,'graphics')],table)
    assert len(rows)==1 and rows[0]['used_gpu_memory_bytes']==80*MIB
    assert rows[0]['kinds']==['compute','graphics']


def test_exited_process_remains_unresolved_not_silently_external():
    result=account([{'used_bytes':100*MIB}],[gpu(100,80)],{},set(),set(),'abc')
    assert result['unresolved_pid_bytes']==80*MIB
    assert result['unattributed_bytes']==20*MIB


def test_atomic_failure_marker_and_partial_log_flush(tmp_path):
    marker=tmp_path/'failure.marker.json'
    partial=tmp_path/'partial.jsonl'
    partial.write_text('{"sample":1}\n')
    atomic(marker,{'status':'failed','partial_log':str(partial)})
    assert json.loads(marker.read_text())['status']=='failed'
    assert not list(tmp_path.glob('*.tmp'))
    assert partial.read_text().endswith('\n')


def test_cleanup_signals_only_exact_owned_pid_start(monkeypatch):
    sent=[]
    monkeypatch.setattr('eat_nav_g0e_gpu_probe.proc_info',lambda pid:
        {'start_ticks':100} if pid==10 else {'start_ticks':201} if pid==20 else None)
    monkeypatch.setattr('eat_nav_g0e_gpu_probe.os.kill',lambda pid,sig:sent.append((pid,sig)))
    result=safe_signal({(10,100),(20,200),(30,300)},signal.SIGTERM)
    assert sent==[(10,signal.SIGTERM)]
    assert len(result['skipped'])==2


def test_container_cleanup_never_signals_other_container_or_reused_pid(monkeypatch):
    import eat_nav_g0e_gpu_run as run
    calls=[]
    monkeypatch.setattr(run,'proc_info',lambda pid: {
        10:{'start_ticks':100,'cgroup':'docker-abc.scope','container_pid':5},
        20:{'start_ticks':201,'cgroup':'docker-abc.scope','container_pid':6},
        30:{'start_ticks':300,'cgroup':'docker-external.scope','container_pid':7}}.get(pid))
    monkeypatch.setattr(run,'docker',lambda *args,**kwargs:
        calls.append(args) or type('Result',(),{'returncode':0,'stderr':''})())
    result=run.signal_owned_container({(10,100),(20,200),(30,300)},'abc',signal.SIGTERM)
    assert calls==[('exec','isaac-lab-onix','kill','-TERM','5')]
    assert len(result['skipped'])==2


def test_host_container_pid_mapping_from_proc():
    own=proc_info(__import__('os').getpid())
    assert own['namespace_pid_chain'][0]==own['host_pid']
    assert own['container_pid']==own['namespace_pid_chain'][-1]
    assert own['start_ticks']>0


def test_nvml_v2_used_matches_independent_nvidia_smi():
    device,_=Nvml().sample()
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(device)==len(output)
    for row,used in zip(device,output):
        assert abs(row['used_bytes']/MIB-int(used.strip()))<32
        assert row['total_bytes']==row['used_bytes']+row['free_bytes']+row['driver_reserved_bytes']
