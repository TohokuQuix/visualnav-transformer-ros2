"""Frozen P1 v2 GPU ownership semantics, independent of a live device."""
import copy

import pytest

from eat_nav_p1_gpu_attribution import classify_bracket, map_container_pid


MIB=1024*1024
UUID='GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26'


def case():
    device={'uuid':UUID,'total_bytes':16303*MIB,'used_bytes':6000*MIB,'temperature_c':50}
    rows=[{'gpu_uuid':UUID,'host_pid':100,'kind':'compute','used_gpu_memory_bytes':5500*MIB},
          {'gpu_uuid':UUID,'host_pid':20,'kind':'graphics','used_gpu_memory_bytes':400*MIB}]
    table={100:{'host_pid':100,'start_ticks':111,'parent_host_pid':90,'container_pid':7,
                'cgroup':'/docker/container-id','command_line':'/isaac-sim/python.sh'},
           101:{'host_pid':101,'start_ticks':112,'parent_host_pid':100,'container_pid':8,
                'cgroup':'/docker/container-id','command_line':'child'},
           20:{'host_pid':20,'start_ticks':22,'parent_host_pid':1,'container_pid':20,
               'cgroup':'/user.slice','command_line':'/usr/lib/xorg/Xorg'}}
    owned={'simulator':{'host_pid':100,'start_ticks':111},'docker_exec':{'host_pid':90,'start_ticks':99}}
    baseline={(20,22)}
    contract={'required_gpu_uuid':UUID,'required_total_memory_mib_min':16000,
              'gpu_device_total_hard_cap_mib':7680,'bracket_nonatomic_sampling_gap_hard_cap_mib':256,
              'absolute_attribution_gap_hard_cap_mib':256,'unresolved_gpu_bytes_hard_cap':0}
    return device,rows,table,owned,baseline,contract


def evaluate(*, alter=None, accounting_mode='Disabled'):
    d,r,t,o,b,c=copy.deepcopy(case())
    after=copy.deepcopy(d)
    after_table=copy.deepcopy(t)
    if alter:alter(d,r,t,after,after_table,o,b,c)
    return classify_bracket(d,r,t,after,after_table,o,b,c,container_id='container-id',
                            expected_simulator_alive=True,health_ok=True,accounting_mode=accounting_mode)


def test_owned_simulator_and_child_exact_start_ticks():
    def child(d,r,t,a,at,o,b,c):
        r.append({'gpu_uuid':UUID,'host_pid':101,'kind':'compute','used_gpu_memory_bytes':25*MIB})
        d['used_bytes']+=25*MIB;a['used_bytes']+=25*MIB
    result=evaluate(alter=child)
    assert result['pass']
    assert result['owned_gpu_pids']==[100,101]
    assert result['unresolved_gpu_bytes']==0


def test_pid_reuse_start_tick_mismatch():
    def reused(d,r,t,a,at,o,b,c):t[100]['start_ticks']=999;at[100]['start_ticks']=999
    assert 'owned_start_tick_mismatch' in evaluate(alter=reused)['failures']


def test_known_graphics_in_device_total_not_unresolved_or_gap():
    result=evaluate(accounting_mode='Disabled')
    assert result['pass']
    assert result['known_graphics_bytes']==400*MIB
    assert result['unresolved_gpu_bytes']==0
    assert result['bracket_gap_bytes']==100*MIB


def test_unresolved_process_fails():
    def unknown(d,r,t,a,at,o,b,c):del t[20];del at[20]
    assert 'unresolved_gpu_process' in evaluate(alter=unknown)['failures']


def test_missing_process_table_fails():
    def missing(d,r,t,a,at,o,b,c):t.clear()
    assert 'missing_process_table' in evaluate(alter=missing)['failures']


def test_zero_gpu_process_rows_with_nontrivial_device_use_fails():
    def empty(d,r,t,a,at,o,b,c):r.clear()
    assert 'empty_gpu_process_table' in evaluate(alter=empty)['failures']


def test_bracket_gap_overflow_fails():
    def gap(d,r,t,a,at,o,b,c):d['used_bytes']+=200*MIB;a['used_bytes']+=200*MIB
    assert 'bracket_gap_cap' in evaluate(alter=gap)['failures']


def test_device_uuid_mismatch_fails():
    def uuid(d,r,t,a,at,o,b,c):a['uuid']='GPU-other'
    assert 'gpu_uuid_mismatch' in evaluate(alter=uuid)['failures']


def test_simulator_disappears_within_bracket_fails():
    def gone(d,r,t,a,at,o,b,c):del at[100]
    assert 'simulator_disappeared_in_bracket' in evaluate(alter=gone)['failures']


def test_device_total_over_cap_fails():
    def over(d,r,t,a,at,o,b,c):a['used_bytes']=7681*MIB
    assert 'device_total_cap' in evaluate(alter=over)['failures']


def test_unexpected_compute_process_fails():
    def compute(d,r,t,a,at,o,b,c):
        t[30]={'host_pid':30,'start_ticks':33,'parent_host_pid':1,'container_pid':30,
               'cgroup':'/user.slice','command_line':'python unrelated'}
        at[30]=copy.deepcopy(t[30]);r.append({'gpu_uuid':UUID,'host_pid':30,'kind':'compute','used_gpu_memory_bytes':40*MIB})
        d['used_bytes']+=40*MIB;a['used_bytes']+=40*MIB
    assert 'unexpected_compute_process' in evaluate(alter=compute)['failures']


def test_container_pid_mapping_requires_container_and_start_tick():
    _,_,table,_,_,_=case()
    assert map_container_pid(table,7,'container-id') == {'host_pid':100,'start_ticks':111}
    with pytest.raises(ValueError,match='container_pid_mapping'):
        map_container_pid(table,7,'different-container')
