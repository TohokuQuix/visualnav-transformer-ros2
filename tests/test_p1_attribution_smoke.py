import pytest

from eat_nav_p1_attribution_smoke import choose_shard, find_launcher, validate_invocation
from eat_nav_p1_gpu_attribution import capture_live


def test_choose_highest_recorded_gpu_shard_without_candidate_outcome():
    entries=[{'source_shard':'A','fixture_subroot':'/old/shard-00'},
             {'source_shard':'B','fixture_subroot':'/old/shard-01'}]
    peaks={'/old/shard-00':5906,'/old/shard-01':5898}
    assert choose_shard(entries,peaks)['source_shard']=='A'


def test_launcher_must_match_container_and_exact_output():
    table={101:{'host_pid':101,'start_ticks':27,'cgroup':'/docker/abc',
                'command_line':'/isaac-sim/python.sh /old/sim.py --output /new/run --mode full'},
           102:{'host_pid':102,'start_ticks':28,'cgroup':'/docker/other',
                'command_line':'/isaac-sim/python.sh /old/sim.py --output /new/run --mode full'}}
    assert find_launcher(table,'abc','/new/run')=={'host_pid':101,'start_ticks':27}
    with pytest.raises(ValueError,match='launcher'):
        find_launcher(table,'abc','/wrong/run')


def test_invocation_rejects_path_follower_bridge_or_other_output():
    cmd=['docker','exec','container','/isaac-sim/python.sh','/old/sim.py','--output','/new/run',
         '--mode','full']
    validate_invocation(cmd,'/new/run')
    for addition in ['--candidate-path','--follower','--bridge']:
        with pytest.raises(ValueError,match='candidate_free'):
            validate_invocation(cmd+[addition],'/new/run')
    with pytest.raises(ValueError,match='output'):
        validate_invocation(cmd,'/other/run')


def test_startup_owner_resolved_from_same_nvml_bracket(monkeypatch):
    import eat_nav_p1_gpu_attribution as gpu
    mib=1024*1024;uuid='GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26'
    table={101:{'host_pid':101,'start_ticks':27,'parent_host_pid':1,'container_pid':7,
                'cgroup':'/docker/abc','command_line':'python /old/sim.py --output /new/run --mode full'}}
    class FakeNvml:
        last_query_timing={};last_memory_snapshot_timing={}
        def sample(self):
            return ([{'uuid':uuid,'total_bytes':16303*mib,'used_bytes':5500*mib,'temperature_c':50}],
                    [{'gpu_uuid':uuid,'host_pid':101,'kind':'compute','used_gpu_memory_bytes':5400*mib}])
        def memory_snapshot(self):return [{'uuid':uuid,'used_bytes':5500*mib}]
    class Health:
        returncode=0;stdout='ok';stderr=''
    monkeypatch.setattr(gpu,'proc_table',lambda:table)
    monkeypatch.setattr(gpu.subprocess,'run',lambda *a,**k:Health())
    contract={'required_gpu_uuid':uuid,'required_total_memory_mib_min':16000,
              'gpu_device_total_hard_cap_mib':7680,'bracket_nonatomic_sampling_gap_hard_cap_mib':256,
              'absolute_attribution_gap_hard_cap_mib':256,'unresolved_gpu_bytes_hard_cap':0}
    result=capture_live(FakeNvml(),{},set(),contract,container_id='abc',
                        expected_simulator_alive=True,
                        owner_resolver=lambda rows:{'simulator':find_launcher(rows,'abc','/new/run')})
    assert result['pass']
    assert result['resolved_owned']['simulator']=={'host_pid':101,'start_ticks':27}
    assert result['process_table_before_wall_ns'] <= result['process_table_after_wall_ns']
