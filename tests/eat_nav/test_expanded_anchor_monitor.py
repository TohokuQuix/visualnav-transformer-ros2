from eat_nav_goal_subgoal_timing_monitor import memory_stop_reason,parse_anon_shmem


def test_file_cache_does_not_trip_frozen_anon_shmem_gate():
    stat='anon 2500000000\nfile 5500000000\nshmem 1000000\n'
    anon,shmem=parse_anon_shmem(stat)
    assert (anon,shmem)==(2500000000,1000000)
    assert memory_stop_reason(total_bytes=9683791872,anon_bytes=anon,
        shmem_bytes=shmem,anon_shmem_cap_bytes=6442450944,total_cap_bytes=8*1024**3) is None


def test_frozen_anon_shmem_cap_still_stops_actual_memory_excess():
    assert memory_stop_reason(total_bytes=10_000_000_000,anon_bytes=6_440_000_000,
        shmem_bytes=10_000_000,anon_shmem_cap_bytes=6442450944,
        total_cap_bytes=8*1024**3)=='cgroup anon+shmem exceeded frozen cap'
