"""Freeze resource-aware thresholds before the approved timing replay."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

OUT = Path('/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001')
PREV = Path('/tmp/eatnav-candidate-coverage/goal-ood-timing-20261001')


def main():
    target = OUT / 'resource-thresholds.json'
    if target.exists():
        raise FileExistsError('resource thresholds already frozen')
    audit = json.loads((OUT / 'resource-probe-provenance.json').read_text())
    if audit['status'] != 'pass':
        raise ValueError('R0 audit must pass')
    prior = json.loads((PREV / 'runs/train-003-timing-anchor/final-status.json').read_text())
    known_good = int(prior['cpu_anon_peak_bytes'])
    anon_cap = max(6 * 1024**3, 2 * known_good)
    host_total = int(audit['host_meminfo_kib']['MemTotal:']) * 1024
    host_min = max(4 * 1024**3, math.ceil(.10 * host_total))
    config = {
        'status': 'frozen_before_resource_probe',
        'previous_hard_stop': {'metric': 'cgroup memory.current', 'cap_bytes': 8 * 1024**3,
                               'includes_reclaimable_file_cache': True},
        'memory_current_hard_stop': False,
        'known_good_anon_peak_bytes': known_good,
        'anon_shmem_cap_formula': 'max(6 GiB, 2 * known_good_anon_peak_bytes)',
        'anon_shmem_cap_bytes': anon_cap,
        'host_mem_total_bytes': host_total,
        'host_mem_available_formula': 'max(4 GiB, 0.10 * MemTotal)',
        'host_mem_available_min_bytes': host_min,
        'host_low_consecutive_heartbeats': 3,
        'memory_events_baseline': {key: audit['memory_events'][key]
                                   for key in ('low', 'high', 'max', 'oom', 'oom_kill')},
        'memory_max_at_baseline': audit['memory_max'],
        'file_cache_warn_bytes': 6 * 1024**3,
        'gpu_cap_mib': 6656,
        'step_stall_sec': 120,
        'heartbeat_stale_sec': 90,
        'heartbeat_interval_sec': 10,
        'storage_cap_bytes': 800_000_000,
        'run_timeout_sec': 3600,
        'r0_provenance_sha256': hashlib.sha256((OUT / 'resource-probe-provenance.json').read_bytes()).hexdigest(),
    }
    with target.open('x') as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write('\n')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (OUT / 'resource-thresholds.sha256').write_text(digest + '  resource-thresholds.json\n')
    print('R2_FROZEN', target, digest, 'anon_cap', anon_cap, 'host_min', host_min)


if __name__ == '__main__':
    main()
