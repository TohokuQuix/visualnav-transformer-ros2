"""G1 fail-closed GPU monitor using the already validated P1 NVML bracket."""

from __future__ import annotations

from eat_nav_p1_gpu_attribution import capture_live, classify_bracket


def resolve_simulator(table, *, source_path: str, container_id: str) -> dict:
    """Bind one live source command to its exact host PID and /proc start tick."""
    matches = [row for row in table.values()
               if source_path in row.get('command_line', '').split()
               and container_id in row.get('cgroup', '')]
    if len(matches) != 1 or not isinstance(matches[0].get('start_ticks'), int):
        raise ValueError('G1 simulator PID/start-tick unresolved or ambiguous')
    row = matches[0]
    return {'simulator': {'host_pid': row['host_pid'], 'start_ticks': row['start_ticks']}}


def assess_sample(before, rows, table_before, after, table_after, owned, baseline,
                  contract, *, container_id, require_owned_gpu=True):
    result = classify_bracket(before, rows, table_before, after, table_after,
                              owned, baseline, contract, container_id=container_id,
                              expected_simulator_alive=True, health_ok=True,
                              accounting_mode='g1_nvml_per_pid_required')
    failures = set(result['failures'])
    if require_owned_gpu and not result['owned_gpu_pids']:
        failures.add('owned_gpu_pid_missing')
    if any(row['start_ticks'] is None or row['classification'] == 'unresolved'
           for row in result['processes']):
        failures.add('gpu_pid_identity_unresolved')
    for row in result['processes']:
        after_pid = table_after.get(row['host_pid']) if table_after else None
        if after_pid is None or after_pid.get('start_ticks') != row['start_ticks']:
            failures.add('gpu_pid_disappeared_or_reused')
    if result['bracket_gap_bytes'] > 256 * 1024 * 1024:
        failures.add('g1_attribution_gap_over_256_mib')
    result['failures'] = sorted(failures)
    result['pass'] = not failures
    return result


def capture_sample(nvml, owned, baseline, contract, *, container_id, source_path=None,
                   require_owned_gpu=True):
    # NVML and /proc exceptions propagate: callers must stop, never assume zero use.
    result = capture_live(nvml, owned, baseline, contract,
                          container_id=container_id, expected_simulator_alive=True,
                          owner_resolver=(lambda table: resolve_simulator(
                              table, source_path=source_path, container_id=container_id))
                          if source_path else None)
    failures = set(result['failures'])
    if require_owned_gpu and not result['owned_gpu_pids']:
        failures.add('owned_gpu_pid_missing')
    if any(row['start_ticks'] is None or row['classification'] == 'unresolved'
           for row in result['processes']):
        failures.add('gpu_pid_identity_unresolved')
    table_after = result['process_table_after']
    for row in result['processes']:
        after_pid = table_after.get(row['host_pid']) if table_after else None
        if after_pid is None or after_pid.get('start_ticks') != row['start_ticks']:
            failures.add('gpu_pid_disappeared_or_reused')
    if result['bracket_gap_bytes'] > 256 * 1024 * 1024:
        failures.add('g1_attribution_gap_over_256_mib')
    result['failures'] = sorted(failures)
    result['pass'] = not failures
    return result
