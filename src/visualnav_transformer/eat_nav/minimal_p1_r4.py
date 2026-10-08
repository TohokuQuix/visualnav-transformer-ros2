"""Versioned P1 r4 GPU baseline gate; known process memory is not uncertainty."""

from __future__ import annotations

import math


def health_query_command() -> list[str]:
    return ['nvidia-smi', '-q']


def assess_nvidia_smi_health(report: str, returncode: int) -> bool:
    """Return whether a full `nvidia-smi -q` report shows a health failure."""
    if returncode != 0 or not report.strip():
        return True
    for line in report.splitlines():
        if 'GPU Recovery Action' in line:
            action = line.partition(':')[2].strip().lower()
            if action not in ('none', 'not supported', 'n/a'):
                return True
    return False


def evaluate_baseline(sample: dict, processes: list[dict], contract: dict) -> dict:
    failures = []
    if sample['uuid'] != contract['required_gpu_uuid']:
        failures.append('gpu_uuid_changed')
    if sample['name'] != contract['required_gpu_name']:
        failures.append('gpu_name_changed')
    if sample['total_mib'] < contract['required_total_memory_mib_min'] or abs(
            sample['total_mib'] - contract['audited_total_memory_mib']) > 1:
        failures.append('gpu_memory_profile_changed')
    if sample['mig_mode'] not in ('[N/A]', 'Disabled'):
        failures.append('gpu_mig_profile_changed')
    if not sample.get('driver_version') or not sample.get('nvml_version'):
        failures.append('gpu_driver_or_nvml_identity_missing')
    if sample['device_used_mib'] > contract['baseline_device_total_hard_cap_mib']:
        failures.append('baseline_device_over_cap')
    if sample['device_used_mib'] > contract['gpu_device_total_hard_cap_mib']:
        failures.append('gpu_device_over_cap')
    if sample['unresolved_pid_bytes'] != contract['unresolved_gpu_bytes_hard_cap']:
        failures.append('unresolved_gpu_pid_memory')
    if abs(sample['bracket_gap_mib']) > contract['absolute_attribution_gap_hard_cap_mib']:
        failures.append('gpu_attribution_gap')
    if not math.isfinite(float(sample['temperature_c'])):
        failures.append('gpu_temperature_nonfinite')
    if sample.get('gpu_health_error'):
        failures.append('gpu_health_error')
    identities = []
    for process in processes:
        proc = process.get('proc') or {}
        pid = process.get('host_pid')
        tick = proc.get('start_ticks')
        command = proc.get('command_line', '')
        identities.append({'host_pid': pid, 'start_ticks': tick,
                           'used_gpu_memory_bytes': process.get('used_gpu_memory_bytes'),
                           'kinds': process.get('kinds', []), 'command_line': command})
        if not isinstance(pid, int) or not isinstance(tick, int) or tick <= 0:
            failures.append('unknown_other_process_identity')
        if ('compute' in process.get('kinds', []) or any(
                word in command.lower() for word in ('isaacsim', 'isaac-lab', 'train.py', 'play.py'))):
            failures.append('unexpected_compute_workload')
    return {'pass': not failures, 'failures': sorted(set(failures)),
            'known_other_process_mib': sample['other_processes_used_mib'],
            'known_other_processes': identities,
            'device_total_mib': sample['device_used_mib'],
            'contract_schema': contract['schema']}
