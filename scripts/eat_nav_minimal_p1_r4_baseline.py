#!/usr/bin/env python3
"""Fresh, candidate-free P1 r4 GPU baseline with corrected attribution semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.eat_nav_g0e_gpu_probe import Nvml
from scripts.eat_nav_minimal_p1_r3_gpu import capture
from visualnav_transformer.eat_nav.minimal_p1_r4 import (
    assess_nvidia_smi_health, evaluate_baseline, health_query_command,
)


def save_once(path: Path, data: dict) -> str:
    payload = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(root: Path, stage: str) -> dict:
    root = Path(root)
    if stage not in ('pre-x1', 'post-x2'):
        raise ValueError('invalid baseline stage')
    contract_path = root / 'p1-resource-contract-v2.json'
    contract = json.loads(contract_path.read_text())
    if contract['schema'] != 'p1_local_primitive_resource_contract_v2':
        raise ValueError('unfrozen resource contract')
    raw, summary = capture(Nvml())
    health = subprocess.run(health_query_command(), capture_output=True,
                            text=True, timeout=10)
    raw['nvidia_smi_health_query'] = {'command': health_query_command(),
                                      'returncode': health.returncode,
                                      'stdout': health.stdout,
                                      'stderr': health.stderr}
    summary['gpu_health_error'] = assess_nvidia_smi_health(
        health.stdout, health.returncode) or bool(health.stderr.strip())
    result = evaluate_baseline(summary, raw['deduplicated_processes'], contract)
    result.update({'stage': stage, 'sample_summary': summary,
                   'sample_wall_ns': raw['wall_ns'],
                   'r4_contract_sha256': hashlib.sha256(contract_path.read_bytes()).hexdigest()})
    result['raw_sha256'] = save_once(root / f'gpu-baseline-{stage}-raw.json', raw)
    save_once(root / f'gpu-baseline-{stage}.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('pre-x1', 'post-x2'), required=True)
    args = parser.parse_args()
    result = run(args.root, args.stage)
    print(json.dumps({'pass': result['pass'], 'failures': result['failures'],
                      'sample_summary': result['sample_summary']}, indent=2))
