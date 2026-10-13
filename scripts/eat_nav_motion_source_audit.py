"""Read-only audit of the sealed ROS-free 12-trial source for vNext labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from visualnav_transformer.eat_nav.evaluator.dataset import load_sealed_period, verify_host_seal
from visualnav_transformer.eat_nav.motion_vnext import SCHEMA, motion_label_from_trace


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(roots: list[Path]) -> dict:
    if len(roots) != 3 or len(set(map(str, roots))) != 3:
        raise ValueError('three independent period roots required')
    records = []
    for period, root in enumerate(roots):
        samples = load_sealed_period(root, period, 'eatnav_p1_g3_rosfree_crossover_v1')
        batch = root / f'batch-first-differential-0-period-{period}'
        seal_sha, stage = verify_host_seal(batch)
        output = batch / 'output'
        for env, sample in enumerate(samples):
            trace_path = output / f'trace-slot-{env}.jsonl'
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            result = motion_label_from_trace(trace)
            records.append({'period': period, 'env_id': env, 'sample_id': sample['sample_id'],
                            'state_group': sample['state_group'], 'terrain_seed': sample['seed'],
                            'path_role': sample['path_role'], 'trace_sha256': sha(trace_path),
                            'host_seal_sha256': seal_sha, 'host_seal_stage': stage,
                            'label': result})
    return {'schema': 'eatnav_motion_source_audit_v1', 'label_schema': SCHEMA,
            'trial_count': len(records), 'available_40_step_labels': sum(
                item['label']['available'] for item in records),
            'independent_state_group_count': len({item['state_group'] for item in records}),
            'held_out_claim_permitted': False, 'records': records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
