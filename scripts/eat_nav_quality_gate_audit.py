"""Read-only split audit of the extended EAT pilot; outputs JSON only."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.quality_gate_data import assert_split_integrity, audit_windows, raw_dem_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_eat_config(args.config)
    datasets = {}
    summaries = {}
    identities = {}
    split_map = None
    train_ecdf = None
    for split in ('train', 'val', 'test'):
        dataset = EATDataset(args.manifest, split, config, split_seed=123)
        if split_map is None:
            split_map = dataset.split_map
            train_ecdf = dataset.ecdf.values
        elif split_map != dataset.split_map or train_ecdf != dataset.ecdf.values:
            raise ValueError('split map or train ECDF mismatch across splits')
        windows = [dataset[i] for i in range(len(dataset))]
        summaries[split] = audit_windows(windows)
        identities[split] = [{'episode_key': w['episode_key'], 'terrain_seed': w['terrain_seed']} for w in windows]
        print('AUDITED', split, len(windows), flush=True)
        del windows, dataset
    assert_split_integrity(identities)
    split_hash = hashlib.sha256(json.dumps(split_map, sort_keys=True).encode()).hexdigest()
    terminal = {split: Counter() for split in ('train','val','test')}
    terminal_episodes = {split: set() for split in ('train','val','test')}
    records = Counter()
    manifest = json.loads(args.manifest.read_text())
    for chunk in manifest['chunks']:
        with gzip.open(args.manifest.parent / chunk['file'], 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                records['all'] += 1
                status = raw_dem_status(row)
                records['dem_' + status] += 1
                if not row['terminated'] and not row['truncated']:
                    continue
                split = split_map[int(row['terrain_seed'])]
                episode = (row['session_id'], row['env_id'], row['episode_id'])
                if episode in terminal_episodes[split]:
                    raise ValueError('duplicate terminal episode')
                terminal_episodes[split].add(episode)
                reason = '+'.join(row['termination_reason']) or 'unspecified'
                terminal[split][reason] += 1
    result = {'split_hash': split_hash, 'split_seed': 123,
              'terrain_seed_counts': dict(Counter(split_map.values())),
              'train_ecdf_count': len(train_ecdf),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'record_counts': dict(records),
              'terminal_episode_counts': {s: dict(c) for s,c in terminal.items()},
              'splits': summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('AUDIT_DONE', json.dumps({'split_hash':split_hash,'records':dict(records),
                                    'terminal':result['terminal_episode_counts'],
                                    'bins':{s:v['q_bin_counts'] for s,v in summaries.items()}}), flush=True)


if __name__ == '__main__':
    main()
