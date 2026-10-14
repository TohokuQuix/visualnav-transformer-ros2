"""Detached Stage B→C gate runner; never trains on partial collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from eat_nav_motion_14_train import build, evaluate, overfit, train


def atomic_status(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def run(collection_root: Path, training_root: Path, status_path: Path) -> None:
    if training_root.exists():
        raise FileExistsError('fresh Stage C training root required')
    if status_path.exists():
        raise FileExistsError('fresh Stage C pipeline status required')
    atomic_status(status_path, {'state': 'waiting_for_full_collection',
                  'collection_root': str(collection_root),
                  'training_root': str(training_root), 'updated_ns': time.time_ns()})
    while True:
        source = json.loads((collection_root/'status.json').read_text())
        if source.get('state') == 'completed' and source.get('completed') == 1152:
            break
        if source.get('state') == 'stopped':
            atomic_status(status_path, {'state': 'blocked_by_collection_failure',
                          'collection_status': source, 'updated_ns': time.time_ns()})
            return
        time.sleep(30)
    try:
        for stage, action in (('building_dataset', lambda: build(training_root, collection_root)),
                              ('small_overfit', lambda: overfit(training_root)),
                              ('training', lambda: train(training_root)),
                              ('heldout_evaluation', lambda: evaluate(training_root))):
            atomic_status(status_path, {'state': stage, 'updated_ns': time.time_ns(),
                          'collection_root': str(collection_root),
                          'training_root': str(training_root)})
            action()
        result = json.loads((training_root/'results.json').read_text())
        atomic_status(status_path, {'state': 'completed', 'updated_ns': time.time_ns(),
                      'stage_c_adoption': result['demo_adoption'],
                      'training_root': str(training_root)})
    except Exception as error:
        atomic_status(status_path, {'state': 'stopped', 'updated_ns': time.time_ns(),
                      'failure': f'{type(error).__name__}: {error}',
                      'training_root': str(training_root)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    options = parser.parse_args()
    run(options.collection_root, options.training_root, options.status)
