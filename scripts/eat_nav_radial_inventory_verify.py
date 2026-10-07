"""Verify sealed partial inventory and its stop semantics without simulator access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.radial_inventory import build_pose_schedule, pose_world

ROOT = Path('/tmp/eatnav-radial-terrain-inventory-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root=ROOT):
    root = Path(root)
    freeze = json.loads((root/'input-freeze.json').read_text())
    status = json.loads((root/'status.json').read_text())
    assert status['stage'] == 'stopped_anchor_gate' and status['completed_batches'] == 1
    assert status['retry_count'] == 0 and not (root/'runs/batch-20261008').exists()
    assert freeze['candidate_free'] and freeze['candidate_path_count'] == 0
    assert not freeze['external_zenoh_required'] and not freeze['bridge_follower_started']
    schedule = {x['pose_id']: x for x in build_pose_schedule(freeze['batch_seeds'])}
    paths = sorted((root/'runs/batch-20261007/poses').glob('*.json'))
    assert len(paths) == 112
    rows = [json.loads(path.read_text()) for path in paths]
    assert len({r['pose_id'] for r in rows}) == 112
    assert len({r['source_family'] for r in rows}) == 7
    assert all(r['batch_seed'] == 20261007 for r in rows)
    for record in rows:
        expected = schedule[record['pose_id']]
        assert (record['terrain'],record['row'],record['column'],record['terrain_seed']) == (
            expected['terrain'],expected['row'],expected['column'],expected['terrain_seed'])
        world = pose_world(record['env_origin'],record['row'])
        assert all(abs(record['requested_pose'][k] - v) < 1e-6 for k,v in world.items())
        history = record['dem_history']
        assert [h['step'] for h in history] == [4,9,14,19]
        assert [h['logical_stamp_ns'] for h in history] == [250_000_000,500_000_000,750_000_000,1_000_000_000]
        assert np.asarray([h['dem']['transformed'] for h in history]).shape == (4,2,20,20)
        assert record['actor_finite'] and record['dem_finite'] and record['penetration'] is None
        assert not any(k in record for k in ('reward','critic_value','candidate_path','outcome'))
    origins = json.loads((root/'runs/batch-20261007/env-origins.json').read_text())
    assert len(origins['origins']) == len(origins['terrain_metadata']) == 112
    assert all(m['column'] == schedule[f"b20261007-{m['terrain_type']}-a{m['row']:02d}"]['column']
               for m in origins['terrain_metadata'])
    seal = json.loads((root/'review-artifact-sha-ledger.json').read_text())
    for name,meta in seal['files'].items():
        p = root/name
        assert p.stat().st_size == meta['bytes'] and sha(p) == meta['sha256'],name
    assert len(list((root/'figures').glob('*.png'))) == 11
    assert json.loads((root/'final-decision.json').read_text())['small_pilot_authorized'] is False
    return {'status':'pass','pose_count':112,'families_recorded':7,'families_gate_eligible':0,
            'sha_checked_files':len(seal['files']),'figures':11,'no_outcome_or_path':True}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
