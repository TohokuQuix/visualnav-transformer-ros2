"""Read-only provenance and corrected-outcome check for the seven E1 groups."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from eat_nav_e1_analyze import load_trace, outcome, post_xy
from eat_nav_e1_coordinate_addendum import geometry_metrics, map_world_distance_equivalence

NEW = Path('/tmp/eatnav-expanded-feasibility-ros-continuation-20261005')
OLD = Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004')
OLD_CORRECTED = Path('/tmp/eatnav-e1-coordinate-addendum-r2-20261004')


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_nested(value):
    if isinstance(value, dict):
        return all(finite_nested(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_nested(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def same(a, b, label):
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        if not isinstance(b, (int, float)) or not math.isclose(a, b, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f'{label}: {a!r} != {b!r}')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f'{label}: keys differ')
        for key in a:
            same(a[key], b[key], f'{label}.{key}')
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError(f'{label}: list shape differs')
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{label}[{i}]')
    elif a != b:
        raise ValueError(f'{label}: {a!r} != {b!r}')


def audit_run(root: Path, process_id: str, saved: dict, old: bool):
    folder = root / 'runs' / process_id
    done = read(folder / 'run-complete.json')
    if done['identity']['run_id'] != process_id:
        raise ValueError(f'{process_id}: run identity mismatch')
    if read(folder / 'protocol-valid.json')['status'] != 'valid':
        raise ValueError(f'{process_id}: protocol invalid')
    stage_path = root / 'run-stage-markers' / f'{process_id}.json'
    if stage_path.exists():
        stage = read(stage_path)
        if stage['status'] != 'complete' or stage['run_complete_sha256'] != sha(folder / 'run-complete.json'):
            raise ValueError(f'{process_id}: stage marker mismatch')
    elif not old:
        raise ValueError(f'{process_id}: stage marker missing')
    for name, expected in done['output_sha256'].items():
        if sha(folder / name) != expected:
            raise ValueError(f'{process_id}: output SHA mismatch: {name}')
    trace_path = folder / 'live-0.trace.json.gz'
    trace = load_trace(trace_path)
    if len(trace) != 120 or any(row['terminal'] for row in trace) or not finite_nested(trace):
        raise ValueError(f'{process_id}: incomplete or terminal trace')
    for step, row in enumerate(trace):
        stamp = row['logical_sim_time_ns']
        if any(row[key] != stamp for key in ('clock_stamp_ns','tf_stamp_ns',
                'follower_command_stamp_ns','dem_timestamp_ns')):
            raise ValueError(f'{process_id}: exact-stamp mismatch step {step}')
        if row['path_stamp_ns'] != trace[0]['path_stamp_ns']:
            raise ValueError(f'{process_id}: Path stamp changed step {step}')
    summary = read(folder / 'live-0.json')
    path = np.asarray(summary['map_path'], dtype=float)
    if summary['fixed_map_path_sha256'] != done['identity']['map_path_file_sha256']:
        raise ValueError(f'{process_id}: Path file identity mismatch')
    path_file = root / 'inputs' / f"state-{done['identity']['state_index']:03d}" / f'{process_id}-map-path.json'
    if sha(path_file) != done['identity']['map_path_file_sha256'] or \
            not np.array_equal(path, np.asarray(read(path_file), dtype=float)):
        raise ValueError(f'{process_id}: saved candidate Path differs')
    origin = np.asarray(summary['anchor']['root_state'][:2], dtype=float) - path[0]
    world = np.stack([post_xy(row) for row in trace])
    eq = map_world_distance_equivalence(path, world, origin)
    if max(eq.values()) > 1e-10:
        raise ValueError(f'{process_id}: frame equivalence mismatch')
    metrics = geometry_metrics(path, world, origin, anchor_map_xy=path[0])
    raw = outcome(trace, summary, path)
    corrected = dict(raw, cross_track_p95_m=metrics['cross_track_0_6s_diagnostic_p95_m'],
                     endpoint_error_m=metrics['endpoint_error_at_6s_m'],
                     progress_m=metrics['radial_progress_0_6s_m'])
    expected = saved['corrected'] if old else saved['outcome']
    same(corrected, expected, process_id + '.outcome')
    if not old:
        same(metrics, saved['geometry'], process_id + '.geometry')
        if sha(trace_path) != saved['trace_sha256']:
            raise ValueError(f'{process_id}: saved trace SHA mismatch')
    else:
        if raw != saved['old']:
            raise ValueError(f'{process_id}: old outcome mismatch')
        if not saved['coordinate_independent_values_exactly_preserved']:
            raise ValueError(f'{process_id}: correction preservation false')
    return {'process_id': process_id, 'trace_sha256': sha(trace_path),
            'run_complete_sha256': sha(folder / 'run-complete.json'),
            'protocol_valid_sha256': sha(folder / 'protocol-valid.json'),
            'frame_equivalence_max': max(eq.values()),
            'candidate_id': saved['candidate_id'],
            'state_group_id': saved['state_group_id'],
            'episode_key': done['identity']['episode_key'],
            'anchor_step': done['identity']['anchor_step'],
            'map_path_file_sha256': done['identity']['map_path_file_sha256'],
            'seed': done['identity']['seed']}


def main():
    old_source = read(OLD_CORRECTED / 'source-provenance.json')
    for name, entry in old_source['frozen_source'].items():
        if sha(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'frozen source differs: {name}')
    checkpoint = Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt')
    reference = Path('/tmp/eatnav-vnt-goal-ood-simulator-v2/docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
    if sha(checkpoint) != old_source['checkpoint_sha256'] or sha(reference) != old_source['reference_config_sha256']:
        raise ValueError('checkpoint or reference config SHA differs')
    new_source = read(NEW / 'runtime-source-freeze-v2.json')
    if sha(Path(__file__).with_name('eat_nav_e1_sim.py')) != new_source['source_sha256']:
        raise ValueError('expanded simulator source SHA differs')
    old_records = read(OLD_CORRECTED / 'corrected-outcome-results.json')['per_run']['runs']
    new_records = {r['process_id']: r for r in read(NEW / 'analysis' / 'corrected-outcome-records.json')['records']}
    old_rows = read(OLD / 'original-20-row-manifest.json')['rows']
    new_rows = read(NEW / 'run-manifest.json')['rows']
    if len(old_rows) != 20 or len(new_rows) != 50:
        raise ValueError('manifest row count mismatch')
    if len(old_records) != 18 or len(new_records) != 45:
        raise ValueError('outcome record count mismatch')
    audited = []
    for root, rows, records, old in ((OLD, old_rows, old_records, True),
                                     (NEW, new_rows, new_records, False)):
        for row in rows:
            pid = row['process_id']
            if row['role'] == 'anchor_only':
                continue
            if pid not in records:
                raise ValueError(f'{pid}: outcome absent')
            result = audit_run(root, pid, records[pid], old)
            if result['candidate_id'] != row['path_id'] or result['state_group_id'] != row['state_group_id']:
                raise ValueError(f'{pid}: manifest identity mismatch')
            audited.append(result)
    if len(audited) != 63 or len({r['process_id'] for r in audited}) != 63:
        raise ValueError('live trace count or identity mismatch')
    groups = {}
    for row in audited:
        groups.setdefault(row['state_group_id'], []).append(row)
    if len(groups) != 7 or any(len(rows) != 9 for rows in groups.values()):
        raise ValueError('StateGroup composition mismatch')
    duplicate_sha = {}
    for sid, rows in groups.items():
        root = OLD if sid.startswith('g0e-') else NEW
        path = root / f'duplicate-gate-{sid}.json'
        gate = read(path)
        direct = sorted((r for r in rows if r['candidate_id'] == 'direct_reference'),
                        key=lambda r: r['process_id'])
        if len(direct) != 3 or gate['status'] != 'pass' or \
                gate['trace_sha256'] != [r['trace_sha256'] for r in direct] or \
                gate['run_complete_marker_sha256'] != [r['run_complete_sha256'] for r in direct]:
            raise ValueError(f'{sid}: duplicate gate mismatch')
        duplicate_sha[sid] = sha(path)
    new_ledger = read(NEW / 'analysis' / 'provenance-ledger.json')
    for row in audited:
        if row['state_group_id'].startswith('expanded-') and \
                new_ledger['run_complete_sha256'][row['process_id']] != row['run_complete_sha256']:
            raise ValueError(f"{row['process_id']}: provenance ledger mismatch")
    for root in (OLD, NEW):
        states_path = root / ('inputs/selected-states.json' if root == OLD else 'selected-states.json')
        selected = {s['state_id']: s for s in read(states_path)['train']}
        for sid, rows in groups.items():
            if sid not in selected:
                continue
            s = selected[sid]
            for row in rows:
                if row['episode_key'] != s['episode_key'] or row['anchor_step'] != s['step']:
                    raise ValueError(f"{row['process_id']}: selected state mismatch")
    return {'status': 'pass', 'live_trace_count': 63, 'group_count': 7,
            'groups': groups, 'duplicate_gate_sha256': duplicate_sha,
            'frozen_source_file_count_rechecked': len(old_source['frozen_source']),
            'checkpoint_sha256_rechecked': old_source['checkpoint_sha256'],
            'reference_config_sha256_rechecked': old_source['reference_config_sha256'],
            'expanded_simulator_source_sha256_rechecked': new_source['source_sha256'],
            'source_roots': {'old': str(OLD), 'new': str(NEW)},
            'source_sha256': {
                str(OLD_CORRECTED / 'corrected-outcome-results.json'): sha(OLD_CORRECTED / 'corrected-outcome-results.json'),
                str(NEW / 'analysis' / 'corrected-outcome-records.json'): sha(NEW / 'analysis' / 'corrected-outcome-records.json'),
                str(NEW / 'analysis' / 'final-report-detailed-r3.md'): sha(NEW / 'analysis' / 'final-report-detailed-r3.md')}}


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
