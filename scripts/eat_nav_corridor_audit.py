"""Read-only integrity audit for the frozen path-corridor experiment."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem


ROOT = Path('/tmp/eatnav-evaluator-independent-expansion-20261006-v7')
OLD_PROVENANCE = Path('docs/quality-identifiability-gate/evaluator-offline-smoke/source-provenance.json')


def source_frames_path(root: Path, group: str) -> Path:
    if group.startswith('indep-train-'):
        state = group.split('-')[2]
        return root / 'inputs' / f'state-{state}' / 'recorded-frames.json'
    if group.startswith('expanded-train-'):
        state = group.rsplit('-', 1)[1]
        return Path('/tmp/eatnav-expanded-feasibility-ros-continuation-20261005') / 'inputs' / f'state-{state}' / 'recorded-frames.json'
    if group.startswith('g0e-train-'):
        state = group.rsplit('-', 1)[1]
        return Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004') / 'inputs' / f'state-{state}' / 'recorded-frames.json'
    raise ValueError(f'unknown source group: {group}')


def validate_frame_sequence(frames: list[dict]) -> None:
    if len(frames) != 4 or [row['step'] for row in frames] != list(range(frames[0]['step'], frames[0]['step'] + 20, 5)):
        raise ValueError('source DEM frame step sequence mismatch')
    if any(left['timestamp_ns'] >= right['timestamp_ns'] for left, right in zip(frames, frames[1:])):
        raise ValueError('source DEM timestamp sequence mismatch')
    if any(row['pre_reset'] or row['terminated'] or row['truncated'] for row in frames):
        raise ValueError('source DEM reset/termination contamination')


def validate_source_frames(frames: list[dict]) -> dict:
    if len(frames) != 4:
        raise ValueError('source history must have four frames')
    validate_frame_sequence(frames)
    rms, mask_iou = [], []
    identity = {(row['session_id'], row['env_id'], row['episode_id']) for row in frames}
    if len(identity) != 1 or len({row['timestamp_ns'] for row in frames}) != 4:
        raise ValueError('source DEM temporal identity mismatch')
    for frame in frames:
        dem = frame['dem']
        if dem['axis'] != 'row=+world_y,col=+world_x' or dem['resolution_m'] != 0.1 or dem['unknown'] != 'mask=0,height=0':
            raise ValueError('source DEM axis/resolution/unknown mismatch')
        height = np.asarray(dem['source_height'], dtype=float)
        mask = np.asarray(dem['source_mask'], dtype=bool)
        transformed = np.asarray(dem['transformed'], dtype=float)
        if height.shape != (20, 20) or mask.shape != (20, 20) or transformed.shape != (2, 20, 20):
            raise ValueError('source DEM shape mismatch')
        if not np.isfinite(height[mask]).all() or not np.isfinite(transformed).all():
            raise ValueError('source DEM nonfinite')
        grid = SourceGrid(height, mask, tuple(dem['origin_xy']), .1, (0, 0),
                          'map', frame['timestamp_ns'])
        reconstructed = robot_dem(grid, (frame['pose'][0], frame['pose'][1], frame['yaw']), EATConfig())
        both = (reconstructed[1] > .5) & (transformed[1] > .5)
        union = (reconstructed[1] > .5) | (transformed[1] > .5)
        error = float(np.sqrt(np.mean((reconstructed[0][both] - transformed[0][both])**2))) if both.any() else float('inf')
        iou = float(both.sum() / union.sum()) if union.any() else 0.0
        if error > .03 or iou < .95:
            raise ValueError(f'source DEM transform mismatch: RMS={error}, mask IoU={iou}')
        rms.append(error)
        mask_iou.append(iou)
    return {'max_dem_rms_m': max(rms), 'min_mask_iou': min(mask_iou),
            'source_episode_key': list(next(iter(identity))),
            'anchor_step': frames[-1]['step'], 'anchor_timestamp_ns': frames[-1]['timestamp_ns'],
            'anchor_pose': frames[-1]['pose']}


def validate_live_identity(summary: dict, path: np.ndarray, goal: np.ndarray,
                           source_pose_xyz: list[float]) -> dict:
    metadata = summary['goal_metadata']
    saved_path = np.asarray(metadata['base_path'][1:], dtype=float)
    if saved_path.shape != (8, 2) or np.max(np.abs(saved_path - path)) > 1e-6:
        raise ValueError('Path identity mismatch')
    if np.max(np.abs(np.asarray(metadata['goal_xy'], dtype=float) - goal)) > 1e-6:
        raise ValueError('goal identity mismatch')
    position_error = float(np.linalg.norm(np.asarray(summary['anchor']['root_state'][:3]) - source_pose_xyz))
    if position_error > .02:
        raise ValueError(f'anchor pose identity mismatch: {position_error}')
    return {'max_path_error_m': float(np.max(np.abs(saved_path - path))),
            'max_goal_error_m': float(np.max(np.abs(np.asarray(metadata['goal_xy']) - goal))),
            'anchor_position_error_m': position_error}


def validate_label_trace(label: dict, process: str, expected_sha: str) -> None:
    if (label['process_id'] != process or label['source_trace_sha256'] != expected_sha or
            not label['valid']):
        raise ValueError(f'label trace SHA/validity mismatch: {process}')


def validate_manifest_counts(manifest: dict) -> None:
    rows = manifest['rows']
    if (manifest['anchor_count'] != 14 or manifest['live_count'] != 189 or
            manifest['process_count'] != 203 or len(rows) != 203 or
            sum(row['role'] == 'anchor_only' for row in rows) != 14):
        raise ValueError('manifest count mismatch')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_sha(path: Path, expected: str) -> str:
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'SHA mismatch: {path}: expected {expected}, got {actual}')
    return actual


def read(path: Path):
    return json.loads(path.read_text())


def validate_records(rows: list[dict], expected_groups: int = 28) -> dict:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row['process_id'] in seen:
            raise ValueError('duplicate process identity')
        seen.add(row['process_id'])
        groups[row['group']].append(row)
    if len(groups) != expected_groups:
        raise ValueError(f'group count {len(groups)} != {expected_groups}')
    for group, members in groups.items():
        candidates = Counter(row['candidate'] for row in members)
        direct = sorted(row['repeat'] for row in members if row['candidate'] == 'direct_reference')
        if (len(members) != 9 or direct != [0, 1, 2] or
                candidates['direct_reference'] != 3 or len(candidates) != 7 or
                any(count != 1 for name, count in candidates.items() if name != 'direct_reference')):
            raise ValueError(f'candidate composition: {group}')
        if len({row['family'] for row in members}) != 1 or len({row['terrain'] for row in members}) != 1:
            raise ValueError(f'inconsistent group identity: {group}')
    return {'group_count': len(groups), 'record_count': len(rows),
            'old_groups': sum(members[0].get('origin') == 'existing' for members in groups.values()),
            'new_groups': sum(members[0].get('origin') == 'new' for members in groups.values())}


def validate_folds(rows: list[dict], folds: list[dict], expected_folds: int = 40) -> dict:
    groups = {row['group'] for row in rows}
    family = {row['group']: row['family'] for row in rows}
    terrain = {row['group']: row['terrain'] for row in rows}
    if len(folds) != expected_folds:
        raise ValueError(f'fold count {len(folds)} != {expected_folds}')
    counts = Counter()
    for fold in folds:
        train, test = set(fold['train']), set(fold['test'])
        if not train or not test or train & test or train | test != groups:
            raise ValueError(f'group leakage/incomplete fold: {fold["fold_id"]}')
        if fold['mode'] != 'E-goal' and {family[g] for g in train} & {family[g] for g in test}:
            raise ValueError(f'family leakage: {fold["fold_id"]}')
        if fold['mode'] == 'E-terrain' and {terrain[g] for g in train} & {terrain[g] for g in test}:
            raise ValueError(f'terrain leakage: {fold["fold_id"]}')
        if not fold.get('test_never_selects_checkpoint', True):
            raise ValueError(f'test checkpoint selection: {fold["fold_id"]}')
        counts[fold['mode']] += 1
    return dict(counts)


def audit(root: Path, repository: Path) -> dict:
    ledger = read(root / 'artifact-sha-ledger.json')
    checked = {}
    for relative, expected in ledger['root_files'].items():
        checked[str(root / relative)] = check_sha(root / relative, expected)
    execution_freeze = read(root / 'execution-source-freeze.json')
    for name, path in execution_freeze['source_path'].items():
        checked[str(path)] = check_sha(Path(path), execution_freeze['source_sha256'][name])
    offline_freeze = read(root / 'offline-freeze.json')
    check_sha(root / 'split-folds.json', offline_freeze['split_folds_sha256'])
    check_sha(root / 'training-config.yaml', offline_freeze['training_config_sha256'])
    rows = read(root / 'record-identity.json')
    identities = validate_records(rows)
    folds = read(root / 'split-folds.json')['folds']
    modes = validate_folds(rows, folds)
    old = read(repository / OLD_PROVENANCE)
    old_runs = {row['process_id']: row for row in old['sealed_runs']}
    manifest = read(root / 'run-manifest.json')
    validate_manifest_counts(manifest)
    new_manifest = {row['process_id']: row for row in manifest['rows']}
    for row in manifest['rows']:
        if row['role'] != 'anchor_only':
            continue
        process = row['process_id']
        run_dir = root / 'runs' / process
        complete = run_dir / 'run-complete.json'
        marker = read(root / 'run-stage-markers' / (process + '.json'))
        checked[str(complete)] = check_sha(complete, marker['run_complete_sha256'])
        for relative, expected in read(complete)['output_sha256'].items():
            path = run_dir / relative
            checked[str(path)] = check_sha(path, expected)
    with gzip.open(root / 'combined-labels.jsonl.gz', 'rt') as stream:
        labels = {item['process_id']: item for item in map(json.loads, stream)}
    if len(labels) != len(rows):
        raise ValueError('label count/identity mismatch')
    frame_audits = {}
    for group in sorted({row['group'] for row in rows}):
        path = source_frames_path(root, group)
        checked[str(path)] = sha(path)
        frame_audits[group] = validate_source_frames(read(path))
    with np.load(root / 'combined-features.npz') as features:
        paths = features['candidate_path']
        goals = features['relative_goal']
        if paths.shape != (252, 8, 2) or goals.shape != (252, 2):
            raise ValueError('feature Path/goal shape mismatch')
        identity_results = []
        for index, row in enumerate(rows):
            process = row['process_id']
            summary_path = (Path(old_runs[process]['summary_path']) if row['origin'] == 'existing'
                            else root / 'runs' / process / 'live-0.json')
            identity_results.append(validate_live_identity(read(summary_path), paths[index], goals[index],
                                                         frame_audits[row['group']]['anchor_pose'][:3]))
    for row in rows:
        process = row['process_id']
        if row['origin'] == 'existing':
            source = old_runs[process]
            if source['state_group'] != row['group'] or source['candidate'] != row['candidate']:
                raise ValueError(f'old identity mismatch: {process}')
            for field in ('summary', 'trace'):
                path = Path(source[field + '_path'])
                checked[str(path)] = check_sha(path, source[field + '_sha256'])
            validate_label_trace(labels[process], process, source['trace_sha256'])
            continue
        source = new_manifest[process]
        if source['state_group_id'] != row['group'] or source['path_id'] != row['candidate']:
            raise ValueError(f'new identity mismatch: {process}')
        run_dir = root / 'runs' / process
        complete = run_dir / 'run-complete.json'
        marker = read(root / 'run-stage-markers' / (process + '.json'))
        checked[str(complete)] = check_sha(complete, marker['run_complete_sha256'])
        for relative, expected in read(complete)['output_sha256'].items():
            path = run_dir / relative
            checked[str(path)] = check_sha(path, expected)
        validate_label_trace(labels[process], process,
                             read(complete)['output_sha256']['live-0.trace.json.gz'])
    return {'status': 'pass', **identities, 'fold_modes': modes,
            'old_run_count': len(old_runs), 'new_live_run_count': sum(row['origin'] == 'new' for row in rows),
            'new_anchor_only_run_count': 14,
            'rl_checkpoint_container_path': execution_freeze['rl_checkpoint_container_path'],
            'rl_checkpoint_expected_sha256': execution_freeze['rl_checkpoint_sha256'],
            'execution_source_file_count': len(execution_freeze['source_path']),
            'source_frame_audits': frame_audits,
            'live_identity_max_errors': {key: max(item[key] for item in identity_results)
                                         for key in identity_results[0]},
            'sha_ledger': checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('sha_ledger', 'source_frame_audits')}, sort_keys=True))


if __name__ == '__main__':
    main()
