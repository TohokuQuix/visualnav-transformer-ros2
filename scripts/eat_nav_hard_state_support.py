"""Outcome-blind geometry support count for short-prefix complete-only train states."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import yaml

from eat_nav_evaluator_design_audit import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/quality-identifiability-gate/evaluator-design-review'
CONFIG = OUT / 'hard-state-support-config.yaml'
MANIFEST = Path('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json')
LIST = Path('/tmp/eatnav-candidate-bank-g0e/g0e-selection-candidates.jsonl.gz')
BANK = Path('/tmp/eatnav-candidate-bank-g0e/g0e-candidate-bank.jsonl.gz')


def load_jsonl(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def sample_profile(dem, centerline):
    p = np.asarray(centerline, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or np.max(np.abs(p)) > 1.000001:
        raise ValueError('frozen centerline outside 2 m DEM')
    p = p[np.rint(np.linspace(0, len(p)-1, 64)).astype(int)]
    col = np.clip(np.floor((p[:, 0]+1)/.1).astype(int), 0, 19)
    row = np.clip(np.floor((p[:, 1]+1)/.1).astype(int), 0, 19)
    return dem[0, row, col], dem[1, row, col] > .5


def main():
    config = yaml.safe_load(CONFIG.read_text())
    expected = {'manifest':'55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475',
                'candidate_list':'aa0ec83560db08af59e5b27b9d5dcd6cbb0fc9311b08ecb1db838f5de37e4aed',
                'bank':'c68815526efc8150b229025bfaa196d9ff9edc58076b2042ae5295e5ec985f02'}
    actual = {'manifest': sha(MANIFEST), 'candidate_list': sha(LIST), 'bank': sha(BANK)}
    if actual != expected:
        raise ValueError(f'source SHA mismatch {actual}')
    bank = {r['candidate_id']: r for r in load_jsonl(BANK) if r['state_id']=='g0e-train-000'}
    if len(bank) != 7 or any(r['type_A_reasons'] for r in bank.values()):
        raise ValueError('candidate geometry invalid')
    excluded = set(config['exclude_existing_environment_ids'])
    all_list = list(load_jsonl(LIST))
    if len(all_list) != 49598:
        raise ValueError('complete-only train count differs')
    short_before_prior_probe_exclusion = [r for r in all_list if r['source_episode']==0 and r['episode_key'][2]==0
                  and r['prefix_steps']<=config['max_prefix_control_steps']
                  and r['global_step']==r['local_step']
                  and r['env_id'] not in excluded]
    candidates = [r for r in short_before_prior_probe_exclusion if
                  not (config['exclude_previously_probed_anchors'] and r['exclusion_reasons'])]
    wanted = {(tuple(r['episode_key']), r['global_step']): r for r in candidates}
    max_step = max(r['global_step'] for r in candidates)
    needed = set()
    for r in candidates:
        for lag in (15, 10, 5, 0):
            needed.add((tuple(r['episode_key']), r['global_step']-lag))
    frames = {}
    source = json.loads(MANIFEST.read_text())
    checked_chunks = []
    for chunk in source['chunks']:
        file = MANIFEST.parent / chunk['file']
        if sha(file) != chunk['sha256']:
            raise ValueError('source chunk SHA differs '+str(file))
        checked_chunks.append({'file': chunk['file'], 'sha256': chunk['sha256']})
        with gzip.open(file, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                step = int(row['step'])
                if step > max_step:
                    continue
                key = ((row['session_id'], row['env_id'], row['episode_id']), step)
                if key in needed:
                    if key in frames or row['pre_reset']:
                        raise ValueError('duplicate or reset-crossing frame '+str(key))
                    frames[key] = row
        if len(frames) == len(needed):
            break
    if set(frames) != needed:
        raise ValueError(f'missing {len(needed)-len(frames)} DEM frames')
    counts = defaultdict(Counter)
    support = defaultdict(list)
    for key, candidate in wanted.items():
        terrain = candidate['terrain']
        counts[terrain]['short_prefix_complete_windows'] += 1
        history = []
        for lag in (15, 10, 5, 0):
            row = frames[(key[0], key[1]-lag)]
            if row['terrain_type'] != terrain or row['terrain_seed'] != candidate['terrain_seed']:
                raise ValueError('terrain identity mismatch')
            dem = np.asarray(row['dem']['transformed'], dtype=np.float32)
            if dem.shape != (2,20,20) or not np.isfinite(dem).all() or not np.logical_or(dem[1]==0, dem[1]==1).all():
                raise ValueError('DEM shape/mask/finite mismatch')
            history.append(dem)
        current = history[-1]
        profiles = {cid: sample_profile(current, bank[cid]['dense_centerline_xy_m']) for cid in bank}
        direct_h, direct_m = profiles['direct_reference']
        known = direct_h[direct_m]
        if len(known) < 2:
            counts[terrain]['insufficient_direct_known'] += 1
            continue
        exposure = {'height_range_m': float(np.ptp(known)),
                    'roughness_std_m': float(np.std(known)),
                    'adjacent_height_jump_m': float(np.max(np.abs(np.diff(known)))),
                    'slope_abs': float(np.max(np.abs(np.diff(known))/.1))}
        if not any(exposure[name.removesuffix('_min')] >= threshold for name, threshold in config['direct_exposure_any'].items()):
            counts[terrain]['no_direct_exposure'] += 1
            continue
        distinct = []
        for cid in bank:
            if cid == 'direct_reference':
                continue
            h, mask = profiles[cid]
            common = direct_m & mask
            if common.mean() < config['centerline_common_known_fraction_min']:
                continue
            if np.mean(np.abs(h[common]-direct_h[common])) >= config['detour_difference']['mean_absolute_height_profile_m_min']:
                distinct.append(cid)
        if len(distinct) < config['detour_difference']['required_detour_count_min']:
            counts[terrain]['no_distinct_detour_profile'] += 1
            continue
        counts[terrain]['geometry_supported_windows'] += 1
        rank = hashlib.sha256((config['selection_salt']+'\n'+json.dumps([candidate['episode_key'],candidate['global_step'],candidate['terrain_seed']],separators=(',',':'))).encode()).hexdigest()
        support[terrain].append({'episode_key':candidate['episode_key'], 'terrain_seed':candidate['terrain_seed'],
                                 'step':candidate['global_step'], 'prefix_steps':candidate['prefix_steps'],
                                 'direct_exposure':exposure, 'distinct_detours':distinct, 'selection_rank':rank})
    result = {}
    for terrain in sorted({r['terrain'] for r in all_list}):
        available = support[terrain]
        by_episode = defaultdict(list)
        for row in available:
            by_episode[tuple(row['episode_key'])].append(row)
        unique = [min(rows, key=lambda row: row['selection_rank']) for rows in by_episode.values()]
        unique.sort(key=lambda row: row['selection_rank'])
        prefix = [row['prefix_steps'] for row in available]
        result[terrain] = {'counts': dict(counts[terrain]), 'independent_episode_support': len(unique),
                           'independent_terrain_seed_support': len({r['terrain_seed'] for r in unique}),
                           'supported_prefix_steps':{'min':min(prefix), 'median':statistics.median(prefix), 'max':max(prefix)} if prefix else None,
                           'prospective_first_two': unique[:2],
                           'two_state_target_supported': len(unique) >= 2}
    return {'status':'support_audited_not_selected', 'threshold_config_sha256':sha(CONFIG),
            'source_sha256':actual, 'source_chunk_sha256_checked':checked_chunks,
            'complete_only_train_count':len(all_list), 'short_prefix_before_prior_probe_exclusion':len(short_before_prior_probe_exclusion),
            'previously_probed_anchors_excluded':len(short_before_prior_probe_exclusion)-len(candidates),
            'short_prefix_candidates':len(candidates),
            'terrain':result, 'outcome_fields_read':False,
            'selection_still_requires_review':True}


if __name__ == '__main__':
    result = main()
    (OUT/'hard-state-support-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({t:(x['counts'].get('geometry_supported_windows',0), x['independent_episode_support']) for t,x in result['terrain'].items()},sort_keys=True))
