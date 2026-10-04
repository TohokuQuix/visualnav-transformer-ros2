"""Prepare an identity-only diagnostic view of the frozen readiness-002 input."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml


def build_alias_state(state):
    if (state.get('state_id') != 'readiness-000' or state.get('episode_key') != [0, 52, 0]
            or state.get('step') != 75 or state.get('local_step') != 75
            or state.get('terrain_seed') != 3455129400 or not state.get('boundary_clean')
            or not state.get('complete')):
        raise ValueError('frozen readiness identity differs')
    alias = copy.deepcopy(state)
    alias['original_state_id'] = state['state_id']
    alias['state_id'] = 'g0e-train-readiness-000'
    alias['global_step'] = state['step']
    alias['logged_goal'] = copy.deepcopy(state['goal'])
    return alias


def extract_frames(manifest, state):
    manifest = Path(manifest)
    source = json.loads(manifest.read_text())
    steps = [state['step']-15, state['step']-10, state['step']-5, state['step']]
    rows = {}
    for chunk in source['chunks']:
        path = manifest.parent/chunk['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk['sha256']:
            raise ValueError('raw collection chunk SHA differs')
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['step'] in steps and row['env_id'] == state['episode_key'][1]:
                    if row['step'] in rows:
                        raise ValueError('duplicate DEM source frame')
                    rows[row['step']] = row
        if len(rows) == 4:
            break
    if len(rows) != 4:
        raise ValueError('missing DEM source frame')
    result = [rows[step] for step in steps]
    if any([row['session_id'], row['env_id'], row['episode_id']] != state['episode_key']
           or row['pre_reset'] or row['timestamp_ns'] != (step+1)*50_000_000
           for step, row in zip(steps, result)):
        raise ValueError('DEM source frame identity or stamp differs')
    dem = np.asarray([row['dem']['transformed'] for row in result], dtype=np.float32)
    if dem.shape != (4, 2, 20, 20) or not np.isfinite(dem).all():
        raise ValueError('DEM source frame shape or finite gate failed')
    if hashlib.sha256(dem.tobytes()).hexdigest() != state['dem_sha256']:
        raise ValueError('DEM source history SHA differs')
    return result


def path_digest(points):
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (9, 2) or not np.isfinite(values).all():
        raise ValueError('nine-point Path required')
    return hashlib.sha256(values[1:].tobytes()).hexdigest()


def prepare_fixture(root):
    root = Path(root)
    inputs = root/'inputs'
    if inputs.exists():
        raise FileExistsError('one-shot diagnostic inputs already frozen')
    source = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')
    original = json.loads((source/'selected-states.json').read_text())['train'][0]
    old_entry = json.loads((Path('/tmp/eatnav-vnt-candidate-coverage')/
        'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot/schedule.json').read_text())['runs'][2]
    if (old_entry['run_id'] != 'readiness-002' or old_entry['candidate_id'] != 'bank-nominal'
            or old_entry['episode_key'] != original['episode_key']
            or old_entry['logical_t0_ns'] != 1020000000000):
        raise ValueError('frozen readiness run identity differs')
    alias = build_alias_state(original)
    frames = extract_frames('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json', original)
    raw = np.asarray([row['dem']['transformed'] for row in frames], dtype=np.float32)
    folder = inputs/'state-000'
    folder.mkdir(parents=True)
    old_folder = source/'state-000'
    candidate = 'diagnostic-readiness-002'
    copied = {'prefix-actions.npz':'prefix-actions.npz',
              'suffix-actions-120.npz':'suffix-actions-120.npz',
              'logged-anchor.json':'logged-anchor.json',
              f'{candidate}-base-path.json':'bank-nominal-base-path.json',
              f'{candidate}-map-path.json':'bank-nominal-map-path.json'}
    for target_name, source_name in copied.items():
        shutil.copyfile(old_folder/source_name, folder/target_name)
        if hashlib.sha256((folder/target_name).read_bytes()).hexdigest() != hashlib.sha256((old_folder/source_name).read_bytes()).hexdigest():
            raise ValueError('diagnostic copy SHA differs')
    base = json.loads((folder/f'{candidate}-base-path.json').read_text())
    mapped = json.loads((folder/f'{candidate}-map-path.json').read_text())
    (inputs/'selected-states.json').write_text(json.dumps({'train':[alias]},sort_keys=True,indent=2)+'\n')
    (folder/'recorded-frames.json').write_text(json.dumps(frames,sort_keys=True,indent=2)+'\n')
    with (inputs/'g0e-selected-states.npz').open('wb') as stream:
        np.savez_compressed(stream,goal=np.asarray([original['goal']],dtype=np.float32),
                            dem=raw[np.newaxis])
    config = yaml.safe_load((old_folder/'config-120.yaml').read_text())
    formal = yaml.safe_load(Path('/tmp/eatnav-g0e-formal-anchor-20261003/formal-anchor-config.yaml').read_text())
    config['scope'] = 'e1_live_candidate'
    config['anchor_limits'] = formal['anchor_limits']
    (folder/'config-120.yaml').write_text(yaml.safe_dump(config,sort_keys=True))
    metadata = {'state_id':alias['state_id'],'run_id':candidate,
                'original_run_id':old_entry['run_id'],'original_state_id':original['state_id'],
                'base_path_sha256':hashlib.sha256((folder/f'{candidate}-base-path.json').read_bytes()).hexdigest(),
                'original_base_path_sha256':old_entry['base_path_sha256'],
                'goal_xy':base[-1]}
    (folder/f'{candidate}-goal-metadata.json').write_text(json.dumps(metadata,sort_keys=True,indent=2)+'\n')
    entry = {'order':old_entry['order'],'run_id':candidate,'kind':'duplicate',
             'state_id':alias['state_id'],'original_state_id':original['state_id'],
             'original_run_id':old_entry['run_id'],'state_index':0,'candidate_id':candidate,
             'seed':old_entry['seed'],'anchor_step':original['step'],
             'episode_key':original['episode_key'],'logical_t0_ns':old_entry['logical_t0_ns'],
             'goal':original['goal'],'local_path_sha256':path_digest(base),
             'map_path_sha256':path_digest(mapped),
             'base_path_sha256':metadata['base_path_sha256'],
             'map_path_file_sha256':hashlib.sha256((folder/f'{candidate}-map-path.json').read_bytes()).hexdigest(),
             'goal_metadata_sha256':hashlib.sha256((folder/f'{candidate}-goal-metadata.json').read_bytes()).hexdigest()}
    (root/'diagnostic-entry.json').write_text(json.dumps(entry,sort_keys=True,indent=2)+'\n')
    (root/'diagnostic-input-provenance.json').write_text(json.dumps({
        'physical_state_unchanged':True,'original_entry':old_entry,
        'original_selected_states_sha256':hashlib.sha256((source/'selected-states.json').read_bytes()).hexdigest(),
        'original_state_sha256':original['dem_sha256'],
        'copied_file_sha256':{k:hashlib.sha256((folder/k).read_bytes()).hexdigest() for k in copied},
        'derived_frame_sha256':hashlib.sha256((folder/'recorded-frames.json').read_bytes()).hexdigest(),
        'derived_dem_raw_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
        'diagnostic_identity_alias':alias['state_id'],
        'added_state_fields':['original_state_id','global_step','logged_goal'],
        'added_config_fields':['scope','anchor_limits.command_max_abs','anchor_limits.velocity_max_abs'],
        'original_prefix_suffix_and_Path_bytes_unchanged':True},sort_keys=True,indent=2)+'\n')
    return entry


def validate_fixture(root, entry):
    root = Path(root)
    source = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')
    original = json.loads((source/'selected-states.json').read_text())['train'][0]
    alias = build_alias_state(original)
    folder = root/'inputs/state-000'
    if entry != json.loads((root/'diagnostic-entry.json').read_text()):
        raise ValueError('diagnostic run entry differs')
    if json.loads((root/'inputs/selected-states.json').read_text())['train'] != [alias]:
        raise ValueError('diagnostic state identity differs')
    copied = {'prefix-actions.npz':'prefix-actions.npz',
              'suffix-actions-120.npz':'suffix-actions-120.npz',
              'logged-anchor.json':'logged-anchor.json',
              f"{entry['candidate_id']}-base-path.json":'bank-nominal-base-path.json',
              f"{entry['candidate_id']}-map-path.json":'bank-nominal-map-path.json'}
    for target_name, source_name in copied.items():
        if (folder/target_name).read_bytes() != (source/'state-000'/source_name).read_bytes():
            raise ValueError('physical readiness input SHA differs')
    base = json.loads((folder/f"{entry['candidate_id']}-base-path.json").read_text())
    mapped = json.loads((folder/f"{entry['candidate_id']}-map-path.json").read_text())
    if (path_digest(base) != entry['local_path_sha256'] or
            path_digest(mapped) != entry['map_path_sha256'] or
            hashlib.sha256((folder/f"{entry['candidate_id']}-map-path.json").read_bytes()).hexdigest()
            != entry['map_path_file_sha256']):
        raise ValueError('Path SHA differs')
    frames = json.loads((folder/'recorded-frames.json').read_text())
    raw = np.asarray([row['dem']['transformed'] for row in frames], dtype=np.float32)
    if [row['step'] for row in frames] != [60,65,70,75] or hashlib.sha256(raw.tobytes()).hexdigest() != original['dem_sha256']:
        raise ValueError('diagnostic four-frame DEM differs')
    with np.load(root/'inputs/g0e-selected-states.npz') as selected:
        if not np.array_equal(selected['dem'], raw[np.newaxis]) or not np.array_equal(
                selected['goal'], np.asarray([original['goal']],dtype=np.float32)):
            raise ValueError('diagnostic selected DEM/goal differs')
    config = yaml.safe_load((folder/'config-120.yaml').read_text())
    formal = yaml.safe_load(Path('/tmp/eatnav-g0e-formal-anchor-20261003/formal-anchor-config.yaml').read_text())
    if config['scope'] != 'e1_live_candidate' or config['anchor_limits'] != formal['anchor_limits'] or config['control_steps'] != 120:
        raise ValueError('E1 diagnostic config differs')
    if entry['episode_key'] != original['episode_key'] or entry['anchor_step'] != original['step'] or entry['goal'] != original['goal']:
        raise ValueError('diagnostic physical state changed')
    return folder
