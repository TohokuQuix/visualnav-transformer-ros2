"""Frozen Phase C coverage diagnostics. No training or formal rollout entry point."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import yaml

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from eat_nav_phase_c_candidate_diversity import select_states


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def validate_config(config):
    if not config['reference']['heldout_forbidden'] or config['formal_phase_c_allowed']:
        raise ValueError('forbidden split or formal rollout enabled')
    seeds = config['states']['diffusion_seeds']
    if seeds != list(range(1001, 1033)) or config['sampling']['baseline_k'] != [8, 32]:
        raise ValueError('frozen K8/K32 seed schedule changed')
    if config['structured']['basis'] != 'global_complete_train_residual_pca':
        raise ValueError('unfrozen structured basis')


def require_train_only(record):
    if record['split'] != 'train':
        raise ValueError('train-only operation received nontrain data')


def write_marker(path, outputs, input_hash):
    atomic(path, {'input_sha256': input_hash,
                  'outputs_sha256': {str(p): digest(p) for p in outputs},
                  'finished_unix': time.time()})


def marker_valid(path, input_hash):
    path = Path(path)
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
        return row['input_sha256'] == input_hash and all(Path(p).exists() and digest(p) == h
                for p, h in row['outputs_sha256'].items())
    except (OSError, ValueError, KeyError):
        return False


def verified_view(manifest, config_path, split, expected):
    cfg = load_eat_config(config_path)
    view = CompleteOnlyView(EATDataset(manifest, split, cfg, split_seed=123), cfg.len_traj_pred)
    if len(view) != expected:
        raise ValueError(f'{split} expected {expected} complete windows, found {len(view)}')
    if any(not np.asarray(view[i]['action_mask']).all() for i in range(len(view))):
        raise ValueError('partial trajectory in complete-only view')
    return view


def freeze(args, spec):
    out = args.output
    config_hash = digest(args.coverage_config)
    paths = {'manifest': args.manifest, 'reference_config': args.reference_config,
             'checkpoint': args.checkpoint, 'coverage_config': args.coverage_config,
             'source': Path(__file__), 'matching_source': Path(__file__).parent/'src/visualnav_transformer/eat_nav/relative_quality_matching.py',
             'model_source': Path(__file__).parent/'src/visualnav_transformer/eat_nav/model.py',
             'dataset_source': Path(__file__).parent/'src/visualnav_transformer/eat_nav/dataset.py',
             'prior_states': args.prior_states, 'prior_thresholds': args.prior_thresholds,
             'stage1_source': Path(__file__).parent/'eat_nav_phase_c_coverage_stage1.py',
             'sampling_source': Path(__file__).parent/'eat_nav_phase_c_coverage_sampling.py',
             'structured_source': Path(__file__).parent/'eat_nav_phase_c_coverage_structured.py',
             'evaluator_source': Path(__file__).parent/'eat_nav_phase_c_coverage_evaluator.py',
             'phase_b_evaluator_source': Path(__file__).parent/'eat_nav_forward_evaluator.py',
             'ros_bridge_source': Path(__file__).parent/'eat_nav_phase_c_ros_bridge.py',
             'phase_b_scalers':Path('/tmp/eatnav-quality-identifiability-gate/relative-quality/output/relative-quality-extraction.json'),
             'phase_b_gt_11':Path('/tmp/eatnav-quality-identifiability-gate/forward-evaluator/checkpoints/GT-11/best.pt'),
             'phase_b_gt_29':Path('/tmp/eatnav-quality-identifiability-gate/forward-evaluator/checkpoints/GT-29/best.pt'),
             'phase_b_gt_47':Path('/tmp/eatnav-quality-identifiability-gate/forward-evaluator/checkpoints/GT-47/best.pt')}
    hashes = {name: digest(path) for name,path in paths.items()}
    for key, expected in [('manifest', 'manifest_sha256'), ('reference_config', 'config_sha256'), ('checkpoint', 'checkpoint_sha256')]:
        if hashes[key] != spec['reference'][expected]:
            raise ValueError(f'frozen {key} SHA mismatch')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'freeze output exists: {out}')
    out.mkdir(parents=True,exist_ok=True)
    prior = json.loads(args.prior_states.read_text())
    if prior['split'] != 'train' or len(prior['states']) != spec['states']['train_count']:
        raise ValueError('prior train states invalid')
    thresholds = json.loads(args.prior_thresholds.read_text())
    if thresholds['source_split'] != 'complete_only_train':
        raise ValueError('prior OOD threshold not train-only')
    train = verified_view(args.manifest, args.reference_config, 'train', spec['reference']['complete_train_windows'])
    val = verified_view(args.manifest, args.reference_config, 'val', spec['reference']['complete_validation_windows'])
    for s in prior['states']:
        w = train.dataset.windows[train.indices[s['view_index']]]
        if list(w['episode_key']) != s['episode_key'] or int(w['step']) != s['step']:
            raise ValueError('prior state ID mismatch')
    validation_states = select_states(val, spec['states']['validation_count'])
    ids = {'train': prior['states'], 'val': validation_states,
           'selection': spec['states']['selection'], 'diffusion_seeds': spec['states']['diffusion_seeds'],
           'heldout_ids': [], 'source_sha256': hashes}
    atomic(out/'diagnostic-state-ids.json', ids)
    atomic(out/'source-provenance.json', {'input_sha256': hashes, 'counts': {'train':len(train),'val':len(val)},
        'forbidden_split':'test', 'frozen_before_results': True, 'scheduler_api': {'step_generator': True,
        'step_stochasticity_parameter': False, 'step_variance_parameter': False,
        'allowed_diagnostic_axis':'initial_noise_scale'}, 'previous_train_ood_p95': thresholds['train_ood_nearest_l2_per_dimension_p95']})
    frozen = {'coverage_config':config_hash, **hashes,
              'diagnostic_state_ids':digest(out/'diagnostic-state-ids.json'),
              'source_provenance':digest(out/'source-provenance.json')}
    atomic(out/'artifact-hashes.json', frozen)
    atomic(out/'status.json', {'stage':'frozen','state':'complete','time_unix':time.time(),
          'config_sha256':config_hash, 'formal_phase_c_allowed':False})
    write_marker(out/'freeze.done.json', [out/'diagnostic-state-ids.json',out/'source-provenance.json',out/'artifact-hashes.json'],config_hash)
    print('FROZEN',json.dumps({'states':(len(ids['train']),len(ids['val'])),'config_sha256':config_hash}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--coverage-config',type=Path,required=True)
    p.add_argument('--prior-states',type=Path,required=True)
    p.add_argument('--prior-thresholds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    spec=yaml.safe_load(args.coverage_config.read_text())
    validate_config(spec)
    freeze(args,spec)

if __name__=='__main__': main()
