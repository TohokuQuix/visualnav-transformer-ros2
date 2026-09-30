"""Secondary frozen Phase B GT(P) rescore after all oracle outcomes exist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from eat_nav_phase_c_coverage_evaluator import FrozenEvaluator, SEEDS
from eat_nav_feasible_transfer_evaluator import rank_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_once(path, data):
    content = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError('frozen score result differs: ' + str(path))
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(content)
    tmp.replace(path)


def evaluate_subset(rows, predicted, target, include):
    indices = [i for i, row in enumerate(rows) if include(row) and
               row[target]['status'] == 'observed' and row[target].get('delta_v') is not None]
    if len(indices) < 2:
        return {'status': 'insufficient_support', 'candidate_count': len(indices)}
    truth = np.asarray([rows[i][target]['delta_v'] for i in indices], dtype=float)
    score = np.asarray([predicted[i] for i in indices], dtype=float)
    ids = [rows[i]['candidate_id'] for i in indices]
    return {'status': 'observed', 'candidate_count': len(indices),
            **rank_metrics(score, truth, ids, epsilon=.01)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--outcomes', type=Path, required=True)
    parser.add_argument('--raw-complete-status', type=Path, required=True)
    parser.add_argument('--reference-config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    status = json.loads(a.raw_complete_status.read_text())
    if status['stage'] != 'RAW_COMPLETE' or status['completed_rollouts'] != 112:
        raise ValueError('score only after all 112 outcomes complete')
    selection = json.loads(a.selection.read_text())['selected']
    outcomes = json.loads(a.outcomes.read_text())
    if len(selection) != 7 or len(outcomes) != 98:
        raise ValueError('frozen seven-state or outcome count changed')
    cfg = load_eat_config(a.reference_config)
    view = CompleteOnlyView(EATDataset(a.manifest, 'train', cfg, split_seed=123), 8)
    if len(view) != 49598:
        raise ValueError('train split mismatch')
    import torch
    frozen = FrozenEvaluator('cuda' if torch.cuda.is_available() else 'cpu')
    expected = {'11':'0ca5c9c7e759ac136fc7e6532cdea6071e2903c369b236136846c0eed9832e35',
                '29':'72d76a5f9f3500ff300b78f0fb81a4f8436d6fa2b08b5ba1d5b1371780fff4b2',
                '47':'f49beffca00b9cbe329151ee88a99501bee0756f68ebfb0be4d35a8d4b729b2f'}
    if frozen.checkpoint_sha256 != expected:
        raise ValueError('frozen Evaluator checkpoint SHA mismatch')
    if frozen.scaler_sha256 != '8cf50f7b633f5e659c211da842eb52ba124c0063535e1af4d812d1c8fc541c21':
        raise ValueError('frozen train-only Evaluator scaler SHA mismatch')
    results = []
    for state in selection:
        sid = state['state_id']
        rows = [row for row in outcomes if row['state_id'] == sid]
        if len(rows) != 14:
            raise ValueError('state candidate count changed')
        w = view.dataset.windows[view.indices[state['view_index']]]
        dem = np.asarray(w['dem'], dtype=np.float32)
        goal = np.asarray(w['goal'], dtype=np.float32)
        if (hashlib.sha256(dem.tobytes()).hexdigest() != state['dem_sha256'] or
                not np.allclose(goal, state['goal'], atol=1e-7)):
            raise ValueError('frozen DEM/goal changed')
        path = np.asarray([row['planned_waypoints_base'] for row in rows], dtype=np.float32)
        prediction = frozen.predict(dem, goal, path)
        if prediction.shape != (len(SEEDS), 14, 4) or not np.isfinite(prediction).all():
            raise ValueError('Evaluator output shape/nonfinite')
        mean = prediction.mean(axis=0)[:, 0]
        disagreement = prediction[:, :, 0].std(axis=0)
        strata = {
            'all': lambda r: True,
            'nomad_only': lambda r: r['kind'] == 'nomad',
            'bank_only': lambda r: r['kind'] == 'bank',
            'constraint_valid': lambda r: r['constraint_valid'],
        }
        ranks = {horizon:{name:evaluate_subset(rows, mean, horizon, include)
                 for name, include in strata.items()} for horizon in ('two_second','six_second')}
        results.append({'state_id':sid,'terrain_type':state['terrain_type'],
                        'candidate_ids':[r['candidate_id'] for r in rows],
                        'delta_v_prediction_ensemble':prediction[:,:,0].tolist(),
                        'delta_v_prediction_mean':mean.tolist(),
                        'ensemble_disagreement':disagreement.tolist(),
                        'rank_metrics':ranks,
                        'two_vs_six_target_rank_change':
                          [r['candidate_id'] for r in sorted(rows,key=lambda r:r['two_second'].get('delta_v') if r['two_second'].get('delta_v') is not None else -1e99,reverse=True)] !=
                          [r['candidate_id'] for r in sorted(rows,key=lambda r:r['six_second'].get('delta_v') if r['six_second'].get('delta_v') is not None else -1e99,reverse=True)]})
    result = {'scope':'secondary frozen GT(P) rescore; no candidate selection',
              'state_count':7,'candidate_count_with_nominal':98,
              'model_checkpoint_sha256':frozen.checkpoint_sha256,
              'train_scaler_sha256':frozen.scaler_sha256,
              'selection_sha256':sha(a.selection),'outcomes_sha256':sha(a.outcomes),
              'validation_or_heldout_model_selection':False,'states':results}
    atomic_once(a.output,result)
    print('READINESS_FROZEN_EVALUATOR_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
