"""Group-unit evaluation and figures for the frozen A1–A4 comparison."""
from __future__ import annotations

from collections import defaultdict, Counter
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import yaml

from visualnav_transformer.eat_nav.evaluator_offline_train import pareto_set
from visualnav_transformer.eat_nav.path_corridor import corridor_points, resample_path
from eat_nav_corridor_train import HEAD_KEYS


VARIANTS = ('A1', 'A2', 'A3', 'A4')
COMPARISONS = (('A2', 'A1'), ('A3', 'A1'), ('A3', 'A2'), ('A4', 'A3'))
MODES = ('E-anchor', 'E-goal', 'E-random-group', 'E-terrain')
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'docs/quality-identifiability-gate/path-corridor-architecture/corridor-config.yaml'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def paired_group_bootstrap(rows, metric, *, seed=123, repeats=1000):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            grouped[row['group']].append(float(value))
    values = np.asarray([np.mean(grouped[group]) for group in sorted(grouped)], dtype=float)
    if not len(values):
        return {'group_count': 0, 'effect': None, 'ci95': None, 'interpretation': 'insufficient_support'}
    random = np.random.default_rng(seed)
    choices = random.integers(0, len(values), size=(repeats, len(values)))
    bootstrap = values[choices].mean(axis=1)
    low, high = np.quantile(bootstrap, [.025, .975])
    return {'group_count': int(len(values)), 'effect': float(np.mean(values)),
            'ci95': [float(low), float(high)],
            'interpretation': ('insufficient_support' if len(values) < 3 else
                               'inconclusive' if low <= 0 <= high else
                               'positive' if low > 0 else 'negative')}


def independent_interaction_families(support: dict, records: list[dict]) -> set[str]:
    family = {row['group']: row['family'] for row in records}
    return {family[group] for group, value in support.items()
            if value['cluster_count'] >= 2 and
            value['class'] in ('differential_corridor', 'transition_corridor')}


def validate_fit_coverage(metadata: list[dict], folds: list[dict], config: dict) -> None:
    expected = {(fold['mode'], fold['fold_id'], seed, variant)
                for fold in folds for seed in config['training']['seeds']
                for variant in config['models']['variants']}
    actual = [(row['mode'], row['fold_id'], row['seed'], row['variant']) for row in metadata]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('fit coverage mismatch')


def validate_scaler_parity(metadata: list[dict]) -> None:
    by_fold = defaultdict(set)
    for row in metadata:
        identity = (row['mode'], row['fold_id'])
        by_fold[identity].add((row['input_scaler_sha256'], row['target_scaler_sha256'],
                               row['profile_scaler_sha256'], tuple(row['height_scaler'])))
    if any(len(values) != 1 for values in by_fold.values()):
        raise ValueError('scaler parity mismatch across architectures/seeds')


def selection_metrics(predicted: dict, actual: dict) -> dict:
    pred_front, true_front = pareto_set(predicted), pareto_set(actual)
    order = sorted(predicted, key=lambda candidate: (-predicted[candidate][0], candidate))
    true_order = sorted(actual, key=lambda candidate: (-actual[candidate][0], candidate))
    pred_goal = sorted(predicted, key=lambda candidate: (-predicted[candidate][1], candidate))
    true_goal = sorted(actual, key=lambda candidate: (-actual[candidate][1], candidate))
    selected = order[0]
    vectors = np.asarray(list(predicted.values()), dtype=float)
    return {'selected_candidate': selected, 'top1_recall': float(selected == true_order[0]),
            'top3_recall': float(true_order[0] in order[:3]),
            'oracle_reward_regret': float(actual[true_order[0]][0] - actual[selected][0]),
            'oracle_goal_regret': float(actual[true_goal[0]][1] - actual[pred_goal[0]][1]),
            'selected_reward': float(actual[selected][0]),
            'selected_goal_reduction': float(actual[selected][1]),
            'pareto_recall': len(pred_front & true_front) / len(true_front),
            'pareto_precision': len(pred_front & true_front) / len(pred_front),
            'predicted_pareto_size': len(pred_front), 'true_pareto_size': len(true_front),
            'all_candidates_tied': bool(np.max(np.ptp(vectors, axis=0)) <= 1e-8),
            'selected_pareto_member': float(selected in true_front)}


def _candidate_average(rows, values, indices):
    grouped = defaultdict(list)
    for index in indices:
        grouped[rows[index]['candidate']].append(values[index])
    return {candidate: np.mean(group, axis=0) for candidate, group in grouped.items()}


def _group_metrics(group, indices, rows, predicted, actual, pairs):
    p = _candidate_average(rows, predicted, indices)
    t = _candidate_average(rows, actual, indices)
    if len(p) != 7 or set(p) != set(t):
        raise ValueError('incomplete candidate group')
    metrics = selection_metrics({key: [value[0], value[2]] for key, value in p.items()},
                                {key: [value[0], value[2]] for key, value in t.items()})
    selected = metrics['selected_candidate']
    metrics['selected_direct_relative_reward'] = float(t[selected][1])
    metrics['direct_relative_sign_accuracy'] = float(np.mean([
        (p[key][1] >= 0) == (t[key][1] >= 0) for key in p if key != 'direct_reference']))
    for name, head, key in (('reward_pair_accuracy', 0, 'reward_label'),
                            ('goal_pair_accuracy', 2, 'goal_label')):
        correct = []
        for pair in pairs[group]:
            label = pair[key]
            if label in ('i_better', 'j_better'):
                diff = p[pair['candidate_i']][head] - p[pair['candidate_j']][head]
                correct.append((diff > 0) == (label == 'i_better'))
        metrics[name] = float(np.mean(correct)) if correct else None
    metrics['reward_goal_preference_conflict_rate'] = float(np.mean([
        pair['reward_goal_conflict'] for pair in pairs[group]]))
    for head, key in enumerate(HEAD_KEYS):
        errors = predicted[indices, head] - actual[indices, head]
        metrics[f'{key}_mae'] = float(np.mean(np.abs(errors)))
        metrics[f'{key}_rmse'] = float(np.sqrt(np.mean(errors**2)))
        metric = spearmanr(predicted[indices, head], actual[indices, head]).statistic
        metrics[f'{key}_spearman'] = float(metric) if np.isfinite(metric) else None
        metrics[f'{key}_prediction_variance'] = float(np.var(predicted[indices, head]))
        metrics[f'{key}_target_variance'] = float(np.var(actual[indices, head]))
        metrics[f'{key}_collapse'] = float(np.var(predicted[indices, head]) <=
                                     0.01 * max(np.var(actual[indices, head]), 1e-12))
    return metrics


def _aggregate(rows):
    keys = sorted({key for row in rows for key in row if key not in
                   ('group', 'fold_id', 'mode', 'variant', 'seed', 'support_class', 'terrain',
                    'selected_candidate')})
    result = {'group_count': len({row['group'] for row in rows}), 'fit_group_rows': len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and
                  not isinstance(row[key], str) and np.isfinite(row[key])]
        result[key] = float(np.mean(values)) if values else None
    return result


def _paired(rows, metric, mode, support_class=None):
    selected = [row for row in rows if row['mode'] == mode and
                (support_class is None or row['support_class'] == support_class)]
    by_fit = {(row['fold_id'], row['seed'], row['group'], row['variant']): row for row in selected}
    result = {}
    for left, right in COMPARISONS:
        differences = []
        for fold, seed, group, variant in list(by_fit):
            if variant != left:
                continue
            a = by_fit[(fold, seed, group, left)]
            b = by_fit.get((fold, seed, group, right))
            if b is not None and a.get(metric) is not None and b.get(metric) is not None:
                differences.append({'group': group, 'difference': a[metric] - b[metric]})
        result[f'{left}-{right}'] = paired_group_bootstrap(differences, 'difference')
    return result


def _plot_bar(path, aggregates, metric, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(MODES))
    for index, variant in enumerate(VARIANTS):
        y = [aggregates.get(mode, {}).get(variant, {}).get(metric) for mode in MODES]
        ax.bar(x + (index - 1.5) * .2, [np.nan if value is None else value for value in y], .19, label=variant)
    ax.set_xticks(x, MODES, rotation=18)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_overlays(output, root, records, prepared, support):
    with np.load(root / 'combined-features.npz') as data:
        paths = data['candidate_path']
    full = prepared['full_dem']
    corridor = prepared['corridor']
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row['group']].append(index)
    terrain_reps = {}
    class_reps = {}
    for group in sorted(groups):
        index = groups[group][0]
        terrain_reps.setdefault(records[index]['terrain'], index)
        class_reps.setdefault(support[group]['class'], index)
    def draw(representatives, target, title):
        n = len(representatives)
        fig, axes = plt.subplots(1, n, figsize=(4 * n + 1, 4), squeeze=False)
        displayed = [full[index, -1, 0][full[index, -1, 1] > .5]
                     for index in representatives.values() if index is not None]
        vmin = min(float(np.min(values)) for values in displayed)
        vmax = max(float(np.max(values)) for values in displayed)
        image = None
        for ax, (name, index) in zip(axes[0], representatives.items()):
            if index is None:
                ax.text(.5, .5, 'NA: no StateGroup', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(name)
                ax.set_axis_off()
                continue
            image = ax.imshow(np.ma.masked_where(full[index, -1, 1] < .5, full[index, -1, 0]),
                              origin='lower', extent=[-.95, .95, -.95, .95], cmap='terrain',
                              vmin=vmin, vmax=vmax)
            path = np.concatenate([np.zeros((1, 2)), paths[index]], axis=0)
            ax.plot(path[:, 0], path[:, 1], 'r.-', linewidth=1.5)
            stations, tangents, _ = resample_path(paths[index], 16)
            sampled = corridor_points(stations, tangents, np.linspace(-.4, .4, 7))
            ax.plot(sampled[:, 0, 0], sampled[:, 0, 1], 'c--', linewidth=1)
            ax.plot(sampled[:, -1, 0], sampled[:, -1, 1], 'c--', linewidth=1)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(name + '\nvalid=' + format(float(np.mean(corridor[index, -1, 1])), '.2f'))
            ax.set_xlabel('forward x (m)')
            ax.set_ylabel('left y (m)')
        fig.suptitle(title)
        fig.subplots_adjust(left=.045, right=.93, bottom=.18, top=.82, wspace=.35)
        color_axis = fig.add_axes([.95, .18, .008, .64])
        fig.colorbar(image, cax=color_axis, label='anchor-relative height (m)')
        fig.savefig(target, dpi=150)
        plt.close(fig)
    draw(terrain_reps, output / 'terrain-dem-path-corridor-overlay.png', 'Terrain / Path / corridor support')
    ordered_classes = {name: class_reps.get(name) for name in
                       ('flat_corridor', 'common_rough_corridor', 'differential_corridor',
                        'transition_corridor', 'insufficient_or_unknown')}
    draw(ordered_classes, output / 'support-class-representatives.png', 'Outcome-blind support classes')


def report(root: Path, output: Path) -> dict:
    status = json.loads((output / 'corridor-support-audit.json').read_text())
    if status['status'] != 'pass' or status['features_sha256'] != _sha(output / 'corridor-features.npz'):
        raise ValueError('C1 support/tensor mismatch')
    files = sorted((output / 'fits').glob('*.json'))
    if len(files) != 480:
        raise ValueError(f'incomplete comparison: {len(files)}/480 fits')
    records = json.loads((root / 'record-identity.json').read_text())
    fit_meta = [json.loads(path.read_text()) for path in files]
    if status['config_sha256'] != _sha(CONFIG_PATH):
        raise ValueError('frozen config SHA mismatch at reporting')
    config = yaml.safe_load(CONFIG_PATH.read_text())
    folds = json.loads((root / 'split-folds.json').read_text())['folds']
    validate_fit_coverage(fit_meta, folds, config)
    validate_scaler_parity(fit_meta)
    if any(item['config_sha256'] != status['config_sha256'] for item in fit_meta):
        raise ValueError('fit config SHA mismatch')
    labels = {row['process_id']: row for row in map(json.loads, gzip.open(root / 'combined-labels.jsonl.gz', 'rt'))}
    target = np.asarray([[labels[row['process_id']][key] for key in HEAD_KEYS] for row in records], dtype=float)
    pair_data = json.loads((root / 'expanded-pairwise-labels.json').read_text())
    pairs = defaultdict(list)
    for pair in pair_data['candidate_pairs']:
        pairs[pair['state_group_id']].append(pair)
    rows = []
    calibration = defaultdict(lambda: {'target': [], 'prediction': []})
    for path, meta in zip(files, fit_meta):
        npz = path.with_suffix('.npz')
        if meta['prediction_sha256'] != _sha(npz):
            raise ValueError(f'prediction SHA mismatch: {npz}')
        with np.load(npz) as packed:
            predicted = packed['prediction']
        if predicted.shape != (252, 6) or not np.isfinite(predicted).all():
            raise ValueError('prediction shape/finite mismatch')
        by_group = defaultdict(list)
        for index in meta['test_indices']:
            by_group[records[index]['group']].append(index)
        for group, indices in by_group.items():
            item = _group_metrics(group, indices, records, predicted, target, pairs)
            item.update({'group': group, 'fold_id': meta['fold_id'], 'mode': meta['mode'],
                         'variant': meta['variant'], 'seed': meta['seed'],
                         'terrain': records[indices[0]]['terrain'],
                         'support_class': status['group_support'][group]['class']})
            rows.append(item)
            if meta['mode'] == 'E-anchor':
                for head, key in enumerate(HEAD_KEYS):
                    calibration[(meta['variant'], key)]['target'].extend(target[indices, head].tolist())
                    calibration[(meta['variant'], key)]['prediction'].extend(predicted[indices, head].tolist())
    _write(output / 'per-group-metrics.json', rows)
    fold_seed_rows = []
    fit_groups = defaultdict(list)
    for row in rows:
        fit_groups[(row['mode'], row['fold_id'], row['seed'], row['variant'])].append(row)
    for (mode, fold_id, seed, variant), members in sorted(fit_groups.items()):
        fold_seed_rows.append({'mode': mode, 'fold_id': fold_id, 'seed': seed, 'variant': variant,
                               **_aggregate(members)})
    _write(output / 'per-fold-seed-metrics.json', fold_seed_rows)
    aggregates = {mode: {variant: _aggregate([row for row in rows if row['mode'] == mode and
                                             row['variant'] == variant]) for variant in VARIANTS} for mode in MODES}
    stratified = {}
    support_counts = Counter(row['class'] for row in status['group_support'].values())
    for support_class in ('flat_corridor', 'common_rough_corridor', 'differential_corridor',
                          'transition_corridor', 'insufficient_or_unknown'):
        stratified[support_class] = {'group_count': support_counts[support_class],
            'support': 'sufficient' if support_counts[support_class] >= 3 else 'insufficient_support',
            'modes': {mode: {variant: _aggregate([row for row in rows if row['mode'] == mode and
                      row['variant'] == variant and row['support_class'] == support_class])
                      for variant in VARIANTS} for mode in MODES}}
    paired = {mode: {metric: _paired(rows, metric, mode) for metric in
              ('reward_mae', 'goal_mae', 'oracle_reward_regret', 'oracle_goal_regret',
               'reward_pair_accuracy', 'goal_pair_accuracy', 'top1_recall', 'pareto_precision')}
              for mode in MODES}
    paired_support = {support_class: {mode: {metric: _paired(rows, metric, mode, support_class) for metric in
                      ('reward_mae', 'goal_mae', 'oracle_reward_regret', 'reward_pair_accuracy')}
                      for mode in MODES} for support_class in stratified}
    _write(output / 'aggregate-metrics.json', aggregates)
    _write(output / 'stratified-metrics.json', stratified)
    _write(output / 'paired-bootstrap.json', {'overall': paired, 'by_support': paired_support,
           'bootstrap_unit': 'StateGroup', 'repeats': 1000, 'seed': 123,
           'multiple_comparisons_adjusted': False})
    _write(output / 'training-summary.json', {'fit_count': len(fit_meta),
           'total_fit_cpu_seconds': float(sum(item['elapsed_sec'] for item in fit_meta)),
           'epoch_200_count': sum(item['epochs_run'] == 200 for item in fit_meta),
           'early_stopped_count': sum(item['epochs_run'] < 200 for item in fit_meta),
           'best_at_last_epoch_count': sum(item['best_epoch'] == item['epochs_run'] for item in fit_meta),
           'parameter_counts': json.loads((output / 'architecture-freeze.json').read_text())['parameter_counts']})
    terrain_of_group = {row['group']: row['terrain'] for row in records}
    terrain_support = defaultdict(Counter)
    for group, value in status['group_support'].items():
        terrain_support[terrain_of_group[group]][value['class']] += 1
    _write(output / 'terrain-support-counts.json', {terrain: dict(counts) for terrain, counts in terrain_support.items()})
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    with np.load(output / 'corridor-features.npz') as prepared:
        _plot_overlays(figures, root, records, prepared, status['group_support'])
    fig, axes = plt.subplots(len(HEAD_KEYS), len(VARIANTS), figsize=(14, 20))
    for row_index, key in enumerate(HEAD_KEYS):
        for column_index, variant in enumerate(VARIANTS):
            ax = axes[row_index, column_index]
            data = calibration[(variant, key)]
            actual, predicted = np.asarray(data['target']), np.asarray(data['prediction'])
            ax.scatter(actual, predicted, s=5, alpha=.18)
            limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
            ax.plot(limits, limits, 'k--', linewidth=1)
            ax.set_title(f'{variant}: {key}')
            ax.set_xlabel('target')
            ax.set_ylabel('prediction')
    fig.tight_layout()
    fig.savefig(figures / 'e-anchor-calibration-six-heads.png', dpi=130)
    plt.close(fig)
    _plot_bar(figures / 'split-reward-mae.png', aggregates, 'reward_mae', 'Reward MAE by split')
    _plot_bar(figures / 'split-goal-mae.png', aggregates, 'goal_mae', 'Goal reduction MAE by split')
    for metric, name in (('reward_pair_accuracy', 'pairwise-accuracy'),
                         ('top1_recall', 'top1-recall'),
                         ('oracle_reward_regret', 'oracle-reward-regret'),
                         ('pareto_precision', 'pareto-precision'),
                         ('pareto_recall', 'pareto-recall'),
                         ('predicted_pareto_size', 'pareto-set-size')):
        _plot_bar(figures / (name + '.png'), aggregates, metric, metric)
    _plot_bar(figures / 'all-tied-rate.png', aggregates, 'all_candidates_tied', 'All candidates tied')
    # Group-macro support performance and paired effects are also saved as full numeric JSON.
    fig, ax = plt.subplots(figsize=(9, 4))
    classes = list(stratified)
    for index, variant in enumerate(VARIANTS):
        y = [stratified[item]['modes']['E-anchor'][variant].get('reward_mae') for item in classes]
        ax.bar(np.arange(len(classes)) + (index - 1.5) * .2,
               [np.nan if value is None else value for value in y], .19, label=variant)
    ax.set_xticks(np.arange(len(classes)), [item.replace('_corridor', '') for item in classes], rotation=25)
    ax.set_ylabel('E-anchor reward MAE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'support-stratified-performance.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    effect_names = ('A3-A1', 'A4-A3')
    for index, comparison in enumerate(effect_names):
        effect = paired['E-anchor']['reward_mae'][comparison]
        low, high = effect['ci95'] if effect['ci95'] else (np.nan, np.nan)
        ax.errorbar(index, effect['effect'], yerr=[[effect['effect'] - low], [high - effect['effect']]],
                    fmt='o', capsize=5)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xticks(range(len(effect_names)), effect_names)
    ax.set_ylabel('paired StateGroup reward MAE effect')
    fig.tight_layout()
    fig.savefig(figures / 'paired-effects.png', dpi=150)
    plt.close(fig)
    decisions = _decision(status, paired, paired_support, support_counts)
    distinct_families = independent_interaction_families(status['group_support'], records)
    decisions['distinct_interaction_anchor_family_count'] = len(distinct_families)
    decisions['distinct_interaction_anchor_families'] = sorted(distinct_families)
    if len(distinct_families) < 3:
        decisions['terrain_conditioned_candidate_selection_supported'] = 'insufficient_terrain_interaction_support'
    _write(output / 'final-decision.json', decisions)
    _write_report(output / 'analysis-report.md', status, aggregates, paired, stratified, decisions,
                  json.loads((output / 'training-summary.json').read_text()), terrain_support)
    artifact_sha = {str(path.relative_to(output)): _sha(path) for path in output.rglob('*')
                    if path.is_file() and path.name != 'artifact-sha-ledger.json' and
                    not path.name.endswith('.tmp')}
    _write(output / 'artifact-sha-ledger.json', artifact_sha)
    return decisions


def _decision(status, paired, paired_support, support_counts):
    def signal(comparison):
        reward = paired['E-anchor']['reward_mae'][comparison]
        goal = paired['E-anchor']['goal_mae'][comparison]
        if min(reward['group_count'], goal['group_count']) < 3 or reward['ci95'] is None or goal['ci95'] is None:
            return 'insufficient_support'
        if reward['interpretation'] == goal['interpretation'] == 'negative':
            return 'supported'
        return 'inconclusive'
    a2 = signal('A2-A1')
    a3 = signal('A3-A1')
    a3a2 = signal('A3-A2')
    a4 = signal('A4-A3')
    enough = support_counts['differential_corridor'] >= 3 and support_counts['transition_corridor'] >= 3
    transition = paired_support['transition_corridor']['E-anchor']
    transition_regret = transition['oracle_reward_regret']['A3-A1']
    transition_pair = transition['reward_pair_accuracy']['A3-A1']
    selection_supported = bool(transition_regret['group_count'] >= 3 and
                               transition_regret['interpretation'] == 'negative' and
                               transition_pair['interpretation'] == 'positive')
    distinct_interaction_groups = sum(value['cluster_count'] >= 2 and
                                      value['class'] in ('differential_corridor', 'transition_corridor')
                                      for value in status['group_support'].values())
    selection_decision = ('insufficient_terrain_interaction_support' if distinct_interaction_groups < 3
                          else 'supported_on_transition_only' if selection_supported else 'inconclusive')
    return {'corridor_support_sufficient_for_architecture_comparison': enough,
            'a2_profile_adds_heldout_signal': a2,
            'a3_corridor_cnn_adds_heldout_signal': a3,
            'a3_outperforms_a2': a3a2,
            'a4_full_dem_adds_signal_beyond_corridor': a4,
            'terrain_conditioned_candidate_selection_supported': selection_decision,
            'distinct_interaction_group_count': distinct_interaction_groups,
            'architecture_ready_for_larger_training': bool(enough and a3 == 'supported'),
            'additional_state_group_collection_needed': not enough,
            'scope_note': 'E-goal shared-anchor and E-terrain diagnostic-only; unadjusted multiple comparisons.'}


def _write_report(path, status, aggregates, paired, stratified, decisions, training, terrain_support):
    lines = ['# Path-corridor architecture offline comparison', '',
             'All 28 StateGroups and 252 saved candidate traces were retained. No rollout or label change occurred.', '',
             '## Outcome-blind terrain support', '',
             f"Frozen config SHA: `{status['config_sha256']}`. Classes: `{status['class_counts']}`.",
             'Layers with fewer than three StateGroups are insufficient support for performance claims.', '',
             f"Parameter counts: `{training['parameter_counts']}`. Completed fits: {training['fit_count']}; "
             f"early stops: {training['early_stopped_count']}; best at last epoch: {training['best_at_last_epoch_count']}.",
             'If training loss is still improving at epoch 200, this is a training-limit caveat and does not justify changing this frozen comparison.', '',
             '| Terrain | flat | common rough | differential | transition | insufficient/unknown |',
             '|---|---:|---:|---:|---:|---:|']
    for terrain, counts in sorted(terrain_support.items()):
        lines.append(f"| {terrain} | {counts['flat_corridor']} | {counts['common_rough_corridor']} | "
                     f"{counts['differential_corridor']} | {counts['transition_corridor']} | "
                     f"{counts['insufficient_or_unknown']} |")
    lines += ['', '## Held-out results', '',
             '| Split | Model | reward MAE | goal MAE | reward pair accuracy | top-1 | reward regret | Pareto precision |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        for variant in VARIANTS:
            row = aggregates[mode][variant]
            fmt = lambda key: 'NA' if row.get(key) is None else f"{row[key]:.4f}"
            lines.append(f"| {mode} | {variant} | {fmt('reward_mae')} | {fmt('goal_mae')} | "
                         f"{fmt('reward_pair_accuracy')} | {fmt('top1_recall')} | "
                         f"{fmt('oracle_reward_regret')} | {fmt('pareto_precision')} |")
    lines += ['', '## Paired StateGroup confidence intervals', '',
              'Effects are model minus comparator; negative MAE/regret and positive accuracy are improvements. '
              'Intervals are 95% group bootstrap intervals with 1,000 resamples. They are not multiple-comparison adjusted.', '',
              '| E-anchor metric | Comparison | Effect | 95% CI | Interpretation |',
              '|---|---|---:|---|---|']
    for metric in ('reward_mae', 'goal_mae', 'oracle_reward_regret', 'reward_pair_accuracy'):
        for comparison, value in paired['E-anchor'][metric].items():
            lines.append(f"| {metric} | {comparison} | {value['effect']:.4f} | "
                         f"[{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] | {value['interpretation']} |")
    a3_reward = paired['E-anchor']['reward_mae']['A3-A1']
    a3_pair = paired['E-anchor']['reward_pair_accuracy']['A3-A1']
    a3_regret = paired['E-anchor']['oracle_reward_regret']['A3-A1']
    lines += ['', '## Interpretation', '',
              f"A3−A1 reward MAE is {a3_reward['effect']:+.4f} with CI {a3_reward['ci95']}; this does not establish a regression gain.",
              f"A3−A1 reward pair accuracy is {a3_pair['effect']:+.4f} with CI {a3_pair['ci95']}, "
              f"and oracle reward regret is {a3_regret['effect']:+.4f} with CI {a3_regret['ci95']}. "
              'These are adverse candidate-selection effects in E-anchor under unadjusted CIs; they do not justify an architecture change on this sparse terrain-interaction support.',
              f"Only {decisions['distinct_interaction_anchor_family_count']} independent source anchor family has multiple candidate corridor clusters. "
              'The current data cannot establish terrain-conditioned selection generalization.', '']
    lines += ['', '## Decision', '']
    for key, value in decisions.items():
        lines.append(f'- **{key}:** {value}')
    lines += ['', 'The A1 flatten branch uses the same anchor-aligned DEM as A4; this differs from historical B4 preprocessing.',
              'The common branch retains the historical 64-point path DEM profile in all four models, so this isolates the added representation rather than all Path-conditioned terrain information.',
              'E-goal shares anchors; E-terrain is diagnostic-only. Candidate pairs are descriptive and never bootstrap units.',
              'A constant predictor can have Pareto recall 1 by marking all candidates Pareto; precision, set size and all-tied rate must accompany recall.', '']
    Path(path).write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/tmp/eatnav-evaluator-independent-expansion-20261006-v7'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/eatnav-path-corridor-architecture-20261007'))
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.output), indent=2))


if __name__ == '__main__':
    main()
