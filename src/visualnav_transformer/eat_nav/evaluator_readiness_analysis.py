"""State-macro oracle and relative-label checks for frozen counterfactual pilot."""
from __future__ import annotations

import math


def oracle_curve(rows, ks=(1, 2, 4, 8), *, epsilon=.01, metric='delta_v'):
    """Compare NoMaD prefixes against nominal + valid NoMaD + valid bank."""
    valid = [r for r in rows if r['constraint_valid'] and r.get(metric) is not None
             and math.isfinite(float(r[metric]))]
    if not valid:
        return {'status': 'no_valid_outcome', 'k': []}
    nomad = sorted((r for r in rows if r['kind'] == 'nomad'),
                   key=lambda r: int(r['candidate_id'].split('-')[-1]))
    nominal = next((r for r in valid if r['kind'] == 'nominal'), None)
    best = max(valid, key=lambda r: float(r[metric]))
    best_value = float(best[metric])
    result = {'status': 'observed', 'feasible_union_best':
              {'candidate_id': best['candidate_id'], metric: best_value},
              'nominal': float(nominal[metric]) if nominal else None, 'k': []}
    for k in ks:
        if k < 1 or k > len(nomad):
            raise ValueError('K exceeds frozen NoMaD set')
        chosen = ([nominal] if nominal else []) + [r for r in nomad[:k] if r in valid]
        winner = max(chosen, key=lambda r: float(r[metric])) if chosen else None
        value = float(winner[metric]) if winner else None
        result['k'].append({'k': k, 'candidate_id': winner['candidate_id'] if winner else None,
                            'generator': winner['kind'] if winner else None,
                            'best_outcome': value,
                            'nominal_gain': value - float(nominal[metric]) if winner and nominal else None,
                            'regret': max(0., best_value-value) if winner else None,
                            'recall': bool(best_value-value <= epsilon) if winner else None})
    return result


def readiness(states, *, min_states=4, delta_v_min=.05, secondary_min=.05):
    delta = [s['state_id'] for s in states if s.get('delta_v_range') is not None
             and s['delta_v_range'] >= delta_v_min]
    secondary = [s['state_id'] for s in states if
                 max(s.get('advantage_range') or 0., s.get('drawdown_range') or 0.) >= secondary_min]
    rank = [s['state_id'] for s in states if (s.get('nomad_distinct_ranks') or 0) >= 2]
    ready = len(delta) >= min_states and len(secondary) >= min_states and len(rank) >= min_states
    return {'ready': ready, 'delta_v_support_states': delta,
            'advantage_or_drawdown_support_states': secondary,
            'nomad_multiple_rank_states': rank, 'minimum_states': min_states}
