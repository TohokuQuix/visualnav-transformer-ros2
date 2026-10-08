"""Outcome-blind, frozen P1 terrain-role classification and selection."""

from __future__ import annotations

import itertools


ROLES = ('R-differential', 'R-transition-common', 'R-graded-common', 'R-flat-control')
SLOTS = (('first', ROLES[0]), ('second', ROLES[0]),
         (None, ROLES[1]), (None, ROLES[2]), (None, ROLES[3]))


def classify_role(row: dict) -> dict:
    paths = row['paths']
    if len(paths) != 5:
        raise ValueError('exactly five frozen primitive Paths required')
    safe = (row['physical_eligible'] and row['hard_valid'] and
            row['follower_compatible'])
    known = all(p['known'] for p in paths)
    classes = [p['class_proxy'] for p in paths]
    transition = [p for p in paths if p['class_proxy'] == 'transition'
                  and p['known'] and p['onset_m'] is not None
                  and 0 <= p['onset_m'] <= .6]
    flags = {
        'R-differential': bool(safe and known and row['pair_count'] > 0 and
                               row['max_pair_rms_m'] >= .01),
        'R-transition-common': bool(safe and known and transition),
        'R-graded-common': bool(safe and known and all(c == 'graded' for c in classes)),
        'R-flat-control': bool(safe and known and all(c == 'flat-only' for c in classes)
                               and row['pair_count'] == 0),
    }
    primary = next((role for role in ROLES if flags[role]), None)
    return {**row, 'flags': flags, 'primary_role': primary,
            'transition_path_count': len(transition),
            'transition_onset_range_m': [min(p['onset_m'] for p in transition),
                                         max(p['onset_m'] for p in transition)] if transition else None,
            'graded_path_count': sum(c == 'graded' for c in classes),
            'known_min_fraction': min(p['known_fraction'] for p in paths)}


def select_five(rows: list[dict]) -> dict:
    options = [[r for r in rows if r['primary_role'] == role and
                (seed is None or r['seed_label'] == seed)] for seed, role in SLOTS]
    missing = [role for (_, role), choices in zip(SLOTS, options) if not choices]
    combinations = []
    if not missing:
        for group in itertools.product(*options):
            if len({r['anchor_id'] for r in group}) != 5:
                continue
            identities = {(r['seed_label'], r['source_family']) for r in group}
            if len(identities) < 4:
                continue
            combinations.append(group)
    if not combinations:
        return {'pass': False, 'selected': [], 'missing_roles': missing,
                'reason': 'role_aware_terrain_support_insufficient',
                'slot_option_counts': [len(x) for x in options],
                'distinct_identity_gate': 4}
    def score(group):
        return (-min(r['known_min_fraction'] for r in group),
                -sum(r['max_pair_rms_m'] for r in group[:2]),
                -len({(r['seed_label'], r['source_family']) for r in group}),
                tuple(r['selection_rank'] for r in group),
                tuple(r['anchor_id'] for r in group))
    chosen = min(combinations, key=score)
    return {'pass': True, 'selected': [r['anchor_id'] for r in chosen],
            'missing_roles': [], 'reason': None,
            'slot_option_counts': [len(x) for x in options],
            'distinct_identity_gate': 4}
