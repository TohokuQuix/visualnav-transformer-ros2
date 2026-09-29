"""Outcome-blind, full-prefix-budgeted state selection for coverage pilot."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


_ALLOWED = ("split", "complete", "natural_history", "terrain_type", "episode_key",
            "view_index", "step", "local_step", "goal", "terrain_seed", "dem_sha256")


def _distribution(values):
    values = np.asarray(values, dtype=np.int64)
    return {
        "count": int(len(values)),
        "min": int(values.min()) if len(values) else None,
        "p50": float(np.median(values)) if len(values) else None,
        "p90": float(np.quantile(values, .9)) if len(values) else None,
        "max": int(values.max()) if len(values) else None,
    }


def _rank(row, seed):
    identity = [seed, row["terrain_type"], row["episode_key"], row["view_index"]]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def freeze_pilot_selection(rows, *, terrains, seed, min_goal_m,
                           max_full_prefix_steps, approved_budget_sec=None,
                           run_count_per_state=17, cost_intercept_sec=0.,
                           cost_slope_sec=0.):
    """Select at most one train state per terrain using only input fields.

    ``step`` is the zero-based global anchor step; replay includes step itself.
    Rows may carry outcomes, but those fields are neither read nor serialized.
    """
    if min_goal_m < .30 or max_full_prefix_steps < 0 or run_count_per_state <= 0:
        raise ValueError("invalid frozen selection limits")
    if approved_budget_sec is not None and approved_budget_sec <= 0:
        raise ValueError("approved budget must be positive")
    terrain_list = sorted(set(terrains))
    if not terrain_list:
        raise ValueError("no terrains")
    support = {t: {"complete_train": 0, "eligible": 0, "goal_rejected": 0,
                   "prefix_rejected": 0} for t in terrain_list}
    eligible = {t: [] for t in terrain_list}
    seen = set()
    all_steps = []
    eligible_steps = []
    for source in rows:
        row = {name: source[name] for name in _ALLOWED if name in source}
        terrain = row["terrain_type"]
        if row["split"] != "train" or not row["complete"] or not row["natural_history"]:
            raise ValueError("only complete train windows with natural history are allowed")
        if terrain not in support:
            raise ValueError("undeclared terrain")
        identity = (tuple(row["episode_key"]), int(row["view_index"]))
        if identity in seen:
            raise ValueError("duplicate window identity")
        seen.add(identity)
        step = int(row["step"])
        if step < 0:
            raise ValueError("negative full-prefix step")
        full_prefix_steps = step + 1
        all_steps.append(full_prefix_steps)
        support[terrain]["complete_train"] += 1
        goal = math.hypot(*map(float, row["goal"]))
        if not math.isfinite(goal):
            raise ValueError("nonfinite goal")
        if goal < min_goal_m:
            support[terrain]["goal_rejected"] += 1
            continue
        if full_prefix_steps > max_full_prefix_steps:
            support[terrain]["prefix_rejected"] += 1
            continue
        row["goal_distance_m"] = goal
        row["full_prefix_steps"] = full_prefix_steps
        row["rank_sha256"] = _rank(row, seed)
        eligible[terrain].append(row)
        eligible_steps.append(full_prefix_steps)
        support[terrain]["eligible"] += 1
    selected = []
    used_episodes = set()
    na = {}
    for terrain in terrain_list:
        for row in sorted(eligible[terrain], key=lambda x: (x["rank_sha256"], x["view_index"])):
            episode = tuple(row["episode_key"])
            if episode not in used_episodes:
                selected.append(dict(row, state_id=f"coverage-{len(selected):03d}"))
                used_episodes.add(episode)
                break
        else:
            na[terrain] = "NA / no state within full-prefix budget"
    predicted = sum(run_count_per_state * (cost_intercept_sec +
                    cost_slope_sec * int(row["full_prefix_steps"])) for row in selected)
    if approved_budget_sec is not None and predicted > approved_budget_sec:
        raise ValueError("predicted pilot runtime exceeds approved budget")
    return {
        "seed": int(seed), "min_goal_m": float(min_goal_m),
        "max_full_prefix_steps": int(max_full_prefix_steps),
        "approved_budget_sec": approved_budget_sec,
        "predicted_pilot_sec": predicted,
        "run_count_per_state": run_count_per_state,
        "selected": selected, "na_terrains": na, "support": support,
        "all_prefix_steps": _distribution(all_steps),
        "eligible_prefix_steps": _distribution(eligible_steps),
        "selected_prefix_steps": _distribution([r["full_prefix_steps"] for r in selected]),
        "selection_fields": list(_ALLOWED), "outcome_fields_used": [],
        "episode_cap": 1,
    }


def validate_frozen_selection(frozen, *, expected_source_sha):
    """Refuse a changed or outcome-contaminated pilot state file."""
    if frozen.get('source_sha256') != expected_source_sha:
        raise ValueError('selection source SHA mismatch')
    if frozen.get('candidate_outcomes_used') or frozen.get('heldout_used'):
        raise ValueError('outcome or held-out leakage in selection')
    if frozen.get('complete_train_windows') != 49598:
        raise ValueError('complete train count mismatch')
    selected = frozen.get('selected', [])
    if len(selected) > 7:
        raise ValueError('too many pilot states')
    if len({r['terrain_type'] for r in selected}) != len(selected):
        raise ValueError('duplicate terrain')
    if len({tuple(r['episode_key']) for r in selected}) != len(selected):
        raise ValueError('duplicate episode')
    for row in selected:
        if row['split'] != 'train' or row['full_prefix_steps'] != row['step'] + 1:
            raise ValueError('invalid train full-prefix state')
        if row.get('complete') is False or row.get('natural_history') is False:
            raise ValueError('incomplete state')
        if row['full_prefix_steps'] > frozen.get('max_full_prefix_steps', float('inf')):
            raise ValueError('state exceeds frozen full-prefix cap')
    return True
