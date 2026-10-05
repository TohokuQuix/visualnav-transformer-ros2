"""Pure, StateGroup-scoped label and ranking-loss primitives for offline review."""
from __future__ import annotations

import itertools
import math
import statistics
from collections import defaultdict


def direct_baseline(values):
    values = [float(value) for value in values]
    if len(values) != 3 or not all(map(math.isfinite, values)):
        raise ValueError("exactly three finite direct repeats required")
    return {"mean": statistics.mean(values), "sample_variance": statistics.variance(values),
            "repeat_count": 3}


def duplicate_margin(group_direct_values, numeric_floor):
    if not math.isfinite(numeric_floor) or numeric_floor <= 0:
        raise ValueError("positive numeric floor required")
    differences = []
    for values in group_direct_values:
        direct_baseline(values)
        differences.extend(abs(a - b) for a, b in itertools.combinations(values, 2))
    if not differences:
        raise ValueError("direct-repeat groups required")
    return max(numeric_floor, *differences)


def pair_label(a, b, margin, *, valid=True):
    if not valid or not math.isfinite(a) or not math.isfinite(b):
        return "invalid_or_censored"
    if abs(a - b) <= margin or math.isclose(abs(a-b), margin, rel_tol=1e-12, abs_tol=1e-15):
        return "tie_within_duplicate_noise"
    return "i_better" if a > b else "j_better"


def group_pairs(rows, margins):
    groups = defaultdict(list)
    for row in rows:
        groups[row["state_group_id"]].append(row)
    pairs = []
    for group_id, group_rows in sorted(groups.items()):
        if len({row["candidate_id"] for row in group_rows}) != len(group_rows):
            raise ValueError("one summary per candidate required")
        for first, second in itertools.combinations(sorted(group_rows, key=lambda x: x["candidate_id"]), 2):
            valid = first.get("valid", True) and second.get("valid", True)
            reward = pair_label(first["reward"], second["reward"], margins["reward"], valid=valid)
            goal = pair_label(first["goal"], second["goal"], margins["goal"], valid=valid)
            endpoint = pair_label(-first["endpoint"], -second["endpoint"], margins["endpoint"], valid=valid)
            strict = {"i_better", "j_better"}
            pairs.append({"state_group_id": group_id, "candidate_i": first["candidate_id"],
                          "candidate_j": second["candidate_id"], "reward_label": reward,
                          "goal_label": goal, "endpoint_label": endpoint,
                          "reward_goal_conflict": reward in strict and goal in strict and reward != goal})
    return pairs


def masked_bt_loss(pairs):
    """Mean Bradley-Terry logistic loss; tie and invalid pairs have zero mask."""
    losses = []
    for prediction_i, prediction_j, label in pairs:
        if label not in {"i_better", "j_better"}:
            continue
        signed = (prediction_i - prediction_j) * (1 if label == "i_better" else -1)
        losses.append(math.log1p(math.exp(-abs(signed))) + max(-signed, 0.))
    return statistics.mean(losses) if losses else None


def masked_huber_loss(prediction_target_valid, *, delta):
    """Mean Huber error on valid continuous labels; absent support returns NA."""
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("positive Huber delta required")
    values = []
    for prediction, target, valid in prediction_target_valid:
        if not valid:
            continue
        if not math.isfinite(prediction) or not math.isfinite(target):
            raise ValueError("nonfinite valid regression value")
        error = abs(prediction-target)
        values.append(.5*error*error if error <= delta else delta*(error-.5*delta))
    return statistics.mean(values) if values else None


def make_labels(rows, numeric_floors):
    """Build per-run targets and 21 unique-candidate pairs per StateGroup."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["state_group_id"]].append(row)
    for group_id, group_rows in groups.items():
        by_candidate = defaultdict(list)
        for row in group_rows:
            by_candidate[row["candidate_id"]].append(row)
        if (len(group_rows) != 9 or len(by_candidate) != 7 or
                len(by_candidate["direct_reference"]) != 3 or
                any(len(value) != 1 for key, value in by_candidate.items() if key != "direct_reference")):
            raise ValueError(f"{group_id}: expected 3 direct repeats and six unique candidates")
    metrics = ("reward", "goal", "endpoint")
    margins = {metric: duplicate_margin(
        [[row[metric] for row in group if row["candidate_id"] == "direct_reference"]
         for group in groups.values()], numeric_floors[metric]) for metric in metrics}
    baselines = {}
    labels = []
    summaries = []
    for group_id, group_rows in sorted(groups.items()):
        directs = [row for row in group_rows if row["candidate_id"] == "direct_reference"]
        baseline = {metric: direct_baseline([row[metric] for row in directs]) for metric in metrics}
        baselines[group_id] = baseline
        for row in group_rows:
            relative = {
                "reward": row["reward"] - baseline["reward"]["mean"],
                "goal": row["goal"] - baseline["goal"]["mean"],
                "endpoint": baseline["endpoint"]["mean"] - row["endpoint"],
            }
            preference = {metric: ("candidate_better" if difference > margins[metric] else
                                   "direct_better" if difference < -margins[metric] else
                                   "tie_within_duplicate_noise")
                          for metric, difference in relative.items()}
            labels.append({**row, "direct_relative_reward": row["reward"] - baseline["reward"]["mean"],
                           "direct_relative_goal": row["goal"] - baseline["goal"]["mean"],
                           "direct_relative_endpoint_improvement": baseline["endpoint"]["mean"] - row["endpoint"],
                           "direct_relative_preference": preference})
        for candidate in sorted({row["candidate_id"] for row in group_rows}):
            selected = [row for row in group_rows if row["candidate_id"] == candidate]
            summaries.append({"state_group_id": group_id, "candidate_id": candidate,
                              **{metric: statistics.median(row[metric] for row in selected) for metric in metrics},
                              "valid": all(row.get("valid", True) for row in selected)})
    return {"margins": margins, "direct_baselines": baselines,
            "absolute_and_relative_labels": labels, "candidate_summaries": summaries,
            "candidate_pairs": group_pairs(summaries, margins)}
