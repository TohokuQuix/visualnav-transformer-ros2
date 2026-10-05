"""Small offline figures for the sealed multi-head label review."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-freeze"


def finish(fig, name):
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    with gzip.open(OUT / "multihead-labels.jsonl.gz", "rt") as source:
        rows = [json.loads(line) for line in source]
    pairs = json.loads((OUT / "pairwise-labels.json").read_text())["candidate_pairs"]
    support = json.loads((OUT / "data-sufficiency-review.json").read_text())["heads"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (field, title, unit) in zip(axes.flat, (
        ("reward", "Observed 0–2 s discounted reward", "reward units"),
        ("direct_relative_reward", "Relative to direct repeat mean", "reward units"),
        ("goal", "Goal-distance reduction", "m"),
        ("endpoint", "Endpoint error at 2 s", "m"))):
        values = [r[field] for r in rows]
        ax.hist(values, bins=12, color="#4a7698", edgecolor="white")
        ax.set_title(title); ax.set_xlabel(unit); ax.set_ylabel("trace count")
    fig.tight_layout(); finish(fig, "fig-01-observed-label-distributions.svg")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, field, title in zip(axes, ("reward_label", "goal_label", "endpoint_label"),
                                 ("Reward ranking", "Goal ranking", "Endpoint ranking")):
        counts = Counter(pair[field] for pair in pairs)
        names = ("i_better", "j_better", "tie_within_duplicate_noise", "invalid_or_censored")
        ax.bar(range(4), [counts.get(name, 0) for name in names], color=("#497c8b", "#aa695c", "#b6a55c", "#9a9a9a"))
        ax.set_xticks(range(4), ("i", "j", "tie", "invalid")); ax.set_title(title)
        ax.set_ylim(0, len(pairs)); ax.set_ylabel("descriptive pairs")
    fig.tight_layout(); finish(fig, "fig-02-pairwise-labels.svg")

    by_group = Counter(pair["state_group_id"] for pair in pairs if pair["reward_goal_conflict"])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    groups = sorted({row["state_group_id"] for row in rows})
    ax.bar(range(len(groups)), [by_group[group] for group in groups], color="#9a5d79")
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    ax.set_ylim(0, 21); ax.set_ylabel("reward/goal conflict pairs out of 21")
    ax.set_title("Within-StateGroup conflicts; pairs are not independent samples")
    fig.tight_layout(); finish(fig, "fig-03-reward-goal-conflicts.svg")

    fig, ax = plt.subplots(figsize=(8, 3))
    names = ["reward", "goal", "endpoint", "encounter", "V hold", "V canonical", "literal zero V"]
    keys = ["discounted_reward_2s", "goal_distance_reduction_2s", "endpoint_error_2s",
            "endpoint_encounter_2s", "terminal_value_hold", "terminal_value_canonical", "terminal_value_literal_zero"]
    valid = [support[key]["valid"] for key in keys]
    ax.barh(names, valid, color=["#8e704c" if key == "endpoint_encounter_2s" else "#55847a" for key in keys])
    ax.set_xlim(0, 63); ax.set_xlabel("valid labels / 63; encounter class support is insufficient")
    ax.set_title("Label coverage; no missing value is plotted as zero")
    fig.tight_layout(); finish(fig, "fig-04-head-support.svg")
    print("4 SVG figures written")


if __name__ == "__main__":
    main()
