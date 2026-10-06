"""Read-only analysis of fixed offline-smoke predictions."""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eat_nav_evaluator_offline_audit import OUT, ROOT, MULTI
from visualnav_transformer.eat_nav.evaluator_offline_train import pareto_set


RAW = pathlib.Path("/tmp/eatnav-evaluator-offline-smoke-20261006-r1")
HEADS = ["reward", "direct_relative_reward", "goal_reduction", "endpoint_error",
         "path_progress", "cross_track_p95"]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _candidate_mean(row):
    grouped = defaultdict(list)
    for candidate, prediction, target in zip(row["candidate_ids"], row["prediction"], row["target"]):
        grouped[candidate].append((prediction, target))
    return {candidate: (np.mean([x[0] for x in samples], axis=0),
                        np.mean([x[1] for x in samples], axis=0))
            for candidate, samples in grouped.items()}


def _mean_nonnull(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def distillation_adoption(core, student, *, mean_primary_delta, fold_primary_improvements):
    """Observed Outcome guard: regression gain cannot excuse worse selection."""
    if mean_primary_delta > 0 or sum(v >= 5 for v in fold_primary_improvements.values()) < 2:
        return False
    if student["top1_recall"] < core["top1_recall"] - 1e-9:
        return False
    if student["oracle_reward_regret"] > core["oracle_reward_regret"] + 1e-9:
        return False
    if student["pairwise_accuracy"] < core["pairwise_accuracy"] - 1e-9:
        return False
    return True


def analyze():
    status = json.loads((RAW / "status.json").read_text())
    if status["stage"] != "complete" or status["completed"] != status["total"]:
        raise ValueError("smoke training incomplete")
    single = json.loads((OUT / "single-batch-overfit-results.json").read_text())
    seven = json.loads((OUT / "seven-group-overfit-results.json").read_text())
    loso = json.loads((OUT / "loso-diagnostic-results.json").read_text())
    predictions = json.loads((RAW / "predictions.json").read_text())
    pair_records = json.loads((MULTI / "pairwise-labels.json").read_text())["candidate_pairs"]
    if len(loso) != 7 * 8 * 3 or len(predictions) != len(loso):
        raise ValueError("LOSO count mismatch")
    keyed = {(row["held_out"], row["seed"], row["variant"]): row for row in predictions}
    variants = [f"B{i}" for i in range(8)]
    ablation = {}
    multi = []
    for variant in variants:
        subset = [row for row in loso if row["variant"] == variant]
        fold_metric = {}
        for group in sorted(set(row["held_out"] for row in subset)):
            rows = [r["test_metrics"] for r in subset if r["held_out"] == group]
            fold_metric[group] = {head: {metric: _mean_nonnull(
                [r["heads"][head][metric] for r in rows]) for metric in
                ("mae", "rmse", "spearman", "prediction_variance")}
                for head in HEADS}
            fold_metric[group]["pairwise_accuracy"] = _mean_nonnull([r["ranking"]["pairwise_accuracy"] for r in rows])
            fold_metric[group]["oracle_reward_regret"] = _mean_nonnull([r["selection"]["oracle_reward_regret"] for r in rows])
            fold_metric[group]["top1_recall"] = _mean_nonnull([r["selection"]["top1_recall"] for r in rows])
            fold_metric[group]["top3_recall"] = _mean_nonnull([r["selection"]["top3_recall"] for r in rows])
            fold_metric[group]["reward_collapse_rate"] = _mean_nonnull(
                [r["heads"]["reward"]["collapsed"] for r in rows])
        ablation[variant] = {"group_macro": {
            "heads": {head: {metric: _mean_nonnull([m[head][metric] for m in fold_metric.values()])
                             for metric in ("mae", "rmse", "spearman", "prediction_variance")}
                      for head in HEADS},
            "pairwise_accuracy": _mean_nonnull([m["pairwise_accuracy"] for m in fold_metric.values()]),
            "oracle_reward_regret": _mean_nonnull([m["oracle_reward_regret"] for m in fold_metric.values()]),
            "top1_recall": _mean_nonnull([m["top1_recall"] for m in fold_metric.values()]),
            "top3_recall": _mean_nonnull([m["top3_recall"] for m in fold_metric.values()])},
            "folds": fold_metric}
        ablation[variant]["group_macro"]["reward_collapse_rate"] = _mean_nonnull(
            [m["reward_collapse_rate"] for m in fold_metric.values()])
    for row in predictions:
        candidate = _candidate_mean(row)
        pred_scores = {key: [value[0][0], value[0][2]] for key, value in candidate.items()}
        real_scores = {key: [value[1][0], value[1][2]] for key, value in candidate.items()}
        predicted_front = pareto_set(pred_scores)
        real_front = pareto_set(real_scores)
        unique_predicted_scores = len({tuple(np.round(score, 10)) for score in pred_scores.values()})
        reward_best = max(candidate, key=lambda key: candidate[key][1][0])
        goal_best = max(candidate, key=lambda key: candidate[key][1][2])
        selected = max(candidate, key=lambda key: candidate[key][0][0])
        direct = candidate["direct_reference"][1]
        margin = candidate[selected][0][0] - candidate["direct_reference"][0][0]
        non_direct = [key for key in candidate if key != "direct_reference"]
        sign_correct = np.mean([(candidate[key][0][1] >= 0) == (candidate[key][1][1] >= 0)
                                for key in non_direct])
        conflict_cases = [p for p in pair_records if p["state_group_id"] == row["held_out"]
                          and p["reward_goal_conflict"]]
        conflict_reward_correct = []
        conflict_goal_correct = []
        all_reward_correct = []
        all_goal_correct = []
        for pair in (p for p in pair_records if p["state_group_id"] == row["held_out"]):
            a, b = pair["candidate_i"], pair["candidate_j"]
            for head, label, sink in ((0, pair["reward_label"], all_reward_correct),
                                      (2, pair["goal_label"], all_goal_correct)):
                if label in ("i_better", "j_better"):
                    sink.append((candidate[a][0][head] > candidate[b][0][head]) ==
                                (label == "i_better"))
        for pair in conflict_cases:
            a, b = pair["candidate_i"], pair["candidate_j"]
            for head, label, sink in ((0, pair["reward_label"], conflict_reward_correct),
                                      (2, pair["goal_label"], conflict_goal_correct)):
                if label in ("i_better", "j_better"):
                    sink.append((candidate[a][0][head] > candidate[b][0][head]) ==
                                (label == "i_better"))
        multi.append({"held_out": row["held_out"], "seed": row["seed"], "variant": row["variant"],
                      "pareto_set_recall": len(predicted_front & real_front) / len(real_front),
                      "pareto_set_precision": len(predicted_front & real_front) / len(predicted_front),
                      "predicted_front_size": len(predicted_front),
                      "all_candidate_predictions_tied": unique_predicted_scores == 1,
                      "reward_pareto_recall": float(reward_best in predicted_front),
                      "goal_pareto_recall": float(goal_best in predicted_front),
                      "predicted_pareto": sorted(predicted_front), "observed_pareto": sorted(real_front),
                      "direct_relative_sign_accuracy": float(sign_correct),
                      "reward_pair_accuracy": _mean_nonnull(all_reward_correct),
                      "goal_pair_accuracy": _mean_nonnull(all_goal_correct),
                      "reward_goal_conflict_count": len(conflict_cases),
                      "conflict_reward_pair_accuracy": _mean_nonnull(conflict_reward_correct),
                      "conflict_goal_pair_accuracy": _mean_nonnull(conflict_goal_correct),
                      "selected": selected, "predicted_margin_over_direct": float(margin),
                      "selected_reward": float(candidate[selected][1][0]),
                      "selected_goal_reduction": float(candidate[selected][1][2]),
                      "reward_minus_direct": float(candidate[selected][1][0] - direct[0]),
                      "goal_minus_direct": float(candidate[selected][1][2] - direct[2])})
    abstention = {}
    for variant in variants:
        selection = [m for m in multi if m["variant"] == variant]
        abstention[variant] = {}
        for threshold in (0.0, 0.25, 0.5):
            margins = np.asarray([m["predicted_margin_over_direct"] for m in selection])
            cutoff = float(np.quantile(margins, threshold))
            accepted = [m for m in selection if m["predicted_margin_over_direct"] > cutoff]
            abstention[variant][str(threshold)] = {
                "diagnostic_margin_cutoff": cutoff, "coverage": len(accepted) / len(selection),
                "mean_reward_minus_direct": _mean_nonnull([m["reward_minus_direct"] for m in accepted]),
                "mean_goal_minus_direct": _mean_nonnull([m["goal_minus_direct"] for m in accepted])}
    save("ablation-comparison.json", ablation)
    save("multiobjective-evaluation.json", {"runs": multi, "abstention_diagnostic": abstention,
        "warning": "Pareto, selection and abstention are descriptive on seven StateGroups; no online policy approval"})
    # Distillation is assessed against observed labels, never only teacher agreement.
    distill = []
    for group in sorted(set(row["held_out"] for row in predictions)):
        for seed in (123, 456, 789):
            teacher = np.asarray(keyed[(group, seed, "B3")]["prediction"])
            core = np.asarray(keyed[(group, seed, "B5")]["prediction"])
            student = np.asarray(keyed[(group, seed, "B7")]["prediction"])
            target = np.asarray(keyed[(group, seed, "B7")]["target"])
            candidate_ids = keyed[(group, seed, "B7")]["candidate_ids"]
            unique_ids = list(dict.fromkeys(candidate_ids))
            teacher_rank = np.asarray([teacher[[i for i, c in enumerate(candidate_ids) if c == key], 0].mean()
                                       for key in unique_ids])
            student_rank = np.asarray([student[[i for i, c in enumerate(candidate_ids) if c == key], 0].mean()
                                       for key in unique_ids])
            distill.append({"held_out": group, "seed": seed,
                            "teacher_student_mae": float(np.abs(teacher - student).mean()),
                            "student_primary_mae": float(np.abs(student[:, :3] - target[:, :3]).mean()),
                            "core_primary_mae": float(np.abs(core[:, :3] - target[:, :3]).mean()),
                            "observed_primary_delta_student_minus_core": float(
                                np.abs(student[:, :3] - target[:, :3]).mean() -
                                np.abs(core[:, :3] - target[:, :3]).mean()),
                            "reward_pair_ranking_retained_vs_teacher": float(np.mean([
                                np.sign(teacher_rank[i] - teacher_rank[j]) ==
                                np.sign(student_rank[i] - student_rank[j])
                                for i in range(len(unique_ids)) for j in range(i + 1, len(unique_ids))]))})
    b4, b5, b3, b6, b7 = [ablation[key]["group_macro"] for key in ("B4", "B5", "B3", "B6", "B7")]
    def improvements(a, b):
        return {head: sum(ablation[a]["folds"][group][head]["mae"] <
                          ablation[b]["folds"][group][head]["mae"]
                          for group in ablation[a]["folds"])
                for head in HEADS[:3]}
    core_vs_baseline = improvements("B4", "B1")
    two_vs_core = improvements("B5", "B4")
    distill_vs_core = improvements("B7", "B5")
    single_actor = [r for r in single if r["variant"] == "B2"]
    result = {
        "scope": "offline_7_StateGroup_smoke_only",
        "actor22_distillation_supported_for_smoke": not any(r["feature_collision"] for r in single_actor)
            and all(r["primary_mae"] < 0.1 for r in single_actor),
        "actor22_not_sufficient_for_path_selection": True,
        "actor22_reason": "pre-Path 22D input is identical for every candidate within each StateGroup; group-level mean fit does not establish candidate-specific Outcome distillation",
        "portable_core_learns_signal": sum(v >= 5 for v in core_vs_baseline.values()) >= 2,
        "portable_2s_adds_signal": sum(v >= 5 for v in two_vs_core.values()) >= 2,
        "portable_2s_not_required": sum(v >= 5 for v in two_vs_core.values()) < 2,
        "simulator_privileged_only": all(v < 5 for v in core_vs_baseline.values()) and
            (improvements("B3", "B4")["reward"] >= 5 or improvements("B6", "B4")["reward"] >= 5),
        "distillation_supported": distillation_adoption(b5, b7,
            mean_primary_delta=_mean_nonnull([d["observed_primary_delta_student_minus_core"] for d in distill]),
            fold_primary_improvements=distill_vs_core),
        "insufficient_state_group_support": True,
        "fold_improvements": {"B4_over_B1": core_vs_baseline, "B5_over_B4": two_vs_core,
                              "B7_over_B5": distill_vs_core},
        "distillation": distill,
        "physical_deployment_blocked": "flipper/IMU/DEM parity and arm stowed reference remain unverified",
        "formal_evaluator_training_approved": False,
        "online_selection_approved": False,
        "nomad_retraining_approved": False,
        "next_evidence": "additional independent StateGroups and physical logging parity before model selection",
    }
    save("final-decision.json", result)
    # Compact figures for review.
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for j, head in enumerate(HEADS):
        ax = axs.flat[j]
        for variant, color in (("B3", "tab:blue"), ("B4", "tab:green"), ("B5", "tab:orange"), ("B7", "tab:red")):
            rows = [r for r in predictions if r["variant"] == variant]
            truth = np.concatenate([np.asarray(r["target"])[:, j] for r in rows])
            guess = np.concatenate([np.asarray(r["prediction"])[:, j] for r in rows])
            ax.scatter(truth, guess, s=6, alpha=.3, label=variant, color=color)
        ax.set_title(head)
        ax.set_xlabel("observed")
        ax.set_ylabel("LOSO prediction")
    axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig-head-prediction-target.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(8)
    for j, head in enumerate(HEADS[:3]):
        ax.bar(x + (j-1)*.24, [ablation[v]["group_macro"]["heads"][head]["mae"] for v in variants],
               width=.24, label=head)
    ax.set_xticks(x, variants)
    ax.set_ylabel("LOSO StateGroup macro MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig-input-ablation.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for j, head in enumerate(HEADS[:3]):
        for variant in ("B4", "B5", "B7"):
            axs[j].plot(sorted(ablation[variant]["folds"]),
                        [ablation[variant]["folds"][g][head]["mae"] for g in sorted(ablation[variant]["folds"])],
                        marker="o", label=variant)
        axs[j].set_title(head)
        axs[j].tick_params(axis="x", rotation=80, labelsize=7)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig-fold-metrics.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([d["core_primary_mae"] for d in distill],
               [d["student_primary_mae"] for d in distill], c="tab:purple")
    lim = max([d["core_primary_mae"] for d in distill] + [d["student_primary_mae"] for d in distill])
    ax.plot([0, lim], [0, lim], "k--")
    ax.set_xlabel("B5 observed primary MAE")
    ax.set_ylabel("B7 observed primary MAE")
    fig.tight_layout()
    fig.savefig(OUT / "fig-teacher-student.png", dpi=150)
    plt.close(fig)
    paths = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "artifact-sha-ledger.json")
    save("artifact-sha-ledger.json", {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in paths})
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
