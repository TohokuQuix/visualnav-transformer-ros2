"""Seven-group offline smoke; no ROS, Isaac or checkpoint inference."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

from eat_nav_evaluator_offline_audit import OUT, ROOT, MULTI, run as audit_run
from visualnav_transformer.eat_nav.evaluator_offline_train import fit_scaler, loso_splits


CONFIG = yaml.safe_load((OUT / "training-smoke-config.yaml").read_text())
OUTPUT = pathlib.Path("/tmp/eatnav-evaluator-offline-smoke-20261006-r1")
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
HEADS = ["reward", "direct_relative_reward", "goal_reduction", "endpoint_error",
         "path_progress", "cross_track_p95"]
TARGET_KEYS = ["reward", "direct_relative_reward", "goal", "endpoint",
               "path_progress_2s", "cross_track_p95_2s"]
VARIANTS = CONFIG["input_variants"]
FEATURES = {
    "B0": ["constant"],
    "B1": ["relative_goal", "candidate_path", "path_geometry", "path_dem_profile", "path_coverage"],
    "B2": ["simulator_actor22_prepath"],
    "B3": ["simulator_actor22_prepath", "dem_history", "relative_goal", "candidate_path",
           "path_geometry", "path_dem_profile", "path_coverage"],
    "B4": ["dem_history", "relative_goal", "candidate_path", "path_geometry",
           "path_dem_profile", "path_coverage", "body_angular_velocity", "projected_gravity",
           "flipper_fold_mode", "flipper_position", "availability_mask", "stale_mask",
           "arm_applicability_mask"],
    "B5": ["dem_history", "relative_goal", "candidate_path", "path_geometry",
           "path_dem_profile", "path_coverage", "body_angular_velocity", "projected_gravity",
           "flipper_fold_mode", "flipper_position", "availability_mask", "stale_mask",
           "arm_applicability_mask", "right_sprocket_angular_state_scaled",
           "left_sprocket_angular_state_scaled"],
    "B6": ["dem_history", "relative_goal", "candidate_path", "path_geometry",
           "path_dem_profile", "path_coverage", "body_angular_velocity", "projected_gravity",
           "flipper_fold_mode", "flipper_position", "availability_mask", "stale_mask",
           "arm_applicability_mask", "simulator_actor_track_aggregate_24j"],
}
FEATURES["B7"] = FEATURES["B5"]
MASKS = {"availability_mask", "stale_mask", "arm_applicability_mask"}


def write_json(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def raw_features(rows, variant):
    blocks = []
    raw_mask = []
    for name in FEATURES[variant]:
        if name == "constant":
            block = np.ones((len(rows), 1), dtype=np.float32)
        else:
            block = np.stack([np.asarray(row[name], dtype=np.float32).ravel() for row in rows])
        blocks.append(block)
        raw_mask.extend([name in MASKS or name == "constant"] * block.shape[1])
    return np.concatenate(blocks, axis=1), np.asarray(raw_mask, dtype=bool)


def make_model(dim, seed):
    torch.manual_seed(int(seed))
    width = CONFIG["network"]["hidden_width"]
    return torch.nn.Sequential(torch.nn.Linear(dim, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, len(HEADS)))


def pair_loss(pred, indices, groups, candidates, pair_labels, *, weights):
    terms = []
    for group in sorted(set(groups[i] for i in indices)):
        pos = [j for j, i in enumerate(indices) if groups[i] == group]
        by_candidate = {}
        for j in pos:
            by_candidate.setdefault(candidates[indices[j]], []).append(j)
        group_terms = {0: [], 2: []}
        for pair in pair_labels[group]:
            left, right = pair["candidate_i"], pair["candidate_j"]
            if left not in by_candidate or right not in by_candidate:
                continue
            for key, head in (("reward_label", 0), ("goal_label", 2)):
                label = pair[key]
                if label not in ("i_better", "j_better"):
                    continue
                a = pred[by_candidate[left], head].mean()
                b = pred[by_candidate[right], head].mean()
                sign = 1.0 if label == "i_better" else -1.0
                group_terms[head].append(F.softplus(-sign * (a - b)))
        weighted = [float(weight) * torch.stack(group_terms[head]).mean()
                    for head, weight in zip((0, 2), weights) if group_terms[head]]
        if weighted:
            terms.append(torch.stack(weighted).sum())
    return torch.stack(terms).mean() if terms else pred.sum() * 0.0


def train_variant(rows, targets, groups, candidates, pair_labels, indices, variant, seed,
                  max_epochs, teacher_train_predictions=None):
    full, raw_mask = raw_features(rows, variant)
    train_groups = [groups[i] for i in indices]
    xm, xs = fit_scaler(full[indices], train_groups)
    xm[raw_mask] = 0.0
    xs[raw_mask] = 1.0
    ym, ys = fit_scaler(targets[indices], train_groups)
    x = torch.from_numpy(((full - xm) / xs).astype(np.float32))
    y = torch.from_numpy(((targets - ym) / ys).astype(np.float32))
    model = make_model(x.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                 weight_decay=CONFIG["weight_decay"])
    weight = torch.tensor(CONFIG["regression"]["weights"], dtype=torch.float32)
    best, best_epoch, wait, best_state = float("inf"), -1, 0, None
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x[indices])
        losses = []
        for group in sorted(set(train_groups)):
            loc = [j for j, i in enumerate(indices) if groups[i] == group]
            regression = F.smooth_l1_loss(pred[loc], y[np.asarray(indices)[loc]],
                                           beta=CONFIG["regression"]["beta"], reduction="none")
            losses.append((regression * weight).mean())
        loss = torch.stack(losses).mean()
        loss = loss + pair_loss(pred, indices, groups, candidates, pair_labels,
                                weights=CONFIG["ranking"]["weights"])
        if teacher_train_predictions is not None:
            teacher = torch.from_numpy(((teacher_train_predictions - ym) / ys).astype(np.float32))
            loss = loss + CONFIG["distillation"]["auxiliary_weight"] * F.mse_loss(pred, teacher[indices])
        loss.backward()
        optimizer.step()
        current = float(loss.detach())
        if current < best - CONFIG["early_stop"]["min_delta"]:
            best, best_epoch, wait = current, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CONFIG["early_stop"]["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(x).numpy() * ys + ym
    return predicted.astype(np.float64), {"best_train_loss": best, "best_epoch": best_epoch + 1,
                                          "epochs_run": epoch + 1, "input_dim": int(x.shape[1]),
                                          "train_groups": sorted(set(train_groups)),
                                          "input_scaler_sha256": hashlib.sha256(xm.tobytes() + xs.tobytes()).hexdigest(),
                                          "target_scaler_sha256": hashlib.sha256(ym.tobytes() + ys.tobytes()).hexdigest()}


def rank_metrics(rows, prediction, target, indices, groups, candidates, pair_labels):
    per_group = []
    for group in sorted(set(groups[i] for i in indices)):
        picks = [i for i in indices if groups[i] == group]
        real = target[picks]
        guessed = prediction[picks]
        result = {"group": group, "n": len(picks), "heads": {}}
        for h, name in enumerate(HEADS):
            err = guessed[:, h] - real[:, h]
            rho = spearmanr(guessed[:, h], real[:, h]).statistic if np.std(guessed[:, h]) > 1e-9 and np.std(real[:, h]) > 1e-9 else np.nan
            result["heads"][name] = {"mae": float(np.mean(np.abs(err))),
                                      "rmse": float(np.sqrt(np.mean(err**2))),
                                      "spearman": None if not np.isfinite(rho) else float(rho),
                                      "prediction_variance": float(np.var(guessed[:, h])),
                                      "collapsed": bool(np.std(guessed[:, h]) < 0.1 * max(np.std(real[:, h]), 1e-6))}
        by_candidate = {}
        for pos, i in enumerate(picks):
            by_candidate.setdefault(candidates[i], []).append(pos)
        mean_pred = {c: np.mean(guessed[j], axis=0) for c, j in by_candidate.items()}
        mean_real = {c: np.mean(real[j], axis=0) for c, j in by_candidate.items()}
        selection = max(mean_pred, key=lambda c: mean_pred[c][0])
        oracle = max(mean_real, key=lambda c: mean_real[c][0])
        top3 = sorted(mean_pred, key=lambda c: mean_pred[c][0], reverse=True)[:3]
        direct = mean_real["direct_reference"]
        result["selection"] = {"candidate": selection, "top1_recall": selection == oracle,
                               "top3_recall": oracle in top3,
                               "oracle_reward_regret": float(mean_real[oracle][0] - mean_real[selection][0]),
                               "selected_reward": float(mean_real[selection][0]),
                               "selected_goal_reduction": float(mean_real[selection][2]),
                               "reward_vs_direct": float(mean_real[selection][0] - direct[0]),
                               "goal_vs_direct": float(mean_real[selection][2] - direct[2])}
        strict, correct, conflicts = 0, 0, 0
        for pair in pair_labels[group]:
            label = pair["reward_label"]
            if label not in ("i_better", "j_better"):
                continue
            a, b = pair["candidate_i"], pair["candidate_j"]
            if a not in mean_pred or b not in mean_pred:
                continue
            strict += 1
            correct += int((mean_pred[a][0] > mean_pred[b][0]) == (label == "i_better"))
            conflicts += int(pair["reward_goal_conflict"])
        result["ranking"] = {"strict_reward_pairs": strict,
                             "pairwise_accuracy": correct / strict if strict else None,
                             "reward_goal_conflict_pairs": conflicts}
        per_group.append(result)
    return per_group


def main():
    start = time.time()
    if CONFIG["status"] != "frozen_before_training" or VARIANTS != list(FEATURES):
        raise ValueError("unfrozen or mismatched training config")
    rows, labels, records = audit_run()
    groups = [record["group"] for record in records]
    candidates = [record["candidate"] for record in records]
    targets = np.asarray([[float(label[key]) for key in TARGET_KEYS] for label in labels], dtype=np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("nonfinite observed target")
    pairs_json = json.loads((MULTI / "pairwise-labels.json").read_text())
    pair_labels = {group: [pair for pair in pairs_json["candidate_pairs"] if pair["state_group_id"] == group]
                   for group in sorted(set(groups))}
    single = []
    first = min(set(groups))
    single_idx = [i for i, record in enumerate(records) if record["group"] == first and
                  ((record["candidate"] == "direct_reference" and record["repeat"] == 0) or
                   record["candidate"] == "left_1")]
    if len(single_idx) != 2:
        raise ValueError("single batch identity changed")
    status = {"stage": "single_batch", "completed": 0, "total": 8 * 3 + 8 * 3 + 7 * 8 * 3,
              "config_sha256": hashlib.sha256((OUT / "training-smoke-config.yaml").read_bytes()).hexdigest()}
    write_json(OUTPUT / "status.json", status)
    teacher_cache = {}
    for variant in VARIANTS:
        for seed in CONFIG["seeds"]:
            teacher = None
            if variant == "B7":
                teacher = teacher_cache[("single", seed)]
            pred, meta = train_variant(rows, targets, groups, candidates, pair_labels, single_idx,
                                       variant, seed, CONFIG["max_epochs"]["single_batch"], teacher)
            if variant == "B3":
                teacher_cache[("single", seed)] = pred
            residual = pred[single_idx] - targets[single_idx]
            single.append({"variant": variant, "seed": seed, **meta,
                           "primary_mae": float(np.abs(residual[:, :3]).mean()),
                           "feature_collision": bool(np.array_equal(raw_features(rows, variant)[0][single_idx[0]],
                                                                     raw_features(rows, variant)[0][single_idx[1]]))})
            status["completed"] += 1
            write_json(OUTPUT / "status.json", status)
    write_json(OUT / "single-batch-overfit-results.json", single)
    status["stage"] = "seven_group"
    write_json(OUTPUT / "status.json", status)
    all_idx = list(range(len(rows)))
    seven = []
    teacher_cache.clear()
    for variant in VARIANTS:
        for seed in CONFIG["seeds"]:
            teacher = teacher_cache.get(seed) if variant == "B7" else None
            pred, meta = train_variant(rows, targets, groups, candidates, pair_labels, all_idx,
                                       variant, seed, CONFIG["max_epochs"]["seven_group"], teacher)
            if variant == "B3":
                teacher_cache[seed] = pred
            seven.append({"variant": variant, "seed": seed, **meta,
                          "primary_mae": float(np.abs(pred[:, :3] - targets[:, :3]).mean()),
                          "groups": rank_metrics(rows, pred, targets, all_idx, groups, candidates, pair_labels)})
            status["completed"] += 1
            write_json(OUTPUT / "status.json", status)
    write_json(OUT / "seven-group-overfit-results.json", seven)
    status["stage"] = "loso"
    write_json(OUTPUT / "status.json", status)
    loso = []
    predictions = []
    for held, train, test in loso_splits(groups):
        teacher_cache.clear()
        for variant in VARIANTS:
            for seed in CONFIG["seeds"]:
                teacher = teacher_cache.get(seed) if variant == "B7" else None
                pred, meta = train_variant(rows, targets, groups, candidates, pair_labels, train,
                                           variant, seed, CONFIG["max_epochs"]["loso"], teacher)
                if variant == "B3":
                    teacher_cache[seed] = pred
                metrics = rank_metrics(rows, pred, targets, test, groups, candidates, pair_labels)
                loso.append({"held_out": held, "variant": variant, "seed": seed, **meta,
                             "test_metrics": metrics[0]})
                predictions.append({"held_out": held, "variant": variant, "seed": seed,
                                    "candidate_ids": [candidates[i] for i in test],
                                    "prediction": pred[test].tolist(), "target": targets[test].tolist()})
                status["completed"] += 1
                write_json(OUTPUT / "status.json", status)
    write_json(OUT / "loso-diagnostic-results.json", loso)
    write_json(OUTPUT / "predictions.json", predictions)
    status.update({"stage": "complete", "elapsed_sec": time.time() - start})
    write_json(OUTPUT / "status.json", status)
    print(f"complete: {len(loso)} LOSO runs, {status['elapsed_sec']:.1f} s")


if __name__ == "__main__":
    main()
