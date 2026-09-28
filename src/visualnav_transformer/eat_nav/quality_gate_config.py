"""Frozen diagnostic-only settings and explicit Quality gate decisions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class GateSettings:
    fixed_probe_q: float = 0.5
    training_seeds: tuple[int, ...] = (11, 29, 47)
    initial_permutations: int = 10
    final_permutations: int = 100
    min_mean_mse_gain: float = 0.05
    min_relative_rms_gain: float = 0.05
    min_delta_r2: float = 0.02
    min_b_effect_fraction: float = 0.5
    min_b_selection_fraction: float = 0.9
    min_b_direction_fraction: float = 0.9
    min_b_sampling_ratio: float = 1.0
    split_seed: int = 123
    bootstrap_seed: int = 613
    bootstrap_replicates: int = 2000
    match_seed: int = 123
    min_b_pairs: int = 32
    max_b_pairs: int = 64
    max_b_episode_uses: int = 3
    b_trajectory_rms_quantile: float = 0.75
    b_direction_abs_quantile: float = 0.5
    max_probe_epochs: int = 100
    probe_patience: int = 12
    probe_min_delta: float = 0.0001
    probe_batch_size: int = 64
    probe_learning_rate: float = 0.0003
    overfit_epochs: int = 300
    overfit_batch_size: int = 32
    overfit_learning_rate: float = 0.0001
    overfit_plateau_patience: int = 50
    overfit_plateau_delta: float = 0.0001
    overfit_loss_ceiling: float = 0.05
    diffusion_seeds: tuple[int, ...] = (11, 29, 47)


def load_gate_settings(path: Path) -> GateSettings:
    values = yaml.safe_load(Path(path).read_text()) or {}
    allowed = {f.name for f in fields(GateSettings)}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("invalid gate config keys")
    for name in ("training_seeds", "diffusion_seeds"):
        if name in values:
            values[name] = tuple(values[name])
    settings = GateSettings(**values)
    if settings.fixed_probe_q != 0.5 or len(settings.training_seeds) < 3:
        raise ValueError("fixed q must be 0.5 and at least three training seeds are required")
    if settings.initial_permutations < 10 or settings.final_permutations < 100:
        raise ValueError("insufficient permutation controls")
    if settings.min_b_pairs < 32 or settings.max_b_pairs > 64 or settings.max_b_episode_uses < 1:
        raise ValueError("invalid overfit pair limits")
    return settings


def probe_quality(kind: str, q: list[float], settings: GateSettings) -> list[float]:
    if kind == "probe_0":
        return [settings.fixed_probe_q] * len(q)
    if kind == "probe_q":
        return list(q)
    raise ValueError(f"unknown probe kind: {kind}")


def decide_gate_a(m: dict, settings: GateSettings) -> str:
    if m["validation_improving_at_cap"] or (
        m["probe_0_mse"] >= m["mean_mse"] * (1 - settings.min_mean_mse_gain)
        and m["probe_q_mse"] >= m["mean_mse"] * (1 - settings.min_mean_mse_gain)
    ):
        return "inconclusive"
    shuffled = np.asarray(m["shuffled"], dtype=float)
    if len(shuffled) < settings.initial_permutations:
        return "inconclusive"
    practical = m["relative_rms_improvement"] >= settings.min_relative_rms_gain or m["delta_r2"] >= settings.min_delta_r2
    other_pass = (m["improvement"] > 0 and m["ci_lower"] > 0 and practical
                  and m["terrain_robust"] and m["direction_agreement"])
    close = bool(m.get("near_boundary", False))
    if len(shuffled) < settings.final_permutations and (other_pass or close):
        return "needs_100_shuffles"
    permutation_pass = (m["improvement"] > float(shuffled.max()) if len(shuffled) < settings.final_permutations
                        else m["improvement"] > float(np.percentile(shuffled, 95)))
    return "pass" if other_pass and permutation_pass else "fail"


def decide_gate_b(m: dict, settings: GateSettings) -> str:
    if not m["training_converged"]:
        return "inconclusive"
    passed = (m["effect_fraction"] >= settings.min_b_effect_fraction
              and m["correct_selection_fraction"] >= settings.min_b_selection_fraction
              and m["direction_fraction"] >= settings.min_b_direction_fraction
              and m["effect_sampling_ratio"] > settings.min_b_sampling_ratio
              and m["seed_robust"])
    return "pass" if passed else "fail"
