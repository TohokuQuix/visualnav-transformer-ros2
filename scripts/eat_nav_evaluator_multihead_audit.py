"""Read-only M0/M1-M5 audit and label export of the sealed E1 train traces."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from eat_nav_evaluator_design_audit import main as source_audit, OLD, NEW, sha
from eat_nav_evaluator_design_analyze import input_records
from eat_nav_g40_teacher_review import metrics_for_run
from eat_nav_evaluator_multihead import make_labels, pair_label

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-freeze"
BOUNDARY = ROOT / "docs/quality-identifiability-gate/g40-boundary-freeze"
PRIOR = ROOT / "docs/quality-identifiability-gate/g40-teacher-review"
CONFIG = OUT / "multihead-analysis-config.yaml"


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def records(path):
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source]


def preflight():
    audit = source_audit()
    if audit["status"] != "pass" or audit["live_trace_count"] != 63 or audit["group_count"] != 7:
        raise ValueError("sealed source audit failed")
    previous = read(BOUNDARY / "source-provenance.json")
    if previous["status"] != "pass" or previous["sealed_trace_count"] != 63:
        raise ValueError("G40 provenance failed")
    for name, expected in previous["source_files_sha256"].items():
        path = ROOT / name if not Path(name).is_absolute() else Path(name)
        if sha(path) != expected:
            raise ValueError(f"frozen source mismatch: {name}")
    ledger = read(BOUNDARY / "artifact-sha-ledger.json")
    for name, item in ledger["files"].items():
        if sha(ROOT / name) != item["sha256"]:
            raise ValueError(f"G40 artifact changed: {name}")
    per = {row["process_id"]: row for row in records(BOUNDARY / "terminal-value-per-candidate.jsonl.gz")}
    if len(per) != 63:
        raise ValueError("prior terminal rows missing")
    trace_sha = {}
    identity = []
    for source in input_records():
        pid = source["process_id"]
        root = OLD if source["state_group_id"].startswith("g0e-") else NEW
        trace = root / "runs" / pid / "live-0.trace.json.gz"
        digest = sha(trace)
        if digest != previous["source_trace_sha256"][pid] or digest != per[pid]["source_trace_sha256"]:
            raise ValueError(f"{pid}: trace SHA mismatch")
        done = read(root / "runs" / pid / "run-complete.json")
        trace_sha[pid] = digest
        identity.append({"process_id": pid, "state_group_id": source["state_group_id"],
                         "candidate_id": source["candidate_id"], "repeat_id": source["repeat_id"],
                         "seed": done["identity"]["seed"], "episode_key": done["identity"]["episode_key"],
                         "map_path_file_sha256": done["identity"]["map_path_file_sha256"]})
    if len(identity) != 63 or len({r["state_group_id"] for r in identity}) != 7:
        raise ValueError("run identity count mismatch")
    save("m0-provenance.json", {"status": "pass", "source_base_commit": "51d9bf201f76fd8e3b54d3d9496ca4c3429cbb3a",
         "prior_g40_source_base_commit": read(BOUNDARY / "artifact-sha-ledger.json")["source_base_commit"],
         "worktree": str(ROOT), "branch": "exp/eatnav-evaluator-multihead",
         "config_sha256_before_labels": sha(CONFIG), "g40_boundary_ledger_sha256": sha(BOUNDARY / "artifact-sha-ledger.json"),
         "g40_boundary_provenance_sha256": sha(BOUNDARY / "source-provenance.json"),
         "g40_teacher_ledger_sha256": sha(PRIOR / "artifact-sha-ledger.json"),
         "critic_replay_artifact_sha256": sha(BOUNDARY / "critic-offline-replay-equivalence.json"),
         "boundary_sensitivity_artifact_sha256": sha(BOUNDARY / "bootstrap-rank-stability.json"),
         "candidate_generator_config_sha256": sha(ROOT / "docs/quality-identifiability-gate/expanded-feasibility/expanded-pilot-config.yaml"),
         "coordinate_frame_contract_sha256": sha(Path('/tmp/eatnav-e1-coordinate-addendum-r2-20261004/coordinate-frame-spec.json')),
         "reward_definition_sha256": sha(PRIOR / "rl-return-definition.json"),
         "trace_index_contract_sha256": sha(PRIOR / "trace-index-contract.json"),
         "previous_input_contract_sha256": sha(BOUNDARY / "portable-evaluator-input-contract.json"),
         "previous_teacher_contract_sha256": sha(BOUNDARY / "teacher-contract.yaml"),
         "previous_safety_contract_sha256": sha(BOUNDARY / "safety-head-disabled-contract.json"),
         "checkpoint_sha256": audit["checkpoint_sha256_rechecked"],
         "source_audit": audit, "trace_sha256": trace_sha, "run_identity": identity,
         "split_use": {"train": "seven groups for offline design only", "validation": "unused", "held_out": "unused"},
         "raw_trace_mutation": False})
    return {"status": "pass", "trace_count": len(trace_sha)}


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"count": 0, "range": None, "median": None, "iqr": None}
    def quantile(q):
        p = q * (len(values)-1)
        lo = math.floor(p); hi = math.ceil(p)
        return values[lo] + (values[hi]-values[lo])*(p-lo)
    return {"count": len(values), "range": [values[0], values[-1]],
            "median": quantile(.5), "iqr": [quantile(.25), quantile(.75)]}


def labels():
    config = yaml.safe_load(CONFIG.read_text())
    provenance = read(OUT / "m0-provenance.json")
    if provenance["status"] != "pass" or sha(CONFIG) != provenance["config_sha256_before_labels"]:
        raise ValueError("M0/config freeze absent or changed")
    prior = {row["process_id"]: row for row in records(PRIOR / "g40-per-candidate.jsonl.gz")}
    boundary = {row["process_id"]: row for row in records(BOUNDARY / "terminal-value-per-candidate.jsonl.gz")}
    all_rows = []
    for source in input_records():
        pid = source["process_id"]
        computed = metrics_for_run(source, config["gamma"])
        old = prior[pid]; conditional = boundary[pid]
        if (computed["trace_sha256"] != provenance["trace_sha256"][pid] or
                computed["R_gamma_40"] != old["R_gamma_40"] or
                computed["R_gamma_40"] != conditional["reward_discounted_2s"]):
            raise ValueError(f"{pid}: reward/source disagreement")
        if (computed["R_gamma_40"] != math.fsum(config["gamma"]**k*r for k,r in enumerate(computed["reward_0_39"])) or
                len(computed["reward_0_39"]) != 40 or computed["body_metrics"]["censored"] or
                computed["body_metrics"]["termination"]):
            raise ValueError(f"{pid}: reward alignment/censor mismatch")
        for key in ("goal_distance_reduction_m", "endpoint_error_m", "cross_track_p95_m", "progress_ratio", "endpoint_encounter_0_2m"):
            if computed["map_metrics"][key] != old["map_metrics"][key]:
                raise ValueError(f"{pid}: corrected frame mismatch at {key}")
        m = computed["map_metrics"]
        zero = conditional["terminal_value"]["zero_recorded"] if max(abs(c) for c in conditional["terminal_command"]["zero_recorded"]) < 1e-12 else None
        row = {"process_id": pid, "state_group_id": source["state_group_id"],
               "candidate_id": source["candidate_id"], "repeat_id": source["repeat_id"],
               "terrain": source["terrain"], "source_trace_sha256": computed["trace_sha256"],
               "reward": computed["R_gamma_40"], "goal": m["goal_distance_reduction_m"],
               "endpoint": m["endpoint_error_m"], "endpoint_encounter_2s": m["endpoint_encounter_0_2m"],
               "path_progress_2s": m["progress_ratio"], "cross_track_p95_2s": m["cross_track_p95_m"],
               "valid": True, "termination": False, "censored": False,
               "terminal_value_hold": conditional["terminal_value"]["hold_candidate"],
               "terminal_value_canonical": conditional["terminal_value"]["canonical"],
               "terminal_value_direct": conditional["terminal_value"]["direct_reference"],
               "recorded_boundary_value": conditional["terminal_value"]["zero_recorded"],
               "terminal_value_literal_zero": zero, "V0": computed["v0"],
               "critic_residual_hold": conditional["A40"]["hold_candidate"],
               "critic_residual_canonical": conditional["A40"]["canonical"],
               "critic_drawdown": computed["critic_drawdown"]["initial_to_minimum"],
               "tilt_rad": computed["body_metrics"]["maximum_roll_pitch_rad"]}
        power = config["gamma"]**config["horizon_transitions"]
        for suffix, value in (("hold", row["terminal_value_hold"]), ("canonical", row["terminal_value_canonical"])):
            if abs(row[f"critic_residual_{suffix}"] - (row["reward"]+power*value-row["V0"])) > 1e-9:
                raise ValueError(f"{pid}: residual mismatch")
        all_rows.append(row)
    if len(all_rows) != 63:
        raise ValueError("63 complete labels required")
    floor = config["tie_margin"]["numeric_floor"]
    result = make_labels(all_rows, {"reward": floor["reward"],
                                    "goal": floor["goal_distance_reduction_m"],
                                    "endpoint": floor["endpoint_error_m"]})
    result["identity"] = "seven train StateGroups; 63 sealed traces; 21 unique candidate pairs/group"
    result["conditional_terminal_semantics"] = "offline Critic query on fixed observed terminal body state, not physically realized continuation"
    with gzip.open(OUT / "multihead-labels.jsonl.gz", "wt") as target:
        for row in result["absolute_and_relative_labels"]:
            target.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    save("pairwise-labels.json", {"margins": result["margins"], "direct_baselines": result["direct_baselines"],
         "candidate_pairs": result["candidate_pairs"], "candidate_summaries": result["candidate_summaries"],
         "statistical_unit": "StateGroup", "pair_count_is_descriptive_not_independent": True})
    head_keys = {"discounted_reward_2s": "reward", "direct_relative_reward_2s": "direct_relative_reward",
                 "goal_distance_reduction_2s": "goal", "endpoint_error_2s": "endpoint",
                 "endpoint_encounter_2s": "endpoint_encounter_2s", "path_progress_2s": "path_progress_2s",
                 "cross_track_p95_2s": "cross_track_p95_2s", "terminal_value_hold": "terminal_value_hold",
                 "terminal_value_canonical": "terminal_value_canonical", "terminal_value_literal_zero": "terminal_value_literal_zero",
                 "critic_residual_hold": "critic_residual_hold", "critic_residual_canonical": "critic_residual_canonical"}
    labels_out = result["absolute_and_relative_labels"]
    group_rows = defaultdict(list)
    for row in labels_out:
        group_rows[row["state_group_id"]].append(row)
    sufficiency = {}
    for head, key in head_keys.items():
        values = [float(row[key]) for row in labels_out if row[key] is not None]
        direct_noise = [max(row[key] for row in group if row["candidate_id"] == "direct_reference")-
                        min(row[key] for row in group if row["candidate_id"] == "direct_reference")
                        for group in group_rows.values() if all(row[key] is not None for row in group if row["candidate_id"] == "direct_reference")]
        spreads = [max(float(row[key]) for row in group if row[key] is not None)-
                   min(float(row[key]) for row in group if row[key] is not None)
                   for group in group_rows.values() if any(row[key] is not None for row in group)]
        noise=max(direct_noise) if direct_noise else None
        spread=statistics.median(spreads) if spreads else None
        signs={"negative":sum(v<0 for v in values), "zero":sum(v==0 for v in values),
               "positive":sum(v>0 for v in values)}
        class_groups = ({str(value): len({row["state_group_id"] for row in labels_out if row[key] == value})
                         for value in (False, True)} if head == "endpoint_encounter_2s" else None)
        classification = ("insufficient_label_variation" if len(set(values)) <= 1 or
                          (head == "endpoint_encounter_2s" and
                           (min(signs["positive"], len(values)-signs["positive"]) < 7 or
                            min(class_groups.values()) < 3))
                          else "supported_for_pipeline_smoke")
        sufficiency[head] = {"valid": len(values), "missing_or_censored": 63-len(values),
                             **distribution(values), "duplicate_noise_max": noise,
                             "median_candidate_spread": spread,
                             "candidate_to_duplicate_ratio": spread/noise if noise not in (None,0) else None,
                             "ratio_status": "defined" if noise not in (None,0) else "denominator_zero",
                             "state_group_support": len({row["state_group_id"] for row in labels_out if row[key] is not None}),
                             "terrain_support": len({row["terrain"] for row in labels_out if row[key] is not None}),
                             "sign_support": signs, "class_state_group_support": class_groups,
                             "classification": classification,
                             "formal_generalization_support": "insufficient_state_groups_seven_train_groups_only"}
    counts={name:dict(Counter(row[name] for row in result["candidate_pairs"])) for name in ("reward_label", "goal_label", "endpoint_label")}
    save("data-sufficiency-review.json", {"heads": sufficiency, "pair_labels": counts,
         "reward_goal_conflict_pairs": sum(p["reward_goal_conflict"] for p in result["candidate_pairs"]),
         "unique_candidate_pairs": len(result["candidate_pairs"]),
         "statistical_unit": "StateGroup", "state_group_count": len(group_rows),
         "no_formal_training_claim": True})
    save("label-generation-audit.json", {"status": "pass", "row_count": len(labels_out),
         "pair_count": len(result["candidate_pairs"]), "source_config_sha256": sha(CONFIG),
         "m0_provenance_sha256": sha(OUT/"m0-provenance.json"),
         "old_g40_reward_exact_match": True, "corrected_map_metrics_exact_match": True,
         "reward_steps": "trace[0:40]; once dt-scaled; no step40 or reset reward",
         "termination_censor_count": 0, "literal_zero_terminal_value_valid": sum(row["terminal_value_literal_zero"] is not None for row in labels_out)})
    return {"status": "pass", "rows": len(labels_out), "pairs": len(result["candidate_pairs"]), "margins": result["margins"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("preflight","labels"))
    stage=parser.parse_args().stage
    print(json.dumps(preflight() if stage=="preflight" else labels(),sort_keys=True))
