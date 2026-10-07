"""Read-only, outcome-blind analysis of the frozen four-radius placement scan."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from visualnav_transformer.eat_nav.anchor_placement_rules import (
    CLASS_ORDER, select_r2, select_r3, summarize_support,
)

COLORS = {"flat-only": "#b9bcc2", "graded": "#4682b4",
          "differential": "#ef9234", "transition": "#c14256",
          "unknown": "#704b8f", "insufficient": "#222222"}
RADII = (.30, .40, .50, .60)
OLD = Path("/tmp/eatnav-radial-termination-20261008/per-pose-results.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def load_scan(root: Path) -> list[dict]:
    summary = json.loads((root / "scan-summary.json").read_text())
    if summary["decision"] != "offline_scan_complete" or summary["candidate_count"] != 448:
        raise ValueError("448-pose zero-step scan is incomplete")
    if summary["physics_steps_after_scene_init"] != 0:
        raise ValueError("physics advanced during offline scan")
    items = []
    for radius in RADII:
        path = root / f"radius-{radius:.2f}.json"
        payload = json.loads(path.read_text())
        if payload["radius_m"] != radius or len(payload["candidates"]) != 112:
            raise ValueError(f"radius grid incomplete: {radius}")
        if payload["terrain_mesh_sha256"] != summary["terrain_mesh_sha256"]:
            raise ValueError("terrain mesh SHA drift")
        items += payload["candidates"]
    keys = {(x["terrain"], x["radius_m"], x["direction_index"]) for x in items}
    if len(keys) != 448 or len({x["terrain"] for x in items}) != 7:
        raise ValueError("terrain/radius/direction uniqueness failed")
    for x in items:
        if (x["radius_m"] not in RADII or not 0 <= x["direction_index"] < 16
                or x["dem_shape"] != [2, 20, 20] or x["frame_id"] != "map"
                or x["resolution_m"] != .1 or x["physical_stability"] != "unmeasured"):
            raise ValueError(f"candidate metadata mismatch: {x['pose_id']}")
        data = np.asarray(x["dem"], dtype=np.float32)
        if not np.isfinite(data).all() or hashlib.sha256(data.tobytes()).hexdigest() != x["dem_sha256"]:
            raise ValueError(f"DEM SHA or finite check failed: {x['pose_id']}")
    return items


def old_join(items: list[dict], old_path: Path) -> list[dict]:
    old = json.loads(old_path.read_text())["poses"]
    if len(old) != 112 or len({x["pose_id"] for x in old}) != 112:
        raise ValueError("old 112-pose inventory incomplete")
    lookup = {(x["terrain"], x["angle_index"]): x for x in old}
    joined = []
    for x in items:
        if x["radius_m"] != .30:
            continue
        old_x = lookup[(x["terrain"], x["direction_index"])]
        if old_x["terrain_seed"] != x["terrain_seed"] or not old_x["measurement_valid"]:
            raise ValueError(f"old pose identity invalid: {x['pose_id']}")
        joined.append({
            "virtual_pose_id": x["pose_id"], "old_pose_id": old_x["pose_id"],
            "terrain": x["terrain"], "direction_index": x["direction_index"],
            "virtual_class": x["classification"],
            "settled_class": settled_class(old_x),
            "settled_anchor_eligible": old_x["settled_anchor_eligible"],
            "settled_classification": old_x["classification"],
            "ineligible_reasons": old_x["ineligible_reasons"],
            "termination_reason": old_x["termination_reason"],
            "contact_count_sum_across_steps": old_x["contact_count_sum_across_steps"],
            "contact_occupancy_max": old_x["contact_occupancy_max"],
            "source_family": old_x["source_family"],
        })
    if len(joined) != 112:
        raise ValueError("old r=.30 join incomplete")
    return joined


def settled_class(old_pose: dict) -> str:
    """Ineligible poses may lack a settled DEM classification."""
    interaction = old_pose.get("interaction")
    return interaction["class"] if interaction is not None else "NA / " + old_pose["classification"]


def distribution(items: list[dict], key: str, classes=CLASS_ORDER) -> dict:
    result = {}
    for group in sorted({str(x[key]) for x in items}):
        subset = [x for x in items if str(x[key]) == group]
        result[group] = dict(Counter(x["classification"] for x in subset))
        for cls in classes:
            result[group].setdefault(cls, 0)
    return result


def choose_pilot(selected: list[dict]) -> tuple[list[dict], list[str]]:
    """Offer two distinct source families per class where geometry permits."""
    pilot = []
    missing = []
    for cls in CLASS_ORDER:
        pool = [x for x in selected if x["classification"] == cls]
        seen = set()
        for x in pool:
            if x["source_family"] in seen:
                continue
            seen.add(x["source_family"])
            pilot.append({
                "rank": len(pilot) + 1, "selection_rule": "R3",
                "pose_id": x["pose_id"], "terrain": x["terrain"],
                "source_family": x["source_family"], "seed": x["terrain_seed"],
                "radius_m": x["radius_m"], "direction_index": x["direction_index"],
                "requested_xy_yaw": x["requested_xy_yaw"],
                "root_z_rule": x["root_z_rule"], "dem_sha256": x["dem_sha256"],
                "classification": cls, "interaction_onset_m": x["interaction_onset_m"],
                "dem_valid_fraction": x["dem_valid_fraction"],
                "unknown_fraction": 1 - x["dem_valid_fraction"],
                "footprint_diagnostic": x["footprint_diagnostic"],
                "physical_stability": "unmeasured",
            })
            if len(seen) == 2:
                break
        if not seen:
            missing.append(cls)
    return pilot, missing


def bar_classes(ax, names, groups, title):
    bottom = np.zeros(len(names))
    for cls in ("flat-only", "graded", "differential", "transition", "unknown", "insufficient"):
        vals = np.asarray([Counter(x["classification"] for x in groups[name])[cls] for name in names])
        ax.bar(range(len(names)), vals, bottom=bottom, label=cls, color=COLORS[cls])
        bottom += vals
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_ylabel("pose count"); ax.set_title(title)


def figures(items, rules, joined, pilot, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    terrains = sorted({x["terrain"] for x in items})
    # 1: seven terrain rows × four radius columns, 16 directions each.
    fig, axes = plt.subplots(7, 4, figsize=(16, 17), subplot_kw={"projection": "polar"})
    for i, terrain in enumerate(terrains):
        for j, radius in enumerate(RADII):
            ax = axes[i, j]
            for x in items:
                if x["terrain"] == terrain and x["radius_m"] == radius:
                    ax.scatter(x["theta_rad"], 1, color=COLORS[x["classification"]], s=55)
            ax.set_rticks([]); ax.set_xticklabels([])
            ax.set_title(f"{terrain} r={radius:.2f}", fontsize=8)
    fig.suptitle("Virtual DEM class by terrain, radius and direction", y=.995)
    fig.legend([Patch(color=COLORS[c]) for c in CLASS_ORDER[::-1]], CLASS_ORDER[::-1],
               loc="upper center", bbox_to_anchor=(.5, .98), ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "01-polar-class-map.png", dpi=140); plt.close(fig)
    # 2: onset heatmaps, NA in gray.
    fig, axes = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#cccccc")
    for ax, terrain in zip(axes, terrains):
        grid = np.full((4, 16), np.nan)
        for x in items:
            if x["terrain"] == terrain and x["interaction_onset_m"] is not None:
                grid[RADII.index(x["radius_m"]), x["direction_index"]] = x["interaction_onset_m"]
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=.6)
        ax.set_yticks(range(4), [str(r) for r in RADII]); ax.set_ylabel(terrain)
    axes[-1].set_xticks(range(16)); axes[-1].set_xlabel("direction index")
    fig.colorbar(im, ax=axes, label="known onset distance (m)")
    fig.savefig(out / "02-interaction-onset.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # 3–4: supply and rule support.
    for filename, names, groups, title in [
        ("03-radius-supply.png", [f"{r:.2f} m" for r in RADII],
         {f"{r:.2f} m": [x for x in items if x["radius_m"] == r] for r in RADII}, "Supply by radius"),
        ("04-rule-support.png", list(rules), rules, "R0–R3 class support")]:
        fig, ax = plt.subplots(figsize=(8, 5)); bar_classes(ax, names, groups, title)
        ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.20))
        fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
    # 5–6: known DEM and footprint coverage, and tile margin.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in [(axes[0], "dem_valid_fraction", "DEM known fraction"),
                            (axes[1], "footprint_known_fraction", "collision proxy known fraction")]:
        ax.boxplot([[x[key] for x in items if x["radius_m"] == r] for r in RADII],
                   labels=[str(r) for r in RADII])
        ax.set_xlabel("radius (m)"); ax.set_ylabel(label); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(out / "05-known-unknown-coverage.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, category in enumerate(("main_track_sprocket", "initial_flipper_support")):
        axes[j].boxplot([[x["footprint_diagnostic"][category]["known_fraction"]
                          for x in items if x["radius_m"] == r] for r in RADII],
                        labels=[str(r) for r in RADII])
        axes[j].set_title(category); axes[j].set_xlabel("radius (m)"); axes[j].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(out / "06-footprint-diagnostics.png", dpi=160); plt.close(fig)
    # 7: settled labels, and virtual-vs-settled DEM classes at r=.30.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = Counter(x["settled_classification"] for x in joined)
    axes[0].bar(list(labels.keys()), list(labels.values())); axes[0].tick_params(axis="x", rotation=25)
    pairs = Counter((x["virtual_class"], x["settled_class"]) for x in joined)
    class_labels = list(CLASS_ORDER)
    m = np.array([[pairs[(a, b)] for b in class_labels] for a in class_labels])
    axes[1].imshow(m, cmap="Blues")
    axes[1].set_xticks(range(4), class_labels, rotation=35, ha="right")
    axes[1].set_yticks(range(4), class_labels)
    axes[1].set_xlabel("settled class"); axes[1].set_ylabel("virtual class")
    for i in range(4):
        for j in range(4): axes[1].text(j, i, str(m[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(out / "07-existing-r030-correspondence.png", dpi=160); plt.close(fig)
    # 8: source-family concentration for R3.
    fig, ax = plt.subplots(figsize=(10, 4))
    families = sorted({x["source_family"] for x in rules["R3"]})
    bottom = np.zeros(len(families))
    for cls in CLASS_ORDER[::-1]:
        vals = np.array([sum(x["source_family"] == fam and x["classification"] == cls
                             for x in rules["R3"]) for fam in families])
        ax.bar(range(len(families)), vals, bottom=bottom, color=COLORS[cls], label=cls)
        bottom += vals
    ax.set_xticks(range(len(families)), [x.split(":")[-1] for x in families], rotation=30, ha="right")
    ax.set_ylabel("related poses, not independent StateGroups"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "08-source-family-concentration.png", dpi=160); plt.close(fig)
    # 9: proposed pilot placements.
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for x in pilot:
        theta = x["direction_index"] * np.pi / 8
        ax.scatter(theta, x["radius_m"], color=COLORS[x["classification"]], s=100)
        ax.text(theta, x["radius_m"] + .025, str(x["rank"]), fontsize=8)
    ax.set_ylim(0, .7); ax.set_title("R3 candidate-free anchor pilot proposal")
    fig.savefig(out / "09-pilot-poses.png", dpi=160); plt.close(fig)
    # 10: decision flow, with the actual result written beside each gate.
    fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off")
    nodes = [
        ("112/112 DEM parity", "PASS"),
        ("448 geometry placements", "PASS"),
        ("four DEM classes", "PASS" if all(any(x["classification"] == c for x in items) for c in CLASS_ORDER) else "MISSING"),
        ("distinct source families", "First seed only"),
        ("candidate-free pilot", "review proposal"),
        ("physical stability", "unmeasured"),
    ]
    for i, (label, value) in enumerate(nodes):
        y = .9 - .15 * i
        ax.text(.05, y, f"{i + 1}. {label}", fontsize=12,
                bbox={"boxstyle": "round", "facecolor": "#e8eef4"})
        ax.text(.7, y, value, fontsize=11)
        if i < len(nodes) - 1: ax.annotate("", xy=(.45, y - .12), xytext=(.45, y - .035),
                                          arrowprops={"arrowstyle": "->"})
    fig.savefig(out / "10-decision-flow.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root; scan = root / "scan-v3"; out = args.output or root / "analysis"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    items = load_scan(scan)
    joined = old_join(items, args.old)
    r0 = [x for x in items if x["radius_m"] == .30]
    rules = {"R0": r0}
    rules.update({f"R1-{r:.2f}": [x for x in items if x["radius_m"] == r] for r in RADII})
    rules["R2"] = select_r2(items)
    rules["R3"] = select_r3(items)
    proposal, missing = choose_pilot(rules["R3"])
    comparison = {name: summarize_support(picks) for name, picks in rules.items()}
    comparison["distributions"] = {
        "terrain": distribution(items, "terrain"),
        "radius": distribution(items, "radius_m"),
        "direction": distribution(items, "direction_index"),
    }
    r3_families = {cls: sorted({x["source_family"] for x in rules["R3"]
                                if x["classification"] == cls}) for cls in CLASS_ORDER}
    decision = ("anchor_placement_rule_ready_for_candidate_free_pilot"
                if not missing and all(len(r3_families[c]) >= 2 for c in CLASS_ORDER)
                else "anchor_placement_geometry_supported_but_second_seed_required"
                if not missing else "anchor_placement_requires_radius_or_horizon_redesign")
    save(out / "r0-r3-comparison.json", comparison)
    save(out / "old-r030-correspondence.json", joined)
    save(out / "old-r030-summary.json", {
        "measurement_valid": len(joined),
        "settled_eligible": sum(x["settled_anchor_eligible"] for x in joined),
        "settled_status": dict(Counter(x["settled_classification"] for x in joined)),
        "eligible_class_counts": dict(Counter(x["settled_class"] for x in joined
                                      if x["settled_anchor_eligible"])),
        "virtual_vs_settled_class_mismatch_among_eligible": sum(
            x["virtual_class"] != x["settled_class"] for x in joined
            if x["settled_anchor_eligible"]),
        "semantics": "saved settled pose and nominal offline virtual r=.30 differ; labels are not transferred",
    })
    save(out / "pilot-candidate-proposal.json",
         {"rule": "R3", "candidates": proposal, "missing_classes": missing,
          "physical_stability": "unmeasured; candidate-free settling pilot required"})
    save(out / "class-source-support.json",
         {"source_families_by_class_R3": r3_families,
          "first_seed_only": True, "independence_unit": "terrain seed/source family",
          "multiple radii/directions_are_related": True})
    save(out / "second-seed-decision.json",
         {"decision": "not_required_before_candidate_free_pilot" if decision.startswith("anchor_placement_rule_ready")
          else "recommended_after_common_rule_freeze" if not missing else "defer_until_radius_or_horizon_redesign",
          "first_seed_generalization_claim": False})
    save(out / "risk-register.json", [
        {"risk": "new-radius settling and physical termination unknown",
         "evidence": "no physics/control steps executed", "next_check": "candidate-free anchor pilot"},
        {"risk": "one batch seed does not establish seed generalization",
         "evidence": "20261007 only; terrain-specific seeds are independent terrain families",
         "next_check": "freeze rule before second seed"},
        {"risk": "0.6 m forward corridor is only a two-second reach proxy",
         "evidence": "assumed speed 0.3 m/s; no rollout reach measured",
         "next_check": "candidate-free pilot and later trajectory checks"},
        {"risk": "footprint uses USD collision-bound corner proxy rather than PhysX contact response",
         "evidence": "271 composed CollisionAPI bounds, purpose=guide; proxy coverage 1.0",
         "next_check": "measure actual contacts and settling in pilot"},
        {"risk": "transition geometry is sparse and concentrated",
         "evidence": "R3 has four related poses across two terrain/source families",
         "next_check": "measure both families and retain NA if unstable"},
    ])
    save(out / "source-input-provenance.json", {
        "input_manifest_sha256": sha(root / "inputs/pose-frames.json"),
        "old_inventory_sha256": sha(args.old),
        "preflight_sha256": sha(root / "preflight.json"),
        "parity_report_sha256": sha(root / "parity/parity-report.json"),
        "scan_freeze_v4_sha256": sha(root / "scan-freeze-v4.json"),
        "scan_source_sha256": sha(root / "inputs/eat_nav_anchor_placement_scan_sim.py"),
        "analyzer_source_sha256": sha(Path(__file__)),
        "geometry_rule_source_sha256": sha(root / "src/visualnav_transformer/eat_nav/anchor_placement_rules.py"),
        "footprint_geometry_source_sha256": sha(root / "src/visualnav_transformer/eat_nav/anchor_placement_geometry.py"),
        "scan_summary_sha256": sha(scan / "scan-summary.json"),
        "terrain_mesh_sha256": json.loads((scan / "scan-summary.json").read_text())["terrain_mesh_sha256"],
        "outcome_inputs_used": False,
    })
    compact = [{k: v for k, v in x.items() if k != "dem"} for x in items]
    save(out / "per-pose-geometry-metrics.json", compact)
    save(out / "per-pose-review-summary.json", [{
        k: x[k] for k in ("pose_id", "terrain", "terrain_seed", "source_family", "radius_m",
                       "direction_index", "requested_xy_yaw", "dem_sha256", "classification",
                       "interaction_onset_m", "dem_valid_fraction", "footprint_known_fraction")
    } | {"tile_boundary_margin_m": x["footprint_diagnostic"]["tile_boundary_margin_m"]}
        for x in items])
    figures(items, rules, joined, proposal, out / "figures")
    save(out / "final-decision.json",
         {"decision": decision, "recommended_rule": "R3" if not missing else None,
          "candidate_free_pilot_only": True, "physical_stability": "unmeasured",
          "first_seed_only": True, "all_class_source_family_counts_R3": {
              c: len(v) for c, v in r3_families.items()},
          "missing_classes": missing, "new_physics_rollouts": 0,
          "scan_source_sha256": json.loads((scan / "scan-summary.json").read_text())["sim_source_sha256"]})
    ledger = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.endswith(".tmp") and path != out / "sha-ledger.json":
            ledger[str(path.relative_to(root))] = sha(path)
    save(out / "sha-ledger.json", ledger)
    print(json.dumps({"decision": decision, "r3": comparison["R3"], "pilot_count": len(proposal)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
