"""Read-only analysis of two sealed candidate-free R3 settling shards."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKTREE = Path("/tmp/eatnav-vnt-r3-settling-pilot")
ROOT = Path(os.environ.get("EATNAV_R3_ROOT", "/tmp/eatnav-r3-settling-pilot-20261008"))
OFFLINE = Path("/tmp/eatnav-anchor-placement-20261008")
sys.path.insert(0, str(WORKTREE / "src"))
from visualnav_transformer.eat_nav.radial_shard_inventory import settled_exceedance_reasons, verify_shard_artifacts
from visualnav_transformer.eat_nav.radial_interaction import classify_dem
from visualnav_transformer.eat_nav.r3_anchor_decision import CLASSES, decide_pilot, physical_supply

COLORS = {"transition": "#c14256", "differential": "#ef9234",
          "graded": "#4682b4", "flat-only": "#b9bcc2"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_virtual():
    result = {}
    for radius in (.30, .40, .50, .60):
        payload = json.loads((OFFLINE / f"scan-v3/radius-{radius:.2f}.json").read_text())
        for item in payload["candidates"]:
            result[item["pose_id"]] = item
    if len(result) != 448:
        raise ValueError("frozen virtual scan incomplete")
    return result


def compare_dem(frame, virtual):
    if (frame["frame_index"] != 3 or frame["physics_step"] != 200 or
        frame["shape"] != [2, 20, 20] or frame["frame_id"] != "map" or
        frame["axis"] != "robot-relative-grid-xy" or frame["resolution_m"] != .1 or
        frame["finite"] is not True):
        raise ValueError("settled primary DEM frame contract changed")
    data = np.asarray(frame["transformed"], dtype=np.float32)
    if (data.shape != (2, 20, 20) or not np.isfinite(data).all() or
        hashlib.sha256(data.tobytes()).hexdigest() != frame["transformed_sha256"]):
        raise ValueError("settled DEM SHA/finite mismatch")
    interaction = classify_dem(data[0], data[1] > .5, .1)
    cls = interaction["class"]
    onset = interaction["interaction_start_m"]
    old_onset = virtual["interaction_onset_m"]
    if cls in ("unknown", "insufficient"):
        comparison = "insufficient/unknown"
    elif cls != virtual["classification"]:
        comparison = "class_mismatch"
    elif ((onset is None) != (old_onset is None) or
          (onset is not None and abs(onset - old_onset) > 1e-6)):
        comparison = "interaction_class_match_onset_difference"
    else:
        comparison = "exact_class_and_onset_match"
    old_profile = virtual["left_center_right_profiles_m"]
    profile = interaction.get("left_center_right_profiles_m")
    profile_rms = None
    if profile and old_profile:
        a = np.asarray([v for band in ("left", "center", "right") for v in profile[band]], dtype=float)
        b = np.asarray([v for band in ("left", "center", "right") for v in old_profile[band]], dtype=float)
        if a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all():
            profile_rms = float(np.sqrt(np.mean((a-b)**2)))
    return {
        "settled_class": cls, "settled_interaction_onset_m": onset,
        "virtual_class": virtual["classification"],
        "virtual_interaction_onset_m": old_onset,
        "comparison": comparison, "profile_rms_m": profile_rms,
        "settled_interaction": interaction,
        "virtual_interaction": virtual["interaction"],
        "dem_frame_index": 3, "dem_frame_physics_step": 200,
        "dem_frame_sha256": frame["transformed_sha256"],
        "dem_valid_fraction": float(frame["valid_fraction"]),
        "known_onset": onset is not None and cls not in ("unknown", "insufficient"),
    }


def analyze_pose(shard, row, virtual):
    root = ROOT / shard / "run"
    pose_id = row["pose_id"]
    trace_path = root / "trace" / f"{pose_id}.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    termination_path = root / "termination" / f"{pose_id}.json"
    terminated = termination_path.exists()
    frame_paths = sorted((root / "dem" / pose_id).glob("frame-*.json"))
    frames = [json.loads(path.read_text()) for path in frame_paths]
    if ((not frames and not terminated) or
        [x["frame_index"] for x in frames] != list(range(len(frames)))):
        raise ValueError("DEM frame sequence invalid: " + pose_id)
    if any(x["frame_id"] != "map" or x["resolution_m"] != .1 or x["shape"] != [2, 20, 20]
           for x in frames):
        raise ValueError("DEM history contract changed: " + pose_id)
    contact_count = [x["contact"]["contact_count"] for x in trace]
    penetration = [x["contact"]["max_decoded_penetration_m"] for x in trace]
    result = {
        "pose_id": pose_id, "rank": row["rank"], "shard_id": shard,
        "terrain": row["terrain"], "source_family": row["source_family"],
        "terrain_seed": row["terrain_seed"], "env_index": row["env_index"],
        "radius_m": row["radius_m"], "direction_index": row["angle_index"],
        "requested_xy_yaw": row["requested_xy_yaw"],
        "virtual_dem_sha256": row["virtual_dem_sha256"],
        "virtual_class": row["virtual_class"],
        "trace_length": len(trace), "dem_frame_count": len(frames),
        "contact_count_sum": sum(contact_count),
        "contact_occupancy_max": max(x["contact"]["buffer_occupancy"] for x in trace),
        "max_decoded_penetration_m": max(penetration),
        "trace_sha256": sha(trace_path),
        "termination_reason": [], "termination_step": None,
        "dem_valid_fraction": min((x["valid_fraction"] for x in frames), default=0),
        "settled_class": None, "known_onset": False,
        "comparison": "NA / no settled eligible DEM",
    }
    if terminated:
        evidence = json.loads(termination_path.read_text())
        result.update({
            "classification": "anchor_ineligible_physical_termination",
            "termination_reason": evidence["termination_reason"],
            "termination_step": evidence["termination_step"],
            "termination_evidence_sha256": sha(termination_path),
            "ineligible_reasons": [],
            "dem_valid_fraction": min((x["valid_fraction"] for x in frames), default=0),
        })
    else:
        if len(trace) != 200 or len(frames) != 4:
            raise ValueError("settled pose trace/DEM incomplete: " + pose_id)
        reasons = settled_exceedance_reasons(trace, [x["valid_fraction"] for x in frames])
        result["classification"] = "anchor_ineligible_numeric" if reasons else "settled_anchor_eligible"
        result["ineligible_reasons"] = reasons
        result["final40_max"] = {
            key: max(abs(x["state"][key]) for x in trace[160:])
            for key in ("xy_drift_m", "h1_signed_rad", "linear_speed_mps", "angular_speed_radps")
        }
        if not reasons:
            result.update(compare_dem(frames[3], virtual))
    result["physical_supply"] = physical_supply(result)
    return result


def resource_summary():
    records = []
    for shard in ("shard-A", "shard-B"):
        path = ROOT / shard / "resource-series.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        if not values:
            raise ValueError("resource series missing: " + shard)
        records.append({
            "shard_id": shard, "samples": len(values),
            "elapsed_sec": max(v["elapsed_sec"] for v in values),
            "gpu_device_peak_mib": max(v["gpu_device_used_mib"] for v in values),
            "container_anon_shmem_peak_bytes": max(v["container_anon_shmem_bytes"] for v in values),
            "host_available_min_bytes": min(v["host_available_bytes"] for v in values),
            "storage_peak_bytes": max(v["storage_bytes"] for v in values),
            "native_exit_code": json.loads((ROOT / shard / "seal.json").read_text())["native_exit_code"],
            "owned_residual": json.loads((ROOT / shard / "status.json").read_text())["owned_residual"],
        })
    return records


def figures(rows, full_virtual, decision, output):
    output.mkdir(parents=True, exist_ok=True)
    # 1: placement in local polar coordinates with class and shard marker.
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for x in rows:
        theta = x["direction_index"] * math.pi / 8
        ax.scatter(theta, x["radius_m"], color=COLORS[x["virtual_class"]],
                   marker="o" if x["shard_id"] == "shard-A" else "s", s=100)
        ax.text(theta, x["radius_m"]+.025, str(x["rank"]))
    ax.set_ylim(0, .7); ax.set_title("Frozen eight R3 placements: circle A, square B")
    fig.savefig(output / "01-eight-placements.png", dpi=160); plt.close(fig)
    # 2–3: root drift, speed, tilt, and contact dynamics.
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for x in rows:
        path = ROOT / x["shard_id"] / "run/trace" / f"{x['pose_id']}.jsonl"
        trace = [json.loads(line) for line in path.read_text().splitlines()]
        ts = [v["physics_step"]*.005 for v in trace]
        axes[0].plot(ts, [v["state"]["xy_drift_m"] for v in trace], label=str(x["rank"]))
        axes[1].plot(ts, [v["state"]["linear_speed_mps"] for v in trace])
        axes[2].plot(ts, [v["state"]["tilt_rad"] for v in trace])
        axes2[0].plot(ts, [v["contact"]["contact_count"] for v in trace], label=str(x["rank"]))
        axes2[1].plot(ts, [v["contact"]["max_decoded_penetration_m"] for v in trace])
    for ax, label in zip(axes, ("XY drift (m)", "linear speed (m/s)", "tilt (rad)")): ax.set_ylabel(label)
    axes[0].legend(title="rank", ncol=8, fontsize=7)
    axes[-1].set_xlabel("simulation time (s)")
    fig.tight_layout(); fig.savefig(output / "02-root-drift-speed-tilt.png", dpi=160); plt.close(fig)
    axes2[0].set_ylabel("contact count"); axes2[0].legend(title="rank", ncol=8, fontsize=7)
    axes2[1].set_ylabel("decoded penetration (m)"); axes2[1].set_xlabel("simulation time (s)")
    fig2.tight_layout(); fig2.savefig(output / "03-contact-count-penetration.png", dpi=160); plt.close(fig2)
    # 4: virtual and settled centerline profiles, eligible only.
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, x in zip(axes.flat, rows):
        old = full_virtual[x["pose_id"]]
        profile = old["centerline_height_profile_m"]
        ax.plot(np.arange(len(profile))*.1+.05, profile, "--", label="virtual")
        if x["classification"] == "settled_anchor_eligible":
            new = x["settled_interaction"]["left_center_right_profiles_m"]["center"]
            ax.plot(np.arange(len(new))*.1+.05, new, label="settled")
        ax.set_title(f"#{x['rank']} {x['virtual_class']}\n{x['classification']}", fontsize=8)
    axes.flat[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(output / "04-virtual-vs-settled-dem-profile.png", dpi=160); plt.close(fig)
    # 5: class agreement matrix / NA ineligible.
    outcomes = list(CLASSES) + ["NA"]
    table = np.zeros((4, 5), dtype=int)
    for x in rows:
        i = CLASSES.index(x["virtual_class"])
        j = outcomes.index(x["settled_class"]) if x["settled_class"] in CLASSES else 4
        table[i, j] += 1
    fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(table, cmap="Blues")
    ax.set_xticks(range(5), outcomes, rotation=30, ha="right")
    ax.set_yticks(range(4), CLASSES); ax.set_xlabel("settled class"); ax.set_ylabel("virtual class")
    for i in range(4):
        for j in range(5): ax.text(j, i, str(table[i,j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output / "05-class-agreement.png", dpi=160); plt.close(fig)
    # 6: class-level physical supply.
    fig, ax = plt.subplots(figsize=(7, 4))
    successes = [sum(x["physical_supply"] for x in rows if x["virtual_class"] == cls) for cls in CLASSES]
    ax.bar(CLASSES, successes, color=[COLORS[c] for c in CLASSES]); ax.set_ylim(0, 2.2)
    ax.set_ylabel("eligible and class-matched / 2"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(output / "06-class-physical-supply.png", dpi=160); plt.close(fig)
    # 7: independent terrain/source family outcomes.
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, x in enumerate(rows):
        ax.scatter(i, CLASSES.index(x["virtual_class"]), s=130,
                   color="#249c67" if x["physical_supply"] else "#c14256",
                   marker="o" if x["shard_id"] == "shard-A" else "s")
    ax.set_xticks(range(8), [x["terrain"] for x in rows], rotation=35, ha="right")
    ax.set_yticks(range(4), CLASSES)
    ax.set_title("Green: physical supply; red: unavailable (first seed only)")
    fig.tight_layout(); fig.savefig(output / "07-source-family-support.png", dpi=160); plt.close(fig)
    # 8: explicit decision flow.
    fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off")
    labels = [
        ("8/8 protocol-valid eligibility", "PASS"),
        ("at least one supply per class", str(successes)),
        ("virtual/settled class review", str(Counter(x["comparison"] for x in rows))),
        ("decision", decision),
        ("next action", "review only; no second seed or Path"),
    ]
    for i, (label, value) in enumerate(labels):
        y = .9 - i*.19
        ax.text(.03, y, label, fontsize=10, bbox={"boxstyle":"round","facecolor":"#e8eef4"})
        ax.text(.44, y, value, fontsize=9)
    fig.savefig(output / "08-decision-flow.png", dpi=160); plt.close(fig)


def main():
    analysis = ROOT / "analysis"
    if analysis.exists():
        raise FileExistsError("analysis already exists; no overwrite")
    analysis.mkdir(parents=True)
    preflight = json.loads((ROOT / "preflight.json").read_text())
    primary = json.loads((ROOT / "primary-manifest.json").read_text())["poses"]
    if sha(ROOT / "primary-manifest.json") != preflight["primary_manifest_sha256"]:
        raise ValueError("primary manifest SHA drift")
    virtual = load_virtual()
    for shard in ("shard-A", "shard-B"):
        sr = ROOT / shard
        seal = json.loads((sr / "seal.json").read_text())
        status = json.loads((sr / "status.json").read_text())
        manifest = json.loads((sr / "pose-manifest.json").read_text())
        if (status.get("stage") != "full_complete" or status.get("owned_residual") is not False or
            seal["native_exit_code"] != 0 or seal["shard_id"] != shard or
            sha(sr / "seal.json") != status["seal_sha256"] or
            sha(sr / "sha-ledger.json") != seal["run_sha_ledger_sha256"]):
            raise ValueError("shard seal/status invalid: " + shard)
        verify_shard_artifacts(sr / "run", seal["pose_ids"], termination_aware=True)
        if seal["pose_ids"] != [x["pose_id"] for x in manifest["poses"]]:
            raise ValueError("shard pose order drift")
    result = []
    for row in primary:
        shard = "shard-A" if row["rank"] % 2 else "shard-B"
        if row["virtual_dem_sha256"] != virtual[row["pose_id"]]["dem_sha256"]:
            raise ValueError("virtual DEM SHA drift")
        result.append(analyze_pose(shard, row, virtual[row["pose_id"]]))
    if len(result) != 8 or len({x["pose_id"] for x in result}) != 8:
        raise ValueError("eight-pose analysis incomplete")
    decision = decide_pilot(result, protocol_valid=True)
    supply = {cls: {
        "measured": 2,
        "eligible": sum(x["classification"] == "settled_anchor_eligible" for x in result if x["virtual_class"] == cls),
        "physical_supply": sum(x["physical_supply"] for x in result if x["virtual_class"] == cls),
        "source_families": sorted({x["source_family"] for x in result if x["virtual_class"] == cls}),
    } for cls in CLASSES}
    write(analysis / "per-pose-results.json", {"poses": result})
    write(analysis / "class-support.json", supply)
    write(analysis / "resource-lifecycle.json", resource_summary())
    write(analysis / "final-decision.json", {
        "decision": decision, "protocol_valid_poses": 8,
        "physical_supply_by_class": {cls: supply[cls]["physical_supply"] for cls in CLASSES},
        "first_seed_only": True, "second_seed_executed": False,
        "candidate_paths_executed": 0, "reserve_executed": False,
    })
    figures(result, virtual, decision, analysis / "figures")
    lines = [
        "# R3 eight-pose candidate-free settling pilot",
        "",
        f"Decision: `{decision}`.",
        "",
        "| Rank | Class | Terrain | Radius | Eligibility | Settled DEM | Supply |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for x in result:
        lines.append(f"| {x['rank']} | {x['virtual_class']} | {x['terrain']} | {x['radius_m']:.2f} | "
                     f"{x['classification']} | {x['settled_class'] or 'NA'} | {x['physical_supply']} |")
    lines += ["", "Physical supply counts: " + json.dumps(
        {cls: supply[cls]["physical_supply"] for cls in CLASSES}, sort_keys=True),
        "", "First batch seed only. No reserve, second seed, Path, follower, Outcome, or learning.",
        "The 0.6 m corridor remains a comparison proxy, not a proven reach distance.",
    ]
    (analysis / "report.md").write_text("\n".join(lines) + "\n")
    ledger = [{
        "path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)
    } for p in sorted(ROOT.rglob("*")) if p.is_file() and p != analysis / "sha-ledger.json"]
    write(analysis / "sha-ledger.json", {"files": ledger, "count": len(ledger)})
    print(decision, {cls: supply[cls]["physical_supply"] for cls in CLASSES})


if __name__ == "__main__":
    main()
