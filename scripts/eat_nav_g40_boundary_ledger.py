"""Create and verify the small-artifact SHA ledger for the G40 boundary review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality-identifiability-gate/g40-boundary-freeze"
ARTIFACTS = (
    "source-provenance.json",
    "boundary-analysis-config.yaml",
    "critic-observation-schema.json",
    "critic-offline-replay-equivalence.json",
    "terminal-command-contract.json",
    "terminal-value-per-candidate.jsonl.gz",
    "bootstrap-teacher-comparison.json",
    "bootstrap-rank-stability.json",
    "portable-evaluator-input-contract.json",
    "optional-latent-input-contract.json",
    "task-head-and-hard-constraint-contract.json",
    "safety-head-disabled-contract.json",
    "next-collection-logging-schema.json",
    "teacher-contract.yaml",
    "freeze-decision.json",
    "boundary-and-input-review-report.md",
    *(f"fig-{i:02d}-{name}.svg" for i, name in enumerate((
        "terminal-value-conditions", "g40-ranks-by-group", "rank-correlations",
        "top1-top3-change", "direct-sign-reversals", "reward-terminal-contribution",
        "portable-input", "task-vs-hard", "next-logging-timeline"), 1)),
)
CODE = (
    "scripts/eat_nav_g40_boundary_audit.py",
    "scripts/eat_nav_g40_boundary_schema.py",
    "scripts/eat_nav_g40_boundary_replay.py",
    "scripts/eat_nav_g40_boundary_analyze.py",
    "scripts/eat_nav_g40_boundary_figures.py",
    "scripts/eat_nav_g40_boundary_ledger.py",
    "tests/eat_nav/test_g40_boundary_freeze.py",
    "tests/eat_nav/test_g40_boundary_rank.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries() -> dict[str, dict[str, int | str]]:
    paths = [OUT / name for name in ARTIFACTS] + [ROOT / name for name in CODE]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate ledger paths")
    return {str(path.relative_to(ROOT)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in paths}


def main() -> None:
    provenance = json.loads((OUT / "source-provenance.json").read_text())
    config_key = str((OUT / "boundary-analysis-config.yaml").relative_to(ROOT))
    if sha256(OUT / "boundary-analysis-config.yaml") != provenance["source_files_sha256"][config_key]:
        raise ValueError("B0 analysis config changed")
    report = {"schema_version": 1, "scope": "G40 boundary offline review, no raw trace or checkpoint",
              "artifact_count_excluding_self": len(ARTIFACTS), "figure_count": 9,
              "source_base_commit": "87d1ef3e1e617bf452949a7f48658718af51723e",
              "files": entries()}
    (OUT / "artifact-sha-ledger.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    actual = json.loads((OUT / "artifact-sha-ledger.json").read_text())
    if actual["files"] != entries():
        raise ValueError("ledger verification failed")
    print(f"SHA ledger verified: {len(actual['files'])} files")


if __name__ == "__main__":
    main()
