"""SHA ledger for only small M11-M13 addendum artifacts and code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-addendum"
ARTIFACTS = (
    "prospective-evaluation-gates.yaml",
    "baseline-and-ablation-contract.json",
    "multiobjective-evaluation-contract.json",
    "future-collection-logging-schema.json",
    "logging-parity-gate.yaml",
    "final-decision.json",
    "m0-m13-consistency-audit.json",
    "analysis-report-addendum.md",
    "fig-01-conflict-pareto.svg",
    "fig-02-phase-heads.svg",
    "fig-03-evaluation-flow.svg",
    "fig-04-logging-timeline.svg",
    "fig-05-decision-flow.svg",
)
CODE = (
    "scripts/eat_nav_evaluator_multihead_addendum.py",
    "scripts/eat_nav_evaluator_multihead_logging.py",
    "scripts/eat_nav_evaluator_multihead_finalize.py",
    "scripts/eat_nav_evaluator_multihead_addendum_figures.py",
    "scripts/eat_nav_evaluator_multihead_addendum_ledger.py",
    "tests/eat_nav/test_evaluator_multihead_addendum.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [OUT / name for name in ARTIFACTS] + [ROOT / name for name in CODE]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate paths")
    files = {str(path.relative_to(ROOT)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
             for path in paths}
    report = {"schema_version": 1, "base_commit": "74bf5029c3818a7335f6b1396cc10c725b84b2a2",
              "branch": "exp/eatnav-evaluator-multihead-addendum", "artifact_count_excluding_self": len(ARTIFACTS),
              "code_test_count": len(CODE), "files": files,
              "prior_m0_m10_ledger_sha256": sha(ROOT / "docs/quality-identifiability-gate/evaluator-multihead-freeze/artifact-sha-ledger.json"),
              "exclusions": ["raw traces", "checkpoints", "live logs", "new rollout artifacts"]}
    path = OUT / "artifact-sha-ledger.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, meta in json.loads(path.read_text())["files"].items():
        if sha(ROOT / name) != meta["sha256"]:
            raise ValueError(f"ledger mismatch {name}")
    print(f"verified {len(files)} SHA entries")


if __name__ == "__main__":
    main()
