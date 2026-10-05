"""Hash only small multi-head design artifacts and source/test files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-freeze"
ARTIFACTS = (
    "multihead-analysis-config.yaml", "m0-provenance.json", "label-generation-audit.json",
    "multihead-labels.jsonl.gz", "pairwise-labels.json", "data-sufficiency-review.json",
    "model-output-contract.yaml", "loss-and-normalization-contract.yaml",
    "inference-selection-contract.yaml", "portable-evaluator-input-contract.json",
    "optional-latent-input-contract.json",
    "future-training-and-evaluation-gates.md", "multihead-design-review.md",
    "fig-01-observed-label-distributions.svg", "fig-02-pairwise-labels.svg",
    "fig-03-reward-goal-conflicts.svg", "fig-04-head-support.svg",
)
CODE = (
    "scripts/eat_nav_evaluator_multihead.py", "scripts/eat_nav_evaluator_multihead_audit.py",
    "scripts/eat_nav_evaluator_multihead_figures.py", "scripts/eat_nav_evaluator_multihead_ledger.py",
    "tests/eat_nav/test_evaluator_multihead_labels.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [OUT / name for name in ARTIFACTS] + [ROOT / name for name in CODE]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate paths")
    files = {str(path.relative_to(ROOT)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
             for path in paths}
    report = {"schema_version": 1, "base_commit": "51d9bf201f76fd8e3b54d3d9496ca4c3429cbb3a",
              "branch": "exp/eatnav-evaluator-multihead", "artifacts_excluding_self": len(ARTIFACTS),
              "code_test_files": len(CODE), "files": files,
              "exclusions": ["raw traces", "checkpoints", "live logs", "new rollout artifacts"]}
    path = OUT / "artifact-sha-ledger.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, meta in json.loads(path.read_text())["files"].items():
        if sha(ROOT / name) != meta["sha256"]:
            raise ValueError(f"ledger mismatch {name}")
    print(f"verified {len(files)} SHA entries")


if __name__ == "__main__":
    main()
