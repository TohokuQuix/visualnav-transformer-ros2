#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/tmp/eatnav-vnt
CONTAINER=isaac-lab-onix
ROOT=/tmp/eatnav-train
OUT=/tmp/eatnav-quality-identifiability-gate/feasible-set
mkdir -p /tmp/eatnav-feasible-set
for file in scripts/eat_nav_feasible_set_freeze.py scripts/eat_nav_feasible_set_sim.py scripts/eat_nav_phase_c_ros_bridge.py; do
  docker cp "$WORKTREE/$file" "$CONTAINER:$ROOT/$(basename "$file")"
done
docker cp "$WORKTREE/src/visualnav_transformer/eat_nav/feasible_set.py" "$CONTAINER:$ROOT/src/visualnav_transformer/eat_nav/feasible_set.py"
docker cp "$WORKTREE/docs/quality-identifiability-gate/feasible-set/feasible-set-config.yaml" "$CONTAINER:$ROOT/feasible-set-config.yaml"
docker exec "$CONTAINER" bash -lc "export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh $ROOT/eat_nav_feasible_set_freeze.py --manifest /tmp/eatnav-extended/output/manifest.json --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --reference /tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --spec $ROOT/feasible-set-config.yaml --source-root $ROOT --output $OUT"
docker exec "$CONTAINER" bash -lc "export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh $ROOT/eat_nav_feasible_set_sim.py --headless --manifest /tmp/eatnav-extended/output/manifest.json --selection $OUT/state-selection.json --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output $OUT"
docker cp "$CONTAINER:$OUT/." "$WORKTREE/docs/quality-identifiability-gate/feasible-set/"
