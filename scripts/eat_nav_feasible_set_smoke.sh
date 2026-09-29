#!/usr/bin/env bash
set -uo pipefail
docker exec isaac-lab-onix bash -lc 'export PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_feasible_set_runner.py --headless --manifest /tmp/eatnav-extended/output/manifest.json --selection /tmp/eatnav-quality-identifiability-gate/feasible-set/state-selection.json --maneuvers /tmp/eatnav-quality-identifiability-gate/feasible-set/maneuver-bank.npz --maneuver-bank /tmp/eatnav-quality-identifiability-gate/feasible-set/maneuver-bank.json --precheck /tmp/eatnav-quality-identifiability-gate/feasible-set/command-distinctness.json --spec /tmp/eatnav-train/feasible-set-config.yaml --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output /tmp/eatnav-quality-identifiability-gate/feasible-set --split train --state-index 19'
code=$?
echo SERIAL_SMOKE_EXIT_CODE=$code
exit "$code"
