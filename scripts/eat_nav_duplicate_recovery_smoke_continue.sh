#!/usr/bin/env bash
set -euo pipefail
cd /tmp/eatnav-vnt
HOST_OUT=docs/quality-identifiability-gate/feasible-set/duplicate-recovery
ROOT=/tmp/eatnav-quality-identifiability-gate/feasible-set/duplicate-recovery
while ! rg -q 'SMOKE_EXIT 19 strong_right live code=0' "$HOST_OUT/runner.log"; do
  if rg -q 'SMOKE_EXIT 19 (slight_left|slight_right|strong_left|strong_right) live code=[1-9]' "$HOST_OUT/runner.log"; then exit 1; fi
  sleep 5
done
docker cp scripts/eat_nav_duplicate_recovery_smoke_paths.py isaac-lab-onix:/tmp/eat_nav_duplicate_recovery_smoke_paths.py
for index in 13 42; do
  folder=$(printf '%s/smoke/state-%03d' "$ROOT" "$index")
  scripts/eat_nav_duplicate_recovery_smoke_stage.sh "$index" nominal anchor
  docker exec isaac-lab-onix python3 /tmp/eat_nav_duplicate_recovery_smoke_paths.py --state-folder "$folder" --nominal-map "$folder/nominal-map-path.json"
  for repeat in 0 1 2; do
    scripts/eat_nav_duplicate_recovery_smoke_stage.sh "$index" nominal live "$repeat"
  done
  for name in slight_left slight_right strong_left strong_right; do
    scripts/eat_nav_duplicate_recovery_smoke_stage.sh "$index" "$name" live
  done
done
printf '%s SMOKE_ALL_RUNS_COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "$HOST_OUT/runner.log"
