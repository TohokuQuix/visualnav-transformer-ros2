#!/usr/bin/env bash
set -euo pipefail
cd /tmp/eatnav-vnt
ROOT=/tmp/eatnav-quality-identifiability-gate/feasible-set/duplicate-recovery
for index in 13 42; do
  folder=$(printf '%s/smoke/state-%03d' "$ROOT" "$index")
  if [[ "$index" == 42 ]]; then
    scripts/eat_nav_duplicate_recovery_smoke_stage.sh "$index" nominal anchor
    docker exec isaac-lab-onix python3 /tmp/eat_nav_duplicate_recovery_smoke_paths.py --state-folder "$folder" --nominal-map "$folder/nominal-map-path.json"
  fi
  for repeat in 0 1 2; do
    scripts/eat_nav_duplicate_recovery_smoke_stage.sh "$index" nominal live "$repeat"
  done
  for name in slight_left slight_right strong_left strong_right; do
    scripts/eat_nav_duplicate_recovery_smoke_stage.sh "$index" "$name" live
  done
done
printf '%s SMOKE_ALL_RUNS_COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a docs/quality-identifiability-gate/feasible-set/duplicate-recovery/runner.log
