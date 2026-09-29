#!/usr/bin/env bash
set -euo pipefail
INDEX=${1:?state index required}
NAME=${2:?maneuver name required}
STAGE=${3:-live}
REPEAT=${4:-0}
ROOT=/tmp/eatnav-quality-identifiability-gate/feasible-set/duplicate-recovery
FOLDER=$(printf '%s/smoke/state-%03d' "$ROOT" "$INDEX")
CASE="$FOLDER/$NAME"
HOST_OUT=/tmp/eatnav-vnt/docs/quality-identifiability-gate/feasible-set/duplicate-recovery
mkdir -p "$HOST_OUT"
printf '%s SMOKE_BEGIN %s %s %s\n' "$(date --iso-8601=seconds)" "$INDEX" "$NAME" "$STAGE" | tee -a "$HOST_OUT/runner.log"
set +e
docker exec isaac-lab-onix bash -lc "export PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_duplicate_recovery_run.py --headless --stage $STAGE --repeat-id $REPEAT --manifest /tmp/eatnav-extended/output/manifest.json --prefix $FOLDER/prefix-actions.npz --other-env-actions $FOLDER/suffix-actions.npz --logged-anchor $FOLDER/logged-anchor.json --state $ROOT/../state-selection.json --base-path $FOLDER/$NAME-base-path.json --fixed-map-path $FOLDER/$NAME-map-path.json --config $FOLDER/config.yaml --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output $CASE" >> "$HOST_OUT/runner.log" 2>&1
CODE=$?
if [[ "$CODE" -eq 0 ]] && ! docker exec isaac-lab-onix test -f "$CASE/$STAGE-$REPEAT.marker.json"; then
  CODE=97
  printf "%s SMOKE_MISSING_MARKER %s %s %s\n" "$(date --iso-8601=seconds)" "$INDEX" "$NAME" "$STAGE" | tee -a "$HOST_OUT/runner.log"
fi
set -e
printf '%s SMOKE_EXIT %s %s %s code=%s\n' "$(date --iso-8601=seconds)" "$INDEX" "$NAME" "$STAGE" "$CODE" | tee -a "$HOST_OUT/runner.log"
exit "$CODE"
