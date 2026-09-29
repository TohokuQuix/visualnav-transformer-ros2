#!/usr/bin/env bash
set -euo pipefail
STAGE=${1:?stage required}
REPEAT=${2:-0}
HOST_OUT=/tmp/eatnav-vnt/docs/quality-identifiability-gate/feasible-set/duplicate-recovery
OUT=/tmp/eatnav-quality-identifiability-gate/feasible-set/duplicate-recovery
BASE=$OUT/baseline
ROOT=/tmp/eatnav-train
EXTRA=()
if [[ "$STAGE" == recorded ]]; then
  EXTRA=(--recorded-command "$OUT/recorded-command.npz")
fi
printf '%s RECOVERY_BEGIN %s %s\n' "$(date --iso-8601=seconds)" "$STAGE" "$REPEAT" | tee -a "$HOST_OUT/runner.log"
set +e
docker exec isaac-lab-onix bash -lc "export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh $ROOT/eat_nav_duplicate_recovery_run.py --headless --stage $STAGE --repeat-id $REPEAT --manifest /tmp/eatnav-extended/output/manifest.json --prefix $BASE/prefix-actions.npz --other-env-actions $OUT/suffix-actions.npz --logged-anchor $BASE/logged-anchor.json --state $OUT/../state-selection.json --base-path $BASE/nominal-path-base.json --fixed-map-path $OUT/fixed-map-path.json --config $OUT/duplicate-recovery-config.yaml --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output $OUT ${EXTRA[*]}" >> "$HOST_OUT/runner.log" 2>&1
CODE=$?
if [[ "$CODE" -eq 0 ]] && ! docker exec isaac-lab-onix test -f "$OUT/$STAGE-$REPEAT.marker.json"; then
  CODE=97
  printf "%s RECOVERY_MISSING_MARKER %s %s\n" "$(date --iso-8601=seconds)" "$STAGE" "$REPEAT" | tee -a "$HOST_OUT/runner.log"
fi
set -e
printf '%s RECOVERY_EXIT %s %s code=%s\n' "$(date --iso-8601=seconds)" "$STAGE" "$REPEAT" "$CODE" | tee -a "$HOST_OUT/runner.log"
exit "$CODE"
