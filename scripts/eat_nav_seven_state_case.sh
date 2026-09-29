#!/usr/bin/env bash
set -euo pipefail
INDEX=${1:?state index}
NAME=${2:?maneuver}
STAGE=${3:?anchor or live}
REPEAT=${4:?repeat id}
ROOT=/tmp/eatnav-quality-identifiability-gate/feasible-set/seven-state-pilot
INPUT=$(printf '%s/pilot-inputs/state-%03d' "$ROOT" "$INDEX")
OUT=$(printf '%s/pilot-runs/state-%03d/%s' "$ROOT" "$INDEX" "$NAME")
HOST=/tmp/eatnav-vnt/docs/quality-identifiability-gate/feasible-set/seven-state-pilot
HOST_CASE=$(printf '%s/pilot-runs/state-%03d/%s' "$HOST" "$INDEX" "$NAME")
mkdir -p "$HOST_CASE"
docker exec isaac-lab-onix mkdir -p "$OUT"
printf '%s PILOT_CASE_BEGIN state=%s name=%s stage=%s repeat=%s\n' "$(date --iso-8601=seconds)" "$INDEX" "$NAME" "$STAGE" "$REPEAT" >> "$HOST/runner.log"
set +e
docker exec isaac-lab-onix bash -lc "export PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_duplicate_recovery_run.py --headless --stage $STAGE --repeat-id $REPEAT --manifest /tmp/eatnav-extended/output/manifest.json --prefix $INPUT/prefix-actions.npz --other-env-actions $INPUT/suffix-actions.npz --logged-anchor $INPUT/logged-anchor.json --state /tmp/eatnav-quality-identifiability-gate/feasible-set/state-selection.json --base-path $INPUT/$NAME-base-path.json --fixed-map-path $INPUT/$NAME-map-path.json --config $INPUT/config.yaml --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output $OUT" >> "$HOST/runner.log" 2>&1
CODE=$?
if [[ "$CODE" -eq 0 ]] && ! docker exec isaac-lab-onix test -f "$OUT/$STAGE-$REPEAT.marker.json"; then CODE=97; fi
set -e
docker cp "isaac-lab-onix:$OUT/." "$HOST_CASE/" >> "$HOST/runner.log" 2>&1 || CODE=98
if [[ "$STAGE" == anchor && "$CODE" -eq 0 ]]; then
  docker cp "isaac-lab-onix:$INPUT/nominal-map-path.json" "$HOST_CASE/nominal-map-path.json" >> "$HOST/runner.log" 2>&1 || CODE=98
fi
printf '%s PILOT_CASE_EXIT state=%s name=%s stage=%s repeat=%s code=%s\n' "$(date --iso-8601=seconds)" "$INDEX" "$NAME" "$STAGE" "$REPEAT" "$CODE" >> "$HOST/runner.log"
exit "$CODE"
