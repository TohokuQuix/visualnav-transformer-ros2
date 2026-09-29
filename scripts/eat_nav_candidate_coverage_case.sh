#!/usr/bin/env bash
set -euo pipefail
RUN_ID=${1:?pilot run ID}
STATE_INDEX=${2:?state index}
CANDIDATE_ID=${3:?candidate ID}
STAGE=${4:?anchor or live}
[[ "$RUN_ID" =~ ^pilot-[0-9]{3}$ ]] || exit 64
[[ "$STATE_INDEX" =~ ^[0-6]$ ]] || exit 64
[[ "$CANDIDATE_ID" =~ ^[a-z0-9_-]+$ ]] || exit 64
[[ "$STAGE" == anchor || "$STAGE" == live ]] || exit 64
HOST_ROOT=/tmp/eatnav-candidate-coverage/pilot
CONTAINER_ROOT=/tmp/eatnav-candidate-coverage
INPUT=$(printf '%s/pilot-inputs/state-%03d' "$CONTAINER_ROOT" "$STATE_INDEX")
OUT="$CONTAINER_ROOT/pilot/runs/$RUN_ID"
HOST_OUT="$HOST_ROOT/runs/$RUN_ID"
mkdir -p "$HOST_OUT"
exec >> "$HOST_OUT/runner.log" 2>&1
printf 'RUN_BEGIN_UNIX %s id=%s state=%s candidate=%s stage=%s\n' "$(date +%s)" "$RUN_ID" "$STATE_INDEX" "$CANDIDATE_ID" "$STAGE"
docker exec isaac-lab-onix mkdir -p "$OUT"
set +e
docker exec isaac-lab-onix bash -lc "export PYTHONPATH=/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/scripts/eat_nav_duplicate_recovery_run.py --headless --stage $STAGE --repeat-id 0 --manifest /tmp/eatnav-extended/output/manifest.json --prefix $INPUT/prefix-actions.npz --other-env-actions $INPUT/suffix-actions.npz --logged-anchor $INPUT/logged-anchor.json --state $CONTAINER_ROOT/pilot-inputs/selected-states.json --base-path $INPUT/$CANDIDATE_ID-base-path.json --fixed-map-path $INPUT/$CANDIDATE_ID-map-path.json --config $INPUT/config.yaml --reference-config /tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output $OUT"
CODE=$?
set -e
if [[ "$CODE" -eq 0 ]] && ! docker exec isaac-lab-onix test -f "$OUT/$STAGE-0.marker.json"; then CODE=97; fi
docker cp "isaac-lab-onix:$OUT/." "$HOST_OUT/" || CODE=98
if [[ "$STAGE" == anchor && "$CODE" -eq 0 ]]; then
  docker cp "isaac-lab-onix:$INPUT/$CANDIDATE_ID-map-path.json" "$HOST_OUT/fixed-map-path.json" || CODE=98
fi
printf 'RUN_EXIT_UNIX %s id=%s code=%s\n' "$(date +%s)" "$RUN_ID" "$CODE"
exit "$CODE"
