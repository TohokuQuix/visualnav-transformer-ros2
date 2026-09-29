#!/usr/bin/env bash
set -euo pipefail
HOST_OUT=/tmp/eatnav-vnt/docs/quality-identifiability-gate/feasible-set/seven-state-pilot
CONTAINER_ROOT=/tmp/eatnav-quality-identifiability-gate/feasible-set/duplicate-recovery
FOLDER=$CONTAINER_ROOT/smoke/state-042
OUT=/tmp/eatnav-quality-identifiability-gate/feasible-set/seven-state-pilot/state-042/diagnostic
mkdir -p "$HOST_OUT"
printf '%s DIAGNOSTIC_BEGIN\n' "$(date --iso-8601=seconds)" >> "$HOST_OUT/runner.log"
docker exec isaac-lab-onix mkdir -p "$OUT"
set +e
docker exec isaac-lab-onix bash -lc "export PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_duplicate_recovery_run.py --headless --stage live --repeat-id 0 --manifest /tmp/eatnav-extended/output/manifest.json --prefix $FOLDER/prefix-actions.npz --other-env-actions $FOLDER/suffix-actions.npz --logged-anchor $FOLDER/logged-anchor.json --state $CONTAINER_ROOT/../state-selection.json --base-path $FOLDER/nominal-base-path.json --fixed-map-path $FOLDER/nominal-map-path.json --config $FOLDER/config.yaml --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output $OUT" >> "$HOST_OUT/runner.log" 2>&1
CODE=$?
if [[ "$CODE" -eq 0 ]] && ! docker exec isaac-lab-onix test -f "$OUT/live-0.marker.json"; then CODE=97; fi
set -e
printf '%s DIAGNOSTIC_EXIT code=%s\n' "$(date --iso-8601=seconds)" "$CODE" >> "$HOST_OUT/runner.log"
python3 - "$HOST_OUT/status.json" "$CODE" <<'PY'
import json,os,sys,tempfile,time
p=sys.argv[1]; code=int(sys.argv[2]); d={'stage':'state42_diagnostic','state':'complete' if code==0 else 'failed','exit_code':code,'updated_at':time.time(),'pilot_allowed':False}
fd,t=tempfile.mkstemp(dir=os.path.dirname(p)); os.write(fd,(json.dumps(d,indent=2)+'\n').encode());os.close(fd);os.replace(t,p)
PY
exit "$CODE"
