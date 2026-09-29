#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
HOST_ROOT=/tmp/eatnav-vnt/docs/quality-identifiability-gate/feasible-set/direct-episode-start
CONTAINER_ROOT=/tmp/eatnav-quality-identifiability-gate/feasible-set/direct-episode-start
mkdir -p "$HOST_ROOT/runs" "$HOST_ROOT/follower-logs"
status() {
  /usr/bin/python3 - "$HOST_ROOT/status.json" "$1" "$2" <<'PY'
import json,os,sys,tempfile,time
path,stage,state=sys.argv[1:]
obj={'stage':stage,'state':state,'updated_at_unix':time.time(),'candidate_rollout_started':False,'formal_rollout_started':False}
fd,temp=tempfile.mkstemp(dir=os.path.dirname(path),prefix='status-')
with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2);f.write('\n')
os.replace(temp,path)
PY
}
run_case() {
  local index=$1 mode=$2 stage=$3 source=$4
  local out="$CONTAINER_ROOT/runs/state-$(printf '%03d' "$index")/$mode"
  local host_out="$HOST_ROOT/runs/state-$(printf '%03d' "$index")/$mode"
  local episode="$CONTAINER_ROOT/inputs/state-$(printf '%03d' "$index")"
  mkdir -p "$host_out"
  docker exec isaac-lab-onix mkdir -p "$out"
  status "state-${index}-${mode}-${stage}" running
  printf '%s BEGIN state=%s mode=%s stage=%s\n' "$(date --iso-8601=seconds)" "$index" "$mode" "$stage" >> "$HOST_ROOT/runner.log"
  set +e
  docker exec isaac-lab-onix bash -lc "export PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_direct_episode_run.py --headless --stage $stage --replay-mode $mode --repeat-id 0 --manifest /tmp/eatnav-extended/output/manifest.json --prefix $source/prefix-actions.npz --other-env-actions $source/suffix-actions.npz --logged-anchor $source/logged-anchor.json --state /tmp/eatnav-quality-identifiability-gate/feasible-set/state-selection.json --base-path $source/nominal-base-path.json --fixed-map-path $source/nominal-map-path.json --config $source/config.yaml --episode-input $episode/episode-local-actions.npz --episode-freeze $episode/episode-input-freeze.json --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --rl-checkpoint /workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt --output $out" >> "$HOST_ROOT/runner.log" 2>&1
  local code=$?
  set -e
  if [[ $code -eq 0 ]] && ! docker exec isaac-lab-onix test -f "$out/$stage-0.marker.json"; then code=97; fi
  docker cp "isaac-lab-onix:$out/." "$host_out/" >> "$HOST_ROOT/runner.log" 2>&1 || code=98
  if [[ "$stage" == anchor || "$stage" == anchor_live ]] && [[ $code -eq 0 ]]; then
    docker cp "isaac-lab-onix:$source/nominal-map-path.json" "$HOST_ROOT/runs/state-$(printf '%03d' "$index")/nominal-map-path.json" >> "$HOST_ROOT/runner.log" 2>&1 || code=98
  fi
  printf '%s EXIT state=%s mode=%s stage=%s code=%s\n' "$(date --iso-8601=seconds)" "$index" "$mode" "$stage" "$code" >> "$HOST_ROOT/runner.log"
  if [[ $code -ne 0 ]]; then status "state-${index}-${mode}-${stage}" failed;exit "$code";fi
}
compare() {
  local index=$1 case="state-$(printf '%03d' "$1")"
  status "$case-compare" running
  /usr/bin/python3 /tmp/eatnav-vnt/scripts/eat_nav_direct_episode_compare.py \
    --full "$HOST_ROOT/runs/$case/full" --direct "$HOST_ROOT/runs/$case/direct" \
    --config "$HOST_ROOT/diagnostic-config.yaml" --output "$HOST_ROOT/runs/$case/comparison.json" >> "$HOST_ROOT/runner.log" 2>&1
  set +e
  /usr/bin/python3 - "$HOST_ROOT/runs/$case/comparison.json" <<'PY'
import json,sys
result=json.load(open(sys.argv[1]))
if not result['pass']:raise SystemExit(42)
PY
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then status "$case-compare" gate_failed;exit "$code";fi
  status "$case-compare" passed
}
trap 'if [[ -n "${BRIDGE_PID:-}" ]]; then kill "$BRIDGE_PID" 2>/dev/null || true; fi' EXIT
status bridge_start running
/usr/bin/python3 /tmp/eatnav-vnt/scripts/eat_nav_phase_c_ros_bridge.py \
  --host 0.0.0.0 \
  --follower /home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node \
  --params /home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml \
  --diagnostic-dir "$HOST_ROOT/follower-logs" >> "$HOST_ROOT/bridge.log" 2>&1 &
BRIDGE_PID=$!
sleep 3
kill -0 "$BRIDGE_PID"
SHORT=/tmp/eatnav-quality-identifiability-gate/feasible-set/duplicate-recovery/smoke/state-042
if ! /usr/bin/python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"]' "$HOST_ROOT/runs/state-042/comparison.json" 2>/dev/null; then
  run_case 42 full live "$SHORT"
  run_case 42 direct live "$SHORT"
  compare 42
else
  printf '%s RESUME_SKIP state42 validated comparison\n' "$(date --iso-8601=seconds)" >> "$HOST_ROOT/runner.log"
fi
BOXES=/tmp/eatnav-quality-identifiability-gate/feasible-set/seven-state-pilot/pilot-inputs/state-007
run_case 7 full live "$BOXES"
run_case 7 direct live "$BOXES"
compare 7
status direct_episode_start_gate passed
