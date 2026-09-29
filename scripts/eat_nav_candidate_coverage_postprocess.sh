#!/usr/bin/env bash
# Detached, train-only postprocessing; never starts validation or another rollout.
set -euo pipefail
ROOT=/tmp/eatnav-candidate-coverage
WORKTREE=/tmp/eatnav-vnt-candidate-coverage
CONTAINER=isaac-lab-onix
LOG="$ROOT/pilot/postprocess.log"
mkdir -p "$ROOT/analysis"
exec >> "$LOG" 2>&1
printf 'POSTPROCESS_LAUNCH_UNIX %s\n' "$(date +%s)"
while true; do
  state=$(/usr/bin/python3 -c 'import json;print(json.load(open("/tmp/eatnav-candidate-coverage/pilot/status.json"))["state"])')
  case "$state" in
    raw_runs_complete) break ;;
    stopped|budget_stop)
      printf 'POSTPROCESS_ABORT_PILOT_STATE %s\n' "$state"
      exit 2 ;;
  esac
  sleep 30
done
cd "$WORKTREE"
if [[ ! -f "$ROOT/analysis/oracle-summary.json" ]]; then
  PYTHONPATH=src:.:scripts /usr/bin/python3 -u scripts/eat_nav_candidate_coverage_analyze.py
fi
if [[ ! -f "$ROOT/analysis/terrain-coverage.png" ]]; then
  PYTHONPATH=src:.:scripts /usr/bin/python3 -u scripts/eat_nav_candidate_coverage_plot.py
fi
# Only after all oracle results: copy exact evaluator code and train-only inputs.
docker cp src/visualnav_transformer/eat_nav/candidate_coverage_evaluator.py "$CONTAINER:$ROOT/baseline-worktree-8e8f4fd/src/visualnav_transformer/eat_nav/candidate_coverage_evaluator.py"
docker cp scripts/eat_nav_candidate_coverage_evaluator.py "$CONTAINER:$ROOT/baseline-worktree-8e8f4fd/scripts/eat_nav_candidate_coverage_evaluator.py"
docker exec "$CONTAINER" mkdir -p "$ROOT/analysis"
for name in oracle-outcomes.json oracle-summary.json; do
  docker cp "$ROOT/analysis/$name" "$CONTAINER:$ROOT/analysis/$name"
done
docker cp "$ROOT/pilot/status.json" "$CONTAINER:$ROOT/pilot/status.json"
docker cp "$ROOT/pilot/oracle-epsilon.json" "$CONTAINER:$ROOT/pilot/oracle-epsilon.json"
if [[ ! -f "$ROOT/analysis/evaluator-ranking.json" ]]; then
  docker exec "$CONTAINER" bash -lc "export PYTHONPATH=$ROOT/baseline-worktree-8e8f4fd:$ROOT/baseline-worktree-8e8f4fd/src:$ROOT/baseline-worktree-8e8f4fd/scripts:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh $ROOT/baseline-worktree-8e8f4fd/scripts/eat_nav_candidate_coverage_evaluator.py"
  docker cp "$CONTAINER:$ROOT/analysis/evaluator-ranking.json" "$ROOT/analysis/evaluator-ranking.json"
  docker cp "$CONTAINER:$ROOT/analysis/evaluator-report.md" "$ROOT/analysis/evaluator-report.md"
fi
/usr/bin/python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,time
root=Path('/tmp/eatnav-candidate-coverage')
files=['oracle-outcomes.json','oracle-summary.json','pilot-report.md','evaluator-ranking.json','evaluator-report.md','recall-at-k.png','k-vs-oracle-best.png','path-vs-command.png','command-vs-outcome.png','nomad-vs-bank-vs-hybrid.png','terrain-coverage.png']
sha={name:hashlib.sha256((root/'analysis'/name).read_bytes()).hexdigest() for name in files}
marker={'stage':'train_pilot_postprocessing_complete','unix_sec':time.time(),'outputs_sha256':sha,'formal_validation_started':False,'retraining_started':False,'real_robot_started':False}
target=root/'analysis/postprocess-complete.json';tmp=target.with_name(target.name+'.tmp');tmp.write_text(json.dumps(marker,indent=2,sort_keys=True)+'\n');os.replace(tmp,target)
PY
printf 'POSTPROCESS_COMPLETE_UNIX %s\n' "$(date +%s)"
