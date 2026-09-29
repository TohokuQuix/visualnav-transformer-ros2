#!/usr/bin/env bash
set -euo pipefail
HOST=/tmp/eatnav-vnt/docs/quality-identifiability-gate/feasible-set/direct-episode-start/cost-selection-v2
CONTAINER=/tmp/eatnav-quality-identifiability-gate/feasible-set/direct-episode-start/cost-selection-v2
mkdir -p "$HOST"
python3 - "$HOST/status.json" <<'PY'
import json,os,sys,tempfile,time
p=sys.argv[1];obj={'stage':'complete_only_train_selection','state':'running','updated_at_unix':time.time(),'rollout_started':False}
fd,t=tempfile.mkstemp(dir=os.path.dirname(p));os.write(fd,(json.dumps(obj,indent=2)+'\n').encode());os.close(fd);os.replace(t,p)
PY
docker exec isaac-lab-onix mkdir -p "$CONTAINER"
set +e
docker exec isaac-lab-onix bash -lc 'export PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_direct_episode_v2_select.py --manifest /tmp/eatnav-extended/output/manifest.json --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --gate-config /tmp/eatnav-quality-identifiability-gate/feasible-set/direct-episode-start/diagnostic-config.yaml --selection-source /tmp/eatnav-quality-identifiability-gate/feasible-set/state-selection.json --output /tmp/eatnav-quality-identifiability-gate/feasible-set/direct-episode-start/cost-selection-v2' > "$HOST/runner.log" 2>&1
CODE=$?
set -e
docker cp "isaac-lab-onix:$CONTAINER/." "$HOST/" >> "$HOST/runner.log" 2>&1 || CODE=98
if [[ "$CODE" -ne 0 ]]; then
  python3 - "$HOST/status.json" "$CODE" <<'PY'
import json,os,sys,tempfile,time
p=sys.argv[1];obj={'stage':'complete_only_train_selection','state':'failed','exit_code':int(sys.argv[2]),'updated_at_unix':time.time(),'rollout_started':False}
fd,t=tempfile.mkstemp(dir=os.path.dirname(p));os.write(fd,(json.dumps(obj,indent=2)+'\n').encode());os.close(fd);os.replace(t,p)
PY
  exit "$CODE"
fi
python3 - "$HOST/selection-v2.json" "$HOST/status.json" <<'PY'
import hashlib,json,os,sys,tempfile,time
sel,status=sys.argv[1:];d=json.load(open(sel))
assert d['rollout_started'] is False and d['approval_required_before_rollout'] is True
obj={'stage':'selection_v2_frozen','state':'awaiting_approval','selected_count':len(d['selected']),
     'selection_sha256':hashlib.sha256(open(sel,'rb').read()).hexdigest(),
     'updated_at_unix':time.time(),'rollout_started':False}
fd,t=tempfile.mkstemp(dir=os.path.dirname(status));os.write(fd,(json.dumps(obj,indent=2)+'\n').encode());os.close(fd);os.replace(t,status)
PY
