#!/usr/bin/env bash
set -eu
R6_ARTIFACT_ROOT=/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030
R6_RESTORED_WORKTREE=$(/usr/bin/python3 - "$R6_ARTIFACT_ROOT/restart-backup/latest.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());p=Path(r['worktree'])
if r.get('state')!='verified' or not (p/'.git').is_dir():raise SystemExit('Verified persistent worktree is missing')
print(p)
PY
)
cd "$R6_RESTORED_WORKTREE"
printf '%s\n' 'Restart handoff: /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/REBOOT-HANDOFF.md'
exec /usr/lib/chatgpt/resources/codex --no-daemon resume 01a1541e-50e0-78f1-9938-d41a6a5e74f7 --cd "$R6_RESTORED_WORKTREE"
