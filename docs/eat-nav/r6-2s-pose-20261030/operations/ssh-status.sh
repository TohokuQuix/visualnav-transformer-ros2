#!/usr/bin/env bash
set -eu
R6_ARTIFACT_ROOT=/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030
/usr/bin/python3 - "$R6_ARTIFACT_ROOT" <<'PY'
import collections,json,sys
from pathlib import Path
root=Path(sys.argv[1])
p=root/'evaluation-pipeline.json'
if p.exists():
 print('Pipeline stages:')
 for name,record in json.loads(p.read_text()).items():
  print(' ',name,record.get('state'),record.get('reason',''))
for role in ('validation','test'):
 p=root/(role+'-evaluation/progress.json')
 if not p.exists():continue
 records=json.loads(p.read_text());counts=collections.Counter(r.get('state') for r in records.values())
 jobs=root/(role+'-evaluation/jobs.json');planned=len(json.loads(jobs.read_text())) if jobs.exists() else '?'
 print(role,dict(counts),'planned jobs',planned)
 for name,r in records.items():
  if r.get('state') in ('running','failed'):print(' ',name,r.get('state'),r.get('reason',''))
p=root/'evaluation-production.log'
if p.exists():
 print('Latest supervisor log:')
 print('\n'.join(p.read_text().splitlines()[-12:]))
print('A running ledger is not proof of a live process. Check processes below.')
PY
pgrep -af 'python.*scripts/eat_nav_r6_(evaluation_pipeline|eval_sim|eval_queue)\.py' || true
