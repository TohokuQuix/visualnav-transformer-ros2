"""Write a deterministic ledger for small, committed teacher-audit artifacts."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
folder=root/'docs/quality-identifiability-gate/g40-teacher-review'
entries=[]
for path in sorted(folder.iterdir()):
    if path.is_file() and path.name!='artifact-sha-ledger.json':
        entries.append({'path':str(path.relative_to(root)),
                        'bytes':path.stat().st_size,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
for name in ('scripts/eat_nav_g40_teacher_review.py',
             'scripts/eat_nav_g40_teacher_figures.py',
             'scripts/eat_nav_g40_teacher_ledger.py',
             'tests/eat_nav/test_g40_teacher_review.py'):
    path=root/name
    entries.append({'path':name,'bytes':path.stat().st_size,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(folder/'artifact-sha-ledger.json').write_text(json.dumps({'schema_version':1,'entries':entries},
                                                   indent=2,sort_keys=True)+'\n')
