"""Verify bundle bytes without importing Isaac or executing a rollout."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for name,want in manifest['files'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=want:raise SystemExit('SHA mismatch: '+name)
print('PASS:',len(manifest['files']),'bundle files verified')
