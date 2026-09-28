"""Freeze source and artifact identities before Phase C remediation execution."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/quality-identifiability-gate/phase-c-remediation'
OUT.mkdir(parents=True, exist_ok=True)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def docker(cmd):
    return subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'bash', '-lc', cmd], text=True).strip()

def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)

paths = ['src/visualnav_transformer/eat_nav/phase_c_runner.py',
         'src/visualnav_transformer/eat_nav/phase_c_reference.py',
         'src/visualnav_transformer/eat_nav/model.py',
         'src/visualnav_transformer/eat_nav/dataset.py',
         'src/visualnav_transformer/eat_nav/phase_c_sim.py',
         'src/visualnav_transformer/eat_nav/phase_c_bridge_client.py',
         'scripts/eat_nav_phase_c_ros_bridge.py',
         'scripts/eat_nav_phase_c_smoke.py',
         'scripts/eat_nav_phase_c_remediation_provenance.py']
source_sha = {p: sha((ROOT / p).read_bytes()) for p in paths}
repo = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
ids = ROOT / 'docs/quality-identifiability-gate/phase-c/complete-window-ids.jsonl.gz'
raw = gzip.decompress(ids.read_bytes())
by_split = {'train': [], 'val': []}
for line in raw.splitlines(keepends=True):
    by_split[json.loads(line)['split']].append(line)
audit = json.loads((ROOT / 'docs/quality-identifiability-gate/phase-c/reference-data-audit.json').read_text())
normalization = {'mean': audit['train_action_mean'], 'std': audit['train_action_std'], 'count': audit['train_action_count']}
original_latest_sha = docker('sha256sum /tmp/eatnav-phase-c/reference/latest.pt | cut -d" " -f1')
copy_latest_sha = docker('sha256sum /tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/latest.pt | cut -d" " -f1')
if original_latest_sha != copy_latest_sha:
    raise ValueError('resume checkpoint copy mismatch')
artifacts = {
    'phase_c0_latest_checkpoint_sha256': original_latest_sha,
    'reference_dataset_manifest_sha256': docker('sha256sum /tmp/eatnav-extended/output/manifest.json | cut -d" " -f1'),
    'complete_ids_decompressed_sha256': sha(raw),
    'complete_train_ids_decompressed_sha256': sha(b''.join(by_split['train'])),
    'complete_validation_ids_decompressed_sha256': sha(b''.join(by_split['val'])),
    'complete_counts': {k: len(v) for k, v in by_split.items()},
    'normalization_canonical_json_sha256': sha(json.dumps(normalization, sort_keys=True, separators=(',', ':')).encode()),
    'normalization': normalization,
    'reference_config_sha256': sha((ROOT / 'docs/quality-identifiability-gate/phase-c/reference-eat.yaml').read_bytes()),
    'remediation_config_sha256': sha((OUT / 'remediation-config.yaml').read_bytes()),
}
versions = docker("/isaac-sim/python.sh -c 'import json,platform,torch,diffusers; print(json.dumps(dict(python=platform.python_version(),pytorch=torch.__version__,cuda=torch.version.cuda,diffusers=diffusers.__version__)))' 2>/dev/null | tail -1")
provenance = {
    'git_commit': repo('rev-parse', 'HEAD'),
    'uncommitted_diff_present': bool(repo('status', '--porcelain')),
    'uncommitted_diff_sha256': sha(repo('diff', '--binary').encode()),
    'source_sha256': source_sha,
    'docker_image': docker('cat /etc/os-release | head -2'),
    'docker_image_id': subprocess.check_output(['docker', 'inspect', 'isaac-lab-onix', '--format', '{{.Image}}'], text=True).strip(),
    'versions': json.loads(versions),
    'isaac_sim_version': docker('cat /isaac-sim/VERSION 2>/dev/null || true'),
    'artifacts': artifacts,
}
atomic(OUT / 'source-provenance.json', provenance)
atomic(OUT / 'artifact-hashes.json', artifacts)
print(json.dumps({'source_provenance_sha256': sha((OUT / 'source-provenance.json').read_bytes()), 'artifacts': artifacts}, sort_keys=True))
