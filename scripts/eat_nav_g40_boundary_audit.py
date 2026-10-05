"""B0: verify frozen source and write a standalone pre-inference ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eat_nav_evaluator_design_audit import main as source_audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/quality-identifiability-gate/g40-boundary-freeze'
OLD=ROOT/'docs/quality-identifiability-gate/g40-teacher-review'
DESIGN=ROOT/'docs/quality-identifiability-gate/evaluator-design-review'
RUN=Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    audit=source_audit()
    old_ledger=json.loads((OLD/'artifact-sha-ledger.json').read_text())
    for entry in old_ledger['entries']:
        path=ROOT/entry['path']
        if digest(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:
            raise ValueError(f'prior G40 artifact mismatch: {path}')
    files={}
    for folder in (OLD,DESIGN):
        for path in folder.iterdir():
            if path.is_file():files[str(path.relative_to(ROOT))]=digest(path)
    for path in (
        RUN/'model_10000.pt',RUN/'params/agent.yaml',RUN/'params/env.yaml',
        Path('/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py'),
        Path('/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/mdp/observations.py'),
        Path('/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py'),
        Path('/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py'),
        ROOT/'src/visualnav_transformer/eat_nav/phase_c_sim.py',
        ROOT/'scripts/eat_nav_e1_sim.py',
        Path('/tmp/eatnav-vnt-goal-ood-simulator-v2/docs/quality-identifiability-gate/phase-c/reference-eat.yaml'),
    ):
        files[str(path)]=digest(path)
    config=OUT/'boundary-analysis-config.yaml'
    files[str(config.relative_to(ROOT))]=digest(config)
    if files[str(RUN/'model_10000.pt')]!=audit['checkpoint_sha256_rechecked']:
        raise ValueError('checkpoint SHA mismatch')
    return {'status':'pass','source_commit':'87d1ef3e1e617bf452949a7f48658718af51723e',
            'branch':'exp/eatnav-g40-boundary-freeze','worktree':str(ROOT),
            'existing_g40_ledger_sha256':digest(OLD/'artifact-sha-ledger.json'),
            'prior_artifact_count_reverified':len(old_ledger['entries']),
            'source_files_sha256':files,'source_audit':audit,
            'sealed_trace_count':63,'state_group_count':7,
            'source_trace_sha256':{r['process_id']:r['trace_sha256'] for rows in audit['groups'].values() for r in rows}}

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source-provenance.json').write_text(json.dumps(main(),indent=2,sort_keys=True)+'\n')
    print('B0 source integrity pass; 63 traces; config frozen before inference')
