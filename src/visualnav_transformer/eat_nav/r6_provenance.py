"""Immutable execution-source and validation-freeze gates for R6 evaluation.

No simulator, ROS, or model imports: reject incompatible experiments before any
expensive launch or bag inference. A source checkout is required for reproduction.
"""
import hashlib
import json
from pathlib import Path


def runtime_source_hashes(repo):
    repo=Path(repo)
    sources={p for p in (repo/'src/visualnav_transformer').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    sources.update((repo/'scripts').glob('eat_nav*.py'))
    sources.update(p for p in (repo/'config').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    sources.update(repo/name for name in (
        'checkpoints/eat_nav/rl_actor/model_10000.pt',
        'checkpoints/eat_nav/rl_actor/env.yaml',
        'checkpoints/eat_nav/fall_stall_r4star_r5_index.json'))
    if not any(p.is_relative_to(repo/'src/visualnav_transformer') for p in sources):
        raise ValueError('complete source checkout required for frozen execution')
    return {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}


def verified_freeze_data(frozen):
    value=dict(frozen);digest=value.pop('freeze_sha256',None)
    actual=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=actual or value.get('schema')!='eatnav_r6_frozen_top_three_v1' or value.get('source')!='validation_only':
        raise ValueError('frozen selection digest/source mismatch')
    value['freeze_sha256']=digest
    return value


def verify_execution_freeze(frozen,model_sha256,source_sha256):
    value=verified_freeze_data(frozen)
    if len(value.get('profiles',[]))!=3:
        raise ValueError('exactly three frozen profiles required')
    if value.get('model_sha256')!=model_sha256:
        raise ValueError('frozen profile/model binding mismatch')
    if value.get('evaluation_source_sha256')!=source_sha256:
        raise ValueError('frozen execution source mismatch; never mix validation and test/bag implementations')
    return value
