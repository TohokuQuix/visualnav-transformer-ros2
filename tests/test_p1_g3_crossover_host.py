"""Synthetic host pre-Path handshake and no-publication failure checks."""

import hashlib
import json
from pathlib import Path

import pytest

from eat_nav_p1_g3_crossover_host import approve_prepath
from test_p1_g3_crossover_runtime import snapshot


def put(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, current=None):
    output=root/'output'
    value=snapshot() if current is None else current
    material={'schema':'eatnav_p1_live_material_audit_v1',
              'terrain_declared_sha256':value['terrain_material_sha256'],
              'terrain_collision_bindings':[{'path':'terrain','canonical_sha256':'terrain-live'}],
              'robot_collision_bindings':[{'path':'robot','canonical_sha256':'robot-live'}],
              'robot_effective_physx':{'canonical_sha256':value['robot_material_sha256']}}
    put(output/'prepath-crossover-snapshot.json',value)
    put(output/'live-material.json',material)
    put(output/'prepath-gate.json',{'pass':True})
    put(output/'direct-actor-gate.json',{'max_abs':0.0,'limit':0.0001})
    put(output/'prepath-ready.json',{'Path_publications':0,'physics_steps_after_clone':0,
        'snapshot_sha256':sha(output/'prepath-crossover-snapshot.json'),
        'live_material_sha256':sha(output/'live-material.json')})


def test_first_period_self_gate_and_second_period_diagnostic_difference(tmp_path,monkeypatch):
    first=tmp_path/'batch-first-differential-0-period-0'
    prepare(first)
    assert approve_prepath(first,'first-differential',0,0)['pass']
    second=tmp_path/'batch-first-differential-0-period-1'
    value=snapshot();value['envs'][0]['contact_current']=[99.0]
    prepare(second,value)
    monkeypatch.setattr('eat_nav_p1_g3_crossover_host.verify_seal',
                        lambda path:{'stage':'G3_crossover_batch_protocol_valid'})
    assert approve_prepath(second,'first-differential',0,1)['pass']
    gate=json.loads((second/'output/cross-period-prepath-gate.json').read_text())
    assert gate['diagnostic_differences'][0]['signal']=='contact_current'
    assert not (second/'output/path-publication.json').exists()


def test_physical_difference_rejects_without_approval(tmp_path,monkeypatch):
    first=tmp_path/'batch-first-differential-0-period-0';prepare(first)
    second=tmp_path/'batch-first-differential-0-period-1'
    value=snapshot();value['envs'][1]['root_state_w'][0]+=0.001
    prepare(second,value)
    monkeypatch.setattr('eat_nav_p1_g3_crossover_host.verify_seal',
                        lambda path:{'stage':'G3_crossover_batch_protocol_valid'})
    with pytest.raises(ValueError,match='cross-period'):
        approve_prepath(second,'first-differential',0,1)
    assert (second/'output/prepath-rejection.json').exists()
    assert not (second/'output/prepath-approval.json').exists()
