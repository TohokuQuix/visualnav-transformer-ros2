import hashlib
import json

import pytest

from scripts.eat_nav_minimal_p1_freeze import verify_ledger, merge_anchor_evidence, source_map
from scripts import eat_nav_minimal_p1_freeze as freeze


def test_ledger_detects_modified_artifact(tmp_path):
    (tmp_path/'decision.json').write_text('{"decision":"x"}\n')
    digest=hashlib.sha256((tmp_path/'decision.json').read_bytes()).hexdigest()
    (tmp_path/'artifact-sha-ledger.json').write_text(json.dumps({'files':{
        'decision.json':{'sha256':digest,'size_bytes':(tmp_path/'decision.json').stat().st_size}}}))
    assert verify_ledger(tmp_path)['verified_file_count']==1
    (tmp_path/'decision.json').write_text('{"decision":"y"}\n')
    with pytest.raises(ValueError,match='SHA'):
        verify_ledger(tmp_path)


def test_merge_refuses_anchor_dem_history_mismatch():
    role={'anchor_id':'a','dem_history_sha256':['0','1','2','3'],
          'physical_eligible':True,'source_path_shas':{'p':'sha'}}
    raw={'anchor_id':'a','dem_history_sha256':['0','1','2','DIFFERENT'],
         'physical_supply':True,'source_path_shas':{'p':'sha'},
         'body_state_actor':[0.]*22}
    with pytest.raises(ValueError,match='DEM history'):
        merge_anchor_evidence(role,raw)


def test_follower_binary_resolves_through_durable_backup():
    sources=source_map()
    follower=sources['follower_binary']
    assert '/home/kojima/eatnav-backups/2026-10-08-pre-shutdown/' in follower['path']
    assert follower['sha256']=='616097e44d199b8d33f6534aedd299f2d20127ed8e100c18777f9ca7a2da389d'


def test_follower_source_drift_rejected(monkeypatch):
    original=freeze.sha
    def drift(path):
        if str(path).endswith('/simple_trajectory_follower.cpp'):
            return '0'*64
        return original(path)
    monkeypatch.setattr(freeze,'sha',drift)
    with pytest.raises(ValueError,match='follower source SHA'):
        source_map()
