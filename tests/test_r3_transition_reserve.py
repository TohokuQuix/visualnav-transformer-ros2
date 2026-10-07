"""Prelaunch guards for the already frozen R3 transition reserve."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from visualnav_transformer.eat_nav.r3_transition_decision import decide_transition

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/tmp/eatnav-r3-transition-reserve-20261008')


def module(name):
    p=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location(name,p)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def test_reserve_selection_inputs_are_outcome_blind(monkeypatch):
    prep=module('eat_nav_r3_transition_prepare.py')
    frozen=json.loads((OUT/'reserve-manifest.json').read_text())
    primary=json.loads((prep.OLD/'primary-manifest.json').read_text())['poses']
    old_read=Path.read_text
    def guarded(path,*args,**kwargs):
        if '/analysis/' in str(path) or '/trace/' in str(path):
            raise AssertionError('selection attempted to read prior result')
        return old_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    rows=prep.selected_rows(frozen,primary)
    assert len(rows)==2
    assert [x['rank'] for x in rows]==[3,4]
    assert all(x['virtual_class']=='transition' for x in rows)
    assert not {x['pose_id'] for x in rows}&{x['pose_id'] for x in primary}


def test_manifest_sealed_and_no_overwrite(tmp_path):
    prep=module('eat_nav_r3_transition_prepare.py')
    ledger=json.loads((OUT/'freeze-sha.json').read_text())
    assert prep.sha(OUT/'reserve-manifest.json')==ledger['reserve_manifest_sha256']
    assert prep.sha(OUT/'reserve-preaudit-selection.json')==ledger['reserve_preaudit_sha256']
    target=tmp_path/'immutable.json';prep.write_new(target,{'a':1})
    with pytest.raises(RuntimeError,match='already exists'):
        prep.write_new(target,{'a':2})
    assert json.loads(target.read_text())=={'a':1}


def test_same_classifier_limits_and_termination_aware_runner():
    assert hashlib.sha256((ROOT/'src/visualnav_transformer/eat_nav/radial_interaction.py').read_bytes()).hexdigest()==(
        'fc12a07a372c3597402bf6002e6ef000c63bd7a50db1717c47f2d0c18a3addf8')
    freeze=json.loads((OUT/'shard-reserve/input-freeze.json').read_text())
    assert freeze['planned_physics_steps']==200
    assert freeze['contact_contract']['view_capacity']==4096
    assert freeze['numeric_diagnostics']=={'angular_speed_max_radps':.2,'dem_valid_fraction_min':.7,
        'h1_abs_max_rad':.1,'linear_speed_max_mps':.1,'xy_drift_max_m':.1}
    assert freeze['retry_after_any_physics_step'] is False
    host=(ROOT/'scripts/eat_nav_r3_transition_run.py').read_text()
    assert 'termination_aware=True' in host
    assert 'one-shot already launched' in host
    assert 'host/container SHA mismatch' in host
    assert 'owned_alive(item)' in host


def test_old_outputs_not_written_by_new_sources():
    for name in ('eat_nav_r3_transition_prepare.py','eat_nav_r3_transition_cause_audit.py',
                 'eat_nav_r3_transition_run.py'):
        source=(ROOT/'scripts'/name).read_text()
        assert "(OLD/'analysis/" not in source or name.endswith('cause_audit.py')
        assert "write_text" not in source.split('OLD =',1)[-1].split('def ',1)[0]


def test_transition_decision_requires_eligible_known_class_supply():
    good={'classification':'settled_anchor_eligible','settled_class':'transition','physical_supply':True}
    bad={'classification':'anchor_ineligible_numeric','settled_class':None,'physical_supply':False}
    assert decide_transition([good,bad],2)=='r3_transition_supply_confirmed_ready_for_second_seed_design'
    assert decide_transition([bad,bad],2)=='r3_transition_supply_not_confirmed_placement_redesign_required'
    mismatch={'classification':'settled_anchor_eligible','settled_class':'flat-only','physical_supply':False}
    assert decide_transition([mismatch,bad],2)=='r3_transition_class_mismatch_review_required'
    assert decide_transition([good],1)=='r3_transition_reserve_insufficient'
    assert decide_transition([good],2)=='r3_transition_pilot_measurement_invalid'
    assert decide_transition([good,{'classification':'measurement_invalid'}],2)=='r3_transition_pilot_measurement_invalid'
