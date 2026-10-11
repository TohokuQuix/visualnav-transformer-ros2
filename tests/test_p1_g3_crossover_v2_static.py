"""Synthetic freeze and source-integrity checks; no Isaac/Path publication."""

import hashlib
import json
from pathlib import Path

import pytest

from eat_nav_p1_g3_crossover_v2_stage import manifest_for, stage
from eat_nav_p1_g3_crossover_v2_host import approve_prepath, validate_period_assignment
from eat_nav_p1_g3_crossover_outer import command_for
from test_p1_g3_crossover_host import prepare as prepare_v1
from visualnav_transformer.eat_nav.p1_g3_material_gate_v2 import SCHEMA, MATERIAL_SCHEMA


REPO = Path(__file__).resolve().parents[1]
OLD = Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_v2_manifest_and_staging(tmp_path):
    assignment_path=REPO/'docs/eat-nav/p1-g3-crossover-v2-assignment.json'
    manifest_path=REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json'
    assignment=json.loads(assignment_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    assert manifest['schema']==SCHEMA
    assert manifest_for(REPO, OLD, sha(assignment_path),assignment)==manifest
    assert manifest['pilot_authorized_max_processes']==3
    assert manifest['live_material']['resolved_startup_randomization']['num_buckets']==128
    result=stage(REPO,OLD,assignment_path,manifest_path,tmp_path/'staged')
    assert result['process_count']==18
    first=tmp_path/'staged/batch-first-differential-0-period-0'
    assert sha(first/'source.py')==manifest['g3_simulator_source_sha256']
    assert sha(first/'src/visualnav_transformer/eat_nav/p1_g3_material_gate_v2.py')==manifest['material_gate_module_sha256']
    assert sha(first/'input/g3-contract.json')==sha(manifest_path)
    cmd=command_for(first,'/remote/batch',REPO/'scripts/eat_nav_p1_g3_crossover_v2_host.py',
                    tmp_path/'router','first-differential',0,0)
    assert cmd[-2:]==['--execution-index','0']


def test_material_gate_is_durable_before_fail_and_precedes_path():
    source=(REPO/'scripts/eat_nav_p1_g3_crossover_v2_sim.py').read_text()
    assert source.index("_durable_json(args.output/'live-material.json',live_materials)") < source.index("if not live_materials['pass']")
    assert source.index("if not live_materials['pass']") < source.index('await_host_prepath_approval(args.output')
    host=(REPO/'scripts/eat_nav_p1_g3_crossover_v2_host.py').read_text()
    assert "material.get('schema')!=MATERIAL_SCHEMA or not material.get('pass')" in host


def test_host_rejects_material_failure_before_path(tmp_path):
    batch=tmp_path/'batch-first-differential-0-period-0'
    prepare_v1(batch)
    material_path=batch/'output/live-material.json'
    material=json.loads(material_path.read_text())
    material.update(schema=MATERIAL_SCHEMA, **{'pass':False,'failures':['shape_count']})
    material_path.write_text(json.dumps(material))
    ready_path=batch/'output/prepath-ready.json'
    ready=json.loads(ready_path.read_text())
    ready['live_material_sha256']=sha(material_path)
    ready_path.write_text(json.dumps(ready))
    with pytest.raises(ValueError,match='live material audit incomplete'):
        approve_prepath(batch,'first-differential',0,0)
    assert not (batch/'output/prepath-approval.json').exists()


def test_host_preserves_assignment_rows_through_gpu_sampling():
    source=(REPO/'scripts/eat_nav_p1_g3_crossover_v2_host.py').read_text()
    assert "assigned_rows=sorted(matched[0]['trials']" in source
    assert 'devices,nvml_rows=nvml.sample()' in source
    assert 'validate_g3_output(ROOT,REMOTE,g3,assigned_rows)' in source


def test_period_two_two_left_two_right_is_valid_frozen_assignment():
    manifest=json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    rows=next(p['trials'] for p in manifest['processes'] if
              p['anchor_id']=='first-differential' and p['batch_index']==0 and
              p['execution_index']==2)
    assert [x['path_role'] for x in rows]==['right_20','left_20','right_20','left_20']
    validate_period_assignment(rows,rows)
    bad=json.loads(json.dumps(rows));bad[0]['path_role']='direct'
    with pytest.raises(ValueError,match='assignment'):
        validate_period_assignment(bad,rows)
