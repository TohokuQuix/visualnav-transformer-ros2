from pathlib import Path
import json
import pytest
from eat_nav_clock_diagnostic_stage0 import approved_runs

REPO=Path(__file__).resolve().parents[2]
PRIOR=REPO/'docs/quality-identifiability-gate/candidate-coverage/command-preview-long-horizon'
RESULT=REPO/'docs/quality-identifiability-gate/candidate-coverage/clock-domain-diagnostic'


def test_approved_nominal_triplet_is_frozen():
    identities=json.loads((PRIOR/'long-horizon-run-identities.json').read_text())
    runs=approved_runs(identities)
    assert [(r['state_id'],r['candidate_id'],r['repeat']) for r in runs]==[
        ('coverage-003','bank-nominal',0),('coverage-003','bank-nominal',1),
        ('coverage-003','bank-nominal',2)]
    altered=json.loads(json.dumps(identities))
    altered['runs'][6]['candidate_id']='nomad-seed-1001'
    with pytest.raises(AssertionError):approved_runs(altered)


def test_diagnostic_follower_patch_is_additive_only():
    lines=(RESULT/'diagnostic-follower.patch').read_text().splitlines()
    assert not [line for line in lines if line.startswith('-') and not line.startswith('---')]
    patch='\n'.join(lines)
    assert 'CLOCK_DIAG pre' in patch and 'CLOCK_DIAG post' in patch
    assert 'diagnostic_callback_seq_' in patch
    audit=json.loads((RESULT/'diagnostic-instrumentation-diff.json').read_text())
    assert not audit['control_branch_changed'] and not audit['parameter_changed']
    assert not audit['timer_changed'] and not audit['publisher_or_subscriber_changed']


def test_fixture_parity_before_simulator_runs():
    parity=json.loads((RESULT/'diagnostic-fixture-parity.json').read_text())
    assert parity['status']=='pass' and len(parity['fixtures'])==7
    assert max(row['max_abs_error'] for row in parity['fixtures'])<1e-5
    exact=json.loads((RESULT/'diagnostic-fixture-exact-parity.json').read_text())
    assert exact['status']=='pass' and len(exact['rows'])==7
    assert all(row['exact_equal'] for row in exact['rows'])


def test_simulator_scope_guard_and_no_retries():
    sim=(REPO/'scripts/eat_nav_clock_diagnostic_sim.py').read_text()
    runner=(REPO/'scripts/eat_nav_clock_diagnostic_run.py').read_text()
    assert "state['state_id']!='coverage-003'" in sim
    assert 'unapproved nominal Path' in sim
    assert "if any(folder.iterdir()):raise RuntimeError" in runner
    assert "approved_runs" in runner and "[5,6,7]" in runner
