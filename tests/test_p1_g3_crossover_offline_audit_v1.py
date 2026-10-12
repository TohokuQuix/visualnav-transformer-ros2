"""Frozen assignment validation and offline audit controls; no Isaac rollout."""

import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment
from eat_nav_p1_g3_crossover_offline_audit_v1 import classify_checks
from eat_nav_p1_g3_crossover_offline_analyze_v1 import require_audit_pass


REPO = Path(__file__).resolve().parents[1]
ASSIGNMENT = json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v2-assignment.json').read_text())


@pytest.mark.parametrize('period', (0, 1, 2))
def test_each_period_accepts_frozen_prelaunch_and_publication(period):
    process = next(p for p in ASSIGNMENT['processes'] if p['anchor_id']=='first-differential'
                   and p['batch_index']==0 and p['execution_index']==period)
    rows = process['trials']
    validate_frozen_assignment(rows, rows, stage='prelaunch')
    publication = [{'trial_id':f'example-period-{period}-env-{row["slot"]}',
                    'path_role':row['path_role'],'path_sha256':row['path_sha256']}
                   for row in rows]
    validate_frozen_assignment(rows, publication, stage='post_path')


@pytest.mark.parametrize('stage', ('prelaunch', 'post_path'))
@pytest.mark.parametrize('field', ('path_role', 'path_sha256'))
def test_frozen_assignment_rejects_role_or_sha_drift(stage, field):
    rows = next(p['trials'] for p in ASSIGNMENT['processes'] if
                p['anchor_id']=='first-differential' and p['batch_index']==0 and
                p['execution_index']==2)
    observed = json.loads(json.dumps(rows))
    if stage=='post_path':
        observed = [{'trial_id':f'example-env-{row["slot"]}',
                     'path_role':row['path_role'],'path_sha256':row['path_sha256']}
                    for row in rows]
    observed[0][field] = 'wrong'
    with pytest.raises(ValueError, match='assignment'):
        validate_frozen_assignment(rows, observed, stage=stage)


def test_post_path_rejects_duplicate_or_swapped_env():
    rows = next(p['trials'] for p in ASSIGNMENT['processes'] if
                p['anchor_id']=='first-differential' and p['batch_index']==0 and
                p['execution_index']==2)
    observed = [{'trial_id':f'example-env-{row["slot"]}',
                 'path_role':row['path_role'],'path_sha256':row['path_sha256']}
                for row in rows]
    observed[1]['trial_id'] = observed[0]['trial_id']
    with pytest.raises(ValueError, match='assignment'):
        validate_frozen_assignment(rows, observed, stage='post_path')


def test_new_host_uses_one_validator_at_both_stages():
    source = (REPO/'scripts/eat_nav_p1_g3_crossover_v3_host.py').read_text()
    assert "validate_frozen_assignment(frozen_rows, assigned_rows, stage='prelaunch')" in source
    assert "validate_frozen_assignment(assigned, publication['trial_paths'], stage='post_path')" in source
    assert "len({entry['path_sha256'] for entry in publication['trial_paths']})!=3" not in source


def test_audit_classification_fails_closed():
    assert classify_checks({'a':'pass','b':'pass'})=='audit_pass'
    assert classify_checks({'a':'pass','b':'missing'})=='audit_inconclusive'
    assert classify_checks({'a':'fail','b':'missing'})=='audit_fail'


def test_analysis_requires_complete_audit_and_keeps_original_failed_stage():
    report={'decision':'audit_pass','original_period2_stage_preserved':
            'G3_crossover_batch_failed_or_inconclusive','checks':{'a':'pass'}}
    require_audit_pass(report)
    with pytest.raises(ValueError,match='audit_pass'):
        require_audit_pass({**report,'decision':'audit_inconclusive'})
    with pytest.raises(ValueError,match='original'):
        require_audit_pass({**report,'original_period2_stage_preserved':
                            'G3_crossover_batch_protocol_valid'})
