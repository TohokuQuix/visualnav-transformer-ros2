import json
import os
from pathlib import Path

import pytest

from scripts.eat_nav_goal_bank_redesign import run_redesign, verify_baseline


FROZEN = Path(os.environ.get('EATNAV_FROZEN_ARTIFACT_ROOT','/tmp'))
PRIOR = Path(os.environ.get('EATNAV_PRIOR_REVIEW_ROOT','/tmp/eatnav-goal-bearing-bank-review-20261008-r6'))
HAS_FIXTURES = ((FROZEN/'eatnav-r3-second-seed-20261008').exists()
                and (PRIOR/'sha-ledger.json').exists())


@pytest.mark.skipif(not HAS_FIXTURES, reason='frozen artifacts not supplied')
def test_baseline_archive_and_prior_output_are_sha_verified():
    result=verify_baseline(FROZEN,PRIOR)
    assert result['prior_review_output_sha_entries']==35
    assert result['prior_review_source_sha_entries']==87
    assert result['follower_source_sha256']['simple_trajectory_follower.cpp']


@pytest.mark.skipif(not HAS_FIXTURES, reason='frozen artifacts not supplied')
def test_rewritten_prior_ledger_cannot_rebaseline_archive(tmp_path):
    (tmp_path/'sha-ledger.json').write_text('{"files":{}}')
    with pytest.raises(ValueError,match='ledger differs from frozen backup archive'):
        verify_baseline(FROZEN,tmp_path)


@pytest.mark.skipif(not HAS_FIXTURES, reason='frozen artifacts not supplied')
def test_full_redesign_stays_offline_and_fails_five_of_seven_gate(tmp_path):
    output=tmp_path/'redesign'
    result=run_redesign(FROZEN,PRIOR,output,figures=False)
    assert result['decision']=='candidate_bank_compatibility_still_insufficient'
    assert result['candidate_count']=={'A':280,'B':280}
    assert result['candidate_path_rollout_count']==0
    assert not result['bank_readiness']['A']['ready']
    assert not result['bank_readiness']['B']['ready']
    assert all(result['bank_readiness'][bank]['groups_at_least_five']==0 for bank in ('A','B'))
    assert len(json.loads((output/'bank-A-triple-comparison.json').read_text()))==10
    assert len(json.loads((output/'bank-B-triple-comparison.json').read_text()))==10
    assert len(json.loads((output/'follower-screen-reason-matrix.json').read_text()))==560
    boundary=json.loads((output/'amplitude-offset-compatibility-boundary.json').read_text())
    assert boundary['A']['levels']['0.04']['compatible']==80
    assert all(row['compatible']==0 for key,row in boundary['B']['levels'].items())
    assert json.loads((output/'baseline-test-recovery.json').read_text())['expected']=='18 passed'
    source_audit=json.loads((output/'source-exact-follower-audit.json').read_text())
    assert len(source_audit['production_C0_numeric_parity'])==3
    assert max(case['max_abs_error'] for case in source_audit['production_C0_numeric_parity'])<1e-5
    gates=json.loads((output/'prospective-P1-P2-gates.json').read_text())
    assert gates['duplicate_gate']['xy_rms_max_m']==0.0105
    assert gates['A_to_R_per_group']['candidate_to_duplicate_ratio_min']==2.0
    assert gates['P2_only_after_P1_pass'] is True
    candidates=json.loads((output/'selected-candidate-definitions.json').read_text())
    assert all(row['terminal_mode']=='blended_goal_connected' for row in candidates if row['bank']=='B')
    with pytest.raises(FileExistsError,match='new output root'):
        run_redesign(FROZEN,PRIOR,output,figures=False)
