import json

import numpy as np
import pytest
import yaml

from eat_nav_e1_anchor_gate import check_before_path
from eat_nav_e1_prepare import BANK, G0E


def case(index=0):
    summary=json.loads((G0E/'runs'/f'anchor-{index:03d}'/'anchor-0.json').read_text())
    logged=json.loads((BANK/'inputs'/f'state-{index:03d}'/'logged-anchor.json').read_text())
    frames=json.loads((BANK/'inputs'/f'state-{index:03d}'/'recorded-frames.json').read_text())
    with np.load(BANK/'g0e-selected-states.npz') as z:
        goal=z['goal'][index].copy();dem=z['dem'][index].copy()
    return summary,logged,frames,goal,dem


def test_saved_formal_anchor_passes_before_path_gate():
    summary,logged,frames,goal,dem=case()
    result=check_before_path(summary,logged,frames,goal,dem,
        source_sha256=summary['source_sha256']['runner_source'],
        state_config=BANK/'inputs/state-000/anchor-config.yaml',
        reference_config='/tmp/eatnav-vnt-goal-ood-simulator-v2/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
        limits=yaml.safe_load((G0E/'formal-anchor-config.yaml').read_text())['anchor_limits'])
    assert result['status']=='pass'
    assert len(result['dem_history_gate']['frames'])==4


def test_episode_mismatch_blocks_path_publication():
    summary,logged,frames,goal,dem=case()
    summary['state']['episode_key'][2]=1
    with pytest.raises(ValueError,match='anchor'):
        check_before_path(summary,logged,frames,goal,dem,
            source_sha256=summary['source_sha256']['runner_source'],
            state_config=BANK/'inputs/state-000/anchor-config.yaml',
            reference_config='/tmp/eatnav-vnt-goal-ood-simulator-v2/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
            limits=yaml.safe_load((G0E/'formal-anchor-config.yaml').read_text())['anchor_limits'])
