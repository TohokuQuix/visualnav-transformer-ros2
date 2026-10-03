import json

import pytest

from eat_nav_e1_prepare import (ROOT, G0E, build_continuation, verify_anchor_reuse,
    validate_frozen_identity)


def test_continuation_preserves_original_20_row_identity():
    original=json.loads((G0E/'e1-process-manifest.json').read_text())
    continuation=build_continuation(original)
    assert continuation['counts']=={'completed_anchor_only':2,'pending_direct_repeats':6,
        'pending_nondirect_outcomes':12,'pending_total':18,'unique_candidates':14}
    assert [r['order'] for r in continuation['pending_rows']]==[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]
    assert continuation['pending_rows']==[r for r in original['rows'] if r['role']!='anchor_only']
    assert all(r['seed']==123 for r in continuation['pending_rows'])
    validate_frozen_identity(original,continuation)


def test_continuation_rejects_path_sha_or_order_drift():
    original=json.loads((G0E/'e1-process-manifest.json').read_text())
    continuation=build_continuation(original)
    continuation['pending_rows'][0]['map_path_sha256']='0'*64
    with pytest.raises(ValueError,match='identity'):
        validate_frozen_identity(original,continuation)


def test_two_formal_anchors_are_complete_and_sha_valid():
    assert verify_anchor_reuse(G0E)==['g0e-train-000','g0e-train-001']
