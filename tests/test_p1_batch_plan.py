import json
from pathlib import Path

from visualnav_transformer.eat_nav.p1_batch_plan import build_plan


def test_28_slots_have_frozen_path_seed_and_direct_triplicates():
    root = Path('/home/kojima/eatnav-backups/2026-10-09-p1-radial-28-analysis')
    manifest = json.loads((root / 'integrated-manifest.json').read_text())
    contract = json.loads(Path('config/eat_nav_p1_batch_source_contract.json').read_text())
    plan = build_plan(manifest, contract)
    assert len(plan['targets']) == 28
    assert len({r['target_env_index'] for r in plan['targets']}) == 28
    assert {r['target_env_index'] for r in plan['targets']} == set(range(28))
    for group in range(4):
        rows = plan['targets'][group*7:(group+1)*7]
        assert len({r['terrain_seed'] for r in rows}) == 1
        assert len({r['terrain_type'] for r in rows}) == 1
        assert len({r['candidate_path_sha256'] for r in rows}) == 5
        assert rows[2]['candidate_path_sha256'] == rows[3]['candidate_path_sha256'] == rows[4]['candidate_path_sha256']
        assert rows[2]['source_anchor_local_xy'] == rows[3]['source_anchor_local_xy']
