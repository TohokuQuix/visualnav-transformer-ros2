import pytest

from visualnav_transformer.eat_nav.evaluator_readiness_schedule import schedule


def test_112_rollouts_with_anchors_and_prefix_k():
    selection = {'selected': [{'state_id': f'readiness-{i:03d}', 'terrain_type': str(i),
                               'episode_key': [0, i, 0]} for i in range(7)]}
    ids = [f'nomad-seed-{s}' for s in range(1001, 1009)] + [
        f'bank-{n}' for n in ('nominal', 'slight_left', 'slight_right', 'strong_left', 'strong_right')]
    tables = [[{'candidate_id': name, 'base_path_sha256': str(k)} for k, name in enumerate(ids)] for _ in range(7)]
    frozen = schedule(selection, tables)
    assert frozen['rollout_count'] == 112
    assert frozen['anchor_count'] == 7
    assert [r['kind'] for r in frozen['runs'][:17]] == ['anchor'] + ['nominal'] * 3 + ['nomad'] * 8 + ['bank'] * 5
    assert [r['seed'] for r in frozen['runs'][4:12]] == list(range(1001, 1009))
    assert [r['order'] for r in frozen['runs']] == list(range(119))
    tables[0][4]['candidate_id'] = 'replacement'
    with pytest.raises(ValueError, match='candidate order'):
        schedule(selection, tables)
