from copy import deepcopy

import pytest

from test_p1_g3_material_diagnostic import fixture
from visualnav_transformer.eat_nav.p1_g3_material_gate_v2 import (
    assess_material_gate, prepare_shared_output_dir)


DECLARED = 'terrain-sha'
RANGES = {'static_friction_range': [1., 1.],
          'dynamic_friction_range': [1., 1.],
          'restitution_range': [0., 0.], 'num_buckets': 1,
          'make_consistent': True}


def evidence():
    links, collisions, tensor = fixture()
    return {'terrain': [{'binding_status': 'bound', 'attribute_error': None,
                         'composed_canonical_sha256': DECLARED}],
            'collisions': collisions, 'links': links, 'tensor': tensor,
            'startup': deepcopy(RANGES)}


def run(e=None):
    e = e or evidence()
    return assess_material_gate(e['terrain'], e['collisions'], e['links'], e['tensor'],
                                e['startup'], RANGES, DECLARED)


def test_all_uniform_shapes_pass_with_unbound_usd():
    result = run()
    assert result['pass']
    assert result['robot_effective_physx']['mapped_collision_count'] == 12
    assert result['robot_effective_physx']['unbound_collision_count'] == 8


@pytest.mark.parametrize('mutation', ['nonuniform', 'shape_count', 'missing', 'nonfinite',
                                       'startup', 'terrain_binding', 'terrain_attribute'])
def test_fail_closed(mutation):
    e = evidence()
    if mutation == 'nonuniform': e['tensor'][0][1] = [0.8, 0.6, 0.0]
    elif mutation == 'shape_count': e['collisions'].pop()
    elif mutation == 'missing': e['tensor'] = None
    elif mutation == 'nonfinite': e['tensor'][0][0][0] = float('nan')
    elif mutation == 'startup': e['startup']['static_friction_range'] = [.8, .8]
    elif mutation == 'terrain_binding': e['terrain'][0]['binding_status'] = 'unbound'
    elif mutation == 'terrain_attribute': e['terrain'][0]['composed_canonical_sha256'] = 'wrong'
    result = run(e)
    assert not result['pass']
    assert result['failures']


def test_prepath_output_directory_allows_host_approval_marker(tmp_path):
    output = tmp_path / 'output'
    prepare_shared_output_dir(output)
    assert output.stat().st_mode & 0o777 == 0o777
    (output / 'prepath-approval.json.tmp').write_text('{}')
    assert (output / 'prepath-approval.json.tmp').exists()
