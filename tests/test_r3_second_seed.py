import hashlib
from pathlib import Path

from visualnav_transformer.eat_nav.r3_second_seed import choose_second_seed
from visualnav_transformer.eat_nav.r3_robustness import perturb_pose, summarize_perturbations


def test_seed_is_big_endian_sha256_and_avoids_all_first_seeds():
    first = int.from_bytes(hashlib.sha256(b'eatnav-r3-second-seed-v1').digest()[:4], 'big')
    ordinary = choose_second_seed(set())
    assert ordinary['batch_seed'] == first
    assert ordinary['salt'] == 'eatnav-r3-second-seed-v1'
    retry = choose_second_seed({first})
    assert retry['salt'] == 'eatnav-r3-second-seed-v1|1'
    assert retry['batch_seed'] != first
    assert len(set(retry['terrain_seeds'].values())) == 7


def test_seed_freeze_is_independent_of_geometry_and_outcomes():
    first = choose_second_seed(set())
    assert first == choose_second_seed(set())
    assert first['batch_seed'] != 20261007
    assert 20261007 not in first['terrain_seeds'].values()


def test_robot_local_translation_rotates_into_map_and_nominal_is_first():
    import math
    x,y,yaw=perturb_pose((1.0,2.0,math.pi/2),(.02,0.0),2.0)
    assert abs(x-1.0)<1e-12 and abs(y-2.02)<1e-12
    assert abs(yaw-(math.pi/2+math.radians(2)))<1e-12
    assert perturb_pose((1,2,0),(0,0),0)==(1,2,0)


def test_robustness_keeps_invalid_conditions_in_denominator():
    rows=[{'class':'transition','onset_m':.35,'known_fraction':.9,
           'slope':.2,'roughness':.01,'step':.06,'unknown_only':False,'valid':True} for _ in range(15)]
    rows[1]={**rows[1],'class':'graded','onset_m':.45}
    rows[2]={**rows[2],'valid':False,'class':'unknown','onset_m':None,'unknown_only':True}
    result=summarize_perturbations(rows)
    assert result['class_retention_fraction']==13/15
    assert result['interaction_retention_fraction']==14/15
    assert result['perturbation_valid_count']==14
    assert result['unknown_only_count']==1
    assert result['onset_range_m']==.1


def test_second_scan_has_seed_freeze_and_zero_step_robustness_contract():
    source=(Path(__file__).resolve().parents[1]/'scripts/eat_nav_r3_second_scan_sim.py').read_text()
    assert "seed_freeze['batch_seed']" in source
    assert "bare._sim_step_counter" in source
    assert 'summarize_perturbations' in source
    assert 'OFFSETS_LOCAL_M' in source and 'YAW_OFFSETS_DEG' in source
    assert '20261007' not in source
    assert 'env.step(' not in source
