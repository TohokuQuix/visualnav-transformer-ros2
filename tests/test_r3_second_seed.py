import hashlib
from pathlib import Path

from visualnav_transformer.eat_nav.r3_second_seed import choose_second_seed
from visualnav_transformer.eat_nav.r3_robustness import perturb_pose, summarize_perturbations
from visualnav_transformer.eat_nav.r3_second_selection import choose_pilot


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


def test_pilot_selection_uses_r3_rank_and_distinct_family_not_robustness():
    classes=('transition','differential','graded','flat-only')
    ordered=[]
    for cls in classes:
        for index,family in enumerate(('a','a','b')):
            ordered.append({'pose_id':f'{cls}-{index}','classification':cls,
                            'source_family':family,'robustness':{'class_retention_fraction':index/10}})
    selected,missing=choose_pilot(ordered)
    assert not missing
    assert [x['pose_id'] for x in selected]==[f'{c}-0' for c in classes]+[f'{c}-2' for c in classes]
    for x in ordered:x['robustness']['class_retention_fraction']=1.0
    assert [x['pose_id'] for x in choose_pilot(ordered)[0]]==[x['pose_id'] for x in selected]
    _,missing=choose_pilot([x for x in ordered if x['classification']!='graded' or x['source_family']=='a'])
    assert missing==['graded']


def test_second_settling_only_changes_batch_seed_guard():
    root=Path(__file__).resolve().parents[1]
    old=(root/'scripts/eat_nav_r3_settling_sim.py').read_text()
    new=(root/'scripts/eat_nav_r3_second_settle_sim.py').read_text()
    assert new==old.replace(
        "if args.batch_seed != 20261007 or args.batch_seed not in freeze['batch_seeds']:",
        "if args.batch_seed != freeze['batch_seeds'][0] or len(freeze['batch_seeds']) != 1:")
    assert 'max_contact_data_count=view_capacity' in new
    assert 'bare._reset_idx=capture_before_reset' in new


def test_second_seed_decision_requires_all_four_class_supplies():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/eat_nav_r3_second_analyze.py'
    spec=importlib.util.spec_from_file_location('r3_second_analyze_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[{'virtual_class':c,'classification':'settled_anchor_eligible',
           'settled_class':c,'physical_supply_second_seed':True} for c in
          ('transition','differential','graded','flat-only') for _ in range(2)]
    assert module.decide(rows)=='r3_second_seed_physical_supply_pass_ready_for_goal_candidate_bank_design'
    rows[0]['physical_supply_second_seed']=False
    rows[1]['physical_supply_second_seed']=False
    assert module.decide(rows)=='r3_second_seed_partial_class_support_review_required'
    rows[0]['settled_class']='flat-only'
    assert module.decide(rows)=='r3_second_seed_class_mismatch_review_required'
    rows[0]['classification']='measurement_invalid'
    assert module.decide(rows)=='r3_second_seed_measurement_invalid'


def test_second_scan_reuses_first_seed_generator_and_collision_proxy():
    import ast
    root=Path(__file__).resolve().parents[1]
    def definition(path,name):
        tree=ast.parse(path.read_text())
        return next(ast.dump(node,include_attributes=False) for node in ast.walk(tree)
                    if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name)
    old=root/'scripts/eat_nav_anchor_placement_scan_sim.py'
    new=root/'scripts/eat_nav_r3_second_scan_sim.py'
    for name in ('RadialTerrainGenerator','collision_proxy','outside_corridor_difference'):
        assert definition(old,name)==definition(new,name)
