from visualnav_transformer.eat_nav.minimal_p1 import select_four, build_manifest, forecast_resource


def anchor(identifier, seed, role, family, *, pair=.0, step=.0, onset=.2, rank=1):
    return {'anchor_id': identifier, 'seed_label': seed, 'primary_role': role,
            'source_family': family, 'selection_rank': rank,
            'physical_eligible': True, 'hard_valid': True, 'follower_compatible': True,
            'dem_history_sha256': ['0','1','2','3'], 'body_state_actor': [0.] * 22,
            'paths': [{'known': True, 'class_proxy': 'transition' if role == 'R-transition-common' else 'differential',
                       'onset_m': onset, 'class_diagnostic': {'maximum_step_m': step}} for _ in range(5)],
            'pair_count': 1 if pair else 0, 'max_pair_rms_m': pair,
            'terrain_seed': 7, 'terrain': 'boxes', 'pose_id': identifier}


def test_four_slots_choose_outcome_blind_margin_and_source():
    rows = [anchor('d1','first','R-differential','A',pair=.02),
            anchor('d2','second','R-differential','B',pair=.03),
            anchor('t1','first','R-transition-common','C',step=.08,onset=.2),
            anchor('t1-weaker','first','R-transition-common','D',step=.06,onset=.1),
            anchor('t2','second','R-transition-common','E',step=.07,onset=.3)]
    assert select_four(rows)['selected'] == ['d1','d2','t1','t2']
    rows[2]['candidate_reward'] = 1000.0
    assert select_four(rows)['selected'] == ['d1','d2','t1','t2']


def test_missing_transition_or_unverified_history_stops():
    rows=[anchor('d1','first','R-differential','A',pair=.02),
          anchor('d2','second','R-differential','B',pair=.03),
          anchor('t1','first','R-transition-common','C',step=.08)]
    assert select_four(rows)['decision'] == 'minimal_p1_anchor_support_invalid'
    rows.append(anchor('t2','second','R-transition-common','D',step=.08))
    rows[-1]['dem_history_sha256'] = ['0']
    assert select_four(rows)['decision'] == 'minimal_p1_anchor_support_invalid'


def test_manifest_has_28_unique_fresh_runs_and_separate_goals():
    rows=[anchor('d1','first','R-differential','A',pair=.02),
          anchor('d2','second','R-differential','B',pair=.03),
          anchor('t1','first','R-transition-common','C',step=.08),
          anchor('t2','second','R-transition-common','D',step=.08)]
    curves=[{'bearing_deg': bearing,'path_sha256':str(bearing),'endpoint':[.75, bearing/100]}
            for bearing in (-20,-10,0,10,20)]
    result=build_manifest(rows,curves,output_root='/tmp/p1-fixed',source_sha={'checkpoint':'abc'},seed=123)
    items=result['rows']
    assert len(items)==28 and len({x['physical_run_id'] for x in items})==28
    assert sum(x['primitive_bearing_deg']==0 for x in items)==12
    assert sum(x['primitive_bearing_deg']!=0 for x in items)==16
    assert all(x['navigation_global_goal']['base_xy_m']==[.75,0.] for x in items)
    assert any(x['local_primitive_endpoint']['base_xy_m']!=x['navigation_global_goal']['base_xy_m'] for x in items)
    assert all(x['candidate_retry_count']==0 and x['random_seed']==123 for x in items)


def test_gpu_margin_must_fit_hard_cap():
    forecast=forecast_resource({'measured_gpu_device_peak_mib':5949.,
                                'P1_28_processes':{'p95_plus_20_percent_hours':1.8,
                                                  'storage_plus_20_percent_gb':.32},
                                'gpu_device_total_hard_cap_mib':6656},
                               wall_cap_hours=4.,storage_cap_gb=.8)
    assert not forecast['pass']
    assert 'gpu_device_margin' in forecast['failures']
