from visualnav_transformer.eat_nav.anchor_placement_rules import select_r2, select_r3, summarize_support


def candidate(terrain, direction, radius, cls, onset, profile, *, known=1.0):
    return {'pose_id':f'{terrain}-{direction}-{radius}', 'terrain':terrain,
            'source_family':f'seed:{terrain}', 'direction_index':direction,
            'radius_m':radius,'geometry_valid':True,'dem_complete':True,
            'classification':cls,'interaction_onset_m':onset,
            'corridor_known_fraction':known,'centerline_known':True,
            'profile_vector_m':profile,'unknown_only_difference':False,
            'footprint_known_fraction':1.0}


def test_r2_chooses_smallest_valid_radius_per_terrain_direction():
    items=[candidate('a',0,.3,'flat-only',None,[0]*18),
           candidate('a',0,.4,'graded',.2,[.1]*18),
           candidate('a',0,.5,'differential',.3,[.2]*18),
           candidate('a',1,.3,'graded',.05,[.1]*18),
           candidate('a',1,.4,'graded',.2,[.1]*18,known=.5)]
    selected=select_r2(items)
    assert [x['radius_m'] for x in selected]==[.4]


def test_r3_balances_classes_and_uses_profile_distance():
    items=[]
    for cls in ('transition','differential','graded','flat-only'):
        for index in range(3):
            items.append(candidate(cls,index,.3,cls,.2 if cls!='flat-only' else None,[index*.1]*18))
    selected=select_r3(items,budget=8)
    assert len(selected)==8
    assert {cls:sum(x['classification']==cls for x in selected) for cls in
            ('transition','differential','graded','flat-only')}=={
            'transition':2,'differential':2,'graded':2,'flat-only':2}


def test_support_counts_source_families_not_related_directions():
    items=[candidate('a',i,.3,'differential',.2,[0]*18) for i in range(3)]
    summary=summarize_support(items)
    assert summary['candidate_pose_count']==3
    assert summary['source_family_count']==1
    assert summary['class_source_family_count']['differential']==1
