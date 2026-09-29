from scripts.eat_nav_direct_episode_compare import compare_signal, first_divergence


def test_numeric_tolerance_and_first_step_are_explicit():
    assert compare_signal([1.,2.],[1.,2.00005],.0001)['pass']
    bad=compare_signal([1.,2.],[1.,2.01],.0001)
    assert not bad['pass'] and bad['max_abs']>.009
    cases=[('anchor.root',-1,compare_signal([1.],[1.],.0001)),
           ('trace.command',3,bad),('trace.action',3,compare_signal([1.],[2.],.0001))]
    assert first_divergence(cases)=={'signal':'trace.command','step':3,'max_abs':bad['max_abs'],'limit':.0001}


def test_shape_mask_and_nonfinite_always_fail():
    assert not compare_signal([1.,2.],[1.],1.)['pass']
    assert not compare_signal([float('nan')],[0.],1.)['pass']
    assert not compare_signal([[1,0]],[[1,1]],0,exact=True)['pass']


def test_nested_delay_state_reports_first_path():
    a={'ring':{'pointer':2,'buffer':[[0.,1.]]}}
    b={'ring':{'pointer':2,'buffer':[[0.,1.01]]}}
    result=compare_signal(a,b,.001)
    assert not result['pass'] and result['first_path']=='ring.buffer[0][1]'


def test_orientation_and_xy_rms_have_separate_frozen_limits():
    from scripts.eat_nav_direct_episode_compare import quaternion_angle_rad, xy_rms
    assert quaternion_angle_rad([1.,0.,0.,0.],[1.,0.,0.,0.])==0
    assert quaternion_angle_rad([1.,0.,0.,0.],[0.,0.,0.,1.])>3.14
    assert xy_rms([[0.,0.],[1.,1.]],[[0.,0.],[1.01,1.]])>.004
