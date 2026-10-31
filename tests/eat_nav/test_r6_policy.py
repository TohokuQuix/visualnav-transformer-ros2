"""Prediction safety, geometry and validation-only policy derivation."""
import unittest
import numpy as np
from visualnav_transformer.eat_nav import r6_policy as p


def trajectory():
    t=np.zeros((14,20,9));t[:,:,[4,6,8]]=1.;t[:,:,0]=np.linspace(0,.2,20)
    return t


def config():
    return dict(id='synthetic',pitch_deg=30.,roll_deg=20.,step_cm=15.,
                conditional_step_cm={'stairs':15.,'wall':15.,'flat_or_ramp':15.},
                event_thresholds=[.5]*4,tilt_margin_deg=0.)


class PolicyTests(unittest.TestCase):
    def choose(self,t=None,dem=None,profile=None,mode='pose'):
        return p.select_candidate(trajectory() if t is None else t,np.zeros((14,4)),
            np.stack([np.zeros((20,20)),np.ones((20,20))]) if dem is None else dem,
            [1.,0.],config() if profile is None else profile,mode=mode,
            footprint=p.Footprint(.2,.2,.15,.06),event_support=[True]*4)

    def test_unknown_full_corridor_stops(self):
        d=np.zeros((2,20,20));result=self.choose(dem=d)
        self.assertTrue(result['stop']);self.assertTrue(all('map_unknown' in r for r in result['reasons']))

    def test_all_rejected_stops_and_predicted_roll_is_used(self):
        t=trajectory();t[:,:,5]=np.sin(np.deg2rad(25));t[:,:,6]=np.cos(np.deg2rad(25))
        result=self.choose(t=t);self.assertTrue(result['stop'])
        self.assertTrue(all('predicted_roll' in r for r in result['reasons']))

    def test_wrapped_yaw_uses_circular_interpolation(self):
        t=trajectory();angles=np.r_[np.ones(10)*np.deg2rad(179),np.ones(10)*np.deg2rad(-179)]
        t[:,:,3]=np.sin(angles);t[:,:,4]=np.cos(angles)
        self.assertFalse(self.choose(t=t)['stop'])
        self.assertLess(p.circular_error(np.deg2rad(179),np.deg2rad(-179)),np.deg2rad(3))

    def test_supported_flat_prediction_moves(self):
        result=self.choose();self.assertFalse(result['stop']);self.assertEqual(result['selected_index'],0)
        self.assertTrue(all(c==1. for c in result['coverage']))

    def test_stairs_and_wall_geometry_distinct(self):
        stairs=p.longitudinal_geometry(np.arange(12)*.1,np.repeat([0.,.1,.2,.3],3),min_tread_m=.2)
        wall=p.longitudinal_geometry(np.arange(12)*.1,np.r_[np.zeros(6),np.ones(6)*.6],min_tread_m=.2)
        self.assertEqual(stairs['kind'],'stairs');self.assertAlmostEqual(stairs['max_step_m'],.1)
        self.assertEqual(wall['kind'],'wall');self.assertAlmostEqual(wall['max_step_m'],.6)
        self.assertLessEqual(stairs['max_step_m']*100,config()['conditional_step_cm']['stairs'])
        self.assertGreater(wall['max_step_m']*100,config()['conditional_step_cm']['wall'])

    def test_actual_stair_corridor_allowed_wall_rejected(self):
        axis=(np.arange(20)-9.5)*.1
        stair=np.broadcast_to(np.floor((axis+.6)/.3)*.1,(20,20)).copy()
        wall=np.broadcast_to(np.where(axis>=0,.6,0.),(20,20)).copy()
        self.assertFalse(self.choose(dem=np.stack([stair,np.ones((20,20))]))['stop'])
        result=self.choose(dem=np.stack([wall,np.ones((20,20))]))
        self.assertTrue(result['stop'])
        self.assertTrue(all(any(x.startswith('DEM_step_') for x in why) for why in result['reasons']))

    def test_moving_beyond_map_with_full_footprint_stops(self):
        t=trajectory();t[:,:,0]=np.linspace(0,.9,20)
        self.assertTrue(self.choose(t=t)['stop'])

    def test_hybrid_event_veto_and_old_horizon_rejection(self):
        kwargs=dict(footprint=p.Footprint(.2,.2,.1),event_support=[True]*4,mode='hybrid')
        d=np.stack([np.zeros((20,20)),np.ones((20,20))])
        result=p.select_candidate(trajectory(),np.ones((14,4)),d,[1,0],config(),**kwargs)
        self.assertTrue(result['stop']);self.assertIn('event_tip_over',result['reasons'][0])
        with self.assertRaises(ValueError):p.select_candidate(trajectory(),np.zeros((14,4)),d,[1,0],config(),event_horizon_sec=10,**kwargs)

    def test_bad_shapes_and_unsupported_heads_rejected(self):
        with self.assertRaises(ValueError): self.choose(t=np.zeros((14,20,8)))
        bad=config();bad['conditional_step_cm']['wall']=float('nan')
        with self.assertRaises(ValueError):self.choose(profile=bad)
        with self.assertRaises(ValueError):
            p.select_candidate(trajectory(),np.zeros((14,4)),np.ones((2,20,20)),[1,0],config(),
                               footprint=p.Footprint(.2,.2,.1),mode='event',event_support=[True,False,True,True])

    def test_freeze_requires_ten_actual_validation_profiles(self):
        base=dict(split='validation',trials=30,fall_count=0,mean_stop_sec=1.,mean_progress_m=1.,linear_windows=0.,turn_windows=0.,linear_stall_sec=0.,turn_stall_sec=0.)
        profiles=[dict(config(),id=str(i)) for i in range(3)]
        rows=[dict(base,id=str(i)) for i in range(3)]
        with self.assertRaises(ValueError):p.freeze_top_three(profiles,rows)

    def test_lexicographic_fall_priority(self):
        base=dict(split='validation',trials=30,fall_count=0,mean_stop_sec=50.,mean_progress_m=0.,linear_windows=8.,turn_windows=8.,linear_stall_sec=8.,turn_stall_sec=8.)
        worse=dict(base,id='falls',fall_count=1,mean_stop_sec=0.,mean_progress_m=2.)
        better=dict(base,id='safe')
        ranked=p.rank_validation([worse,better]);self.assertEqual(ranked['ranking'][0]['id'],'safe')
        with self.assertRaises(ValueError):p.rank_validation([dict(worse,trials=31),better])

    def test_wilson_empty_and_correlated_groups_are_explicit(self):
        self.assertEqual(p.wilson(0,0),[0.,1.])
        lo,hi=p.wilson(0,30)
        self.assertAlmostEqual(lo,0.);self.assertGreater(hi,.1)
        with self.assertRaises(ValueError):p.wilson(31,30)

    def test_profiles_need_observed_validation_support_and_use_quantiles(self):
        records=[]
        for i in range(60):
            records.append(dict(split='validation',group_id=str(i),failed=i>=40,complete=True,
                max_abs_pitch_deg=float(10+i/2),max_abs_roll_deg=float(5+i/4),step_cm=float(5+i/5),geometry_kind='wall' if i%2 else 'stairs'))
        result=p.derive_profiles(records)
        self.assertEqual(result['status'],'derived');self.assertGreaterEqual(len(result['profiles']),10)
        self.assertNotEqual(result['profiles'][0]['pitch_deg'],55.)
        self.assertTrue(all(x['source']['sample_count']==60 for x in result['profiles']))
        shifted=[dict(x,max_abs_pitch_deg=x['max_abs_pitch_deg']+5) for x in records]
        self.assertAlmostEqual(p.derive_profiles(shifted)['profiles'][0]['pitch_deg']-result['profiles'][0]['pitch_deg'],5.)
        self.assertEqual(p.derive_profiles(records[:5])['status'],'insufficient_support')
        with self.assertRaises(ValueError):p.derive_profiles([dict(records[0],split='test')])
        constant=[dict(x,max_abs_pitch_deg=10.,max_abs_roll_deg=5.,step_cm=2.) for x in records]
        self.assertEqual(p.derive_profiles(constant)['status'],'insufficient_support')

if __name__=='__main__':unittest.main()


def test_low_tilt_stall_failures_do_not_impose_a_monotonic_posture_cap():
    # Stall can occur on level ground: a low failed pitch is not evidence that
    # every larger successful pitch must be forbidden.
    rows=[]
    for i in range(60):
        failed=i>=45
        rows.append(dict(split='validation',group_id=str(i),failed=failed,complete=True,
            max_abs_pitch_deg=.1 if failed else 2+i*.5,
            max_abs_roll_deg=.1 if failed else 1+i*.3,
            step_cm=5+i*.2,geometry_kind='wall' if i%2 else 'stairs'))
    result=p.derive_profiles(rows)
    assert result['status']=='derived' and len(result['profiles'])>=10
    assert all(c['pitch_deg']>.1 for c in result['profiles'])


def test_censored_unknown_fall_cannot_gain_ranking_advantage():
 from visualnav_transformer.eat_nav.r6_policy import rank_validation
 base=dict(split='validation',trials=30,fall_count=0,mean_stop_sec=60.,mean_progress_m=0.,linear_windows=0.,turn_windows=0.,linear_stall_sec=0.,turn_stall_sec=0.)
 censored=dict(base,id='unknown',ranking_fall_count_upper_bound=1,mean_stop_sec=1.)
 complete=dict(base,id='known',ranking_fall_count_upper_bound=0)
 assert rank_validation([censored,complete])['ranking'][0]['id']=='known'


def test_outside_map_prediction_is_rejected_before_dense_allocation(monkeypatch):
 import visualnav_transformer.eat_nav.r6_policy as p
 t=np.zeros((20,9));t[:,0]=1e6;t[:,4]=t[:,6]=t[:,8]=1
 d=np.zeros((2,20,20));d[1]=1
 monkeypatch.setattr(p,'_dense_pose',lambda *args:(_ for _ in ()).throw(AssertionError('must reject before allocation')))
 g=p.swept_geometry(d,t,p.Footprint(.3,.6,.22))
 assert g['unknown'] and g['coverage_is_lower_bound']


def test_single_flat_topped_wall_is_not_a_repeated_staircase():
 from visualnav_transformer.eat_nav.r6_policy import longitudinal_geometry
 x=np.arange(6)*.1
 assert longitudinal_geometry(x,[0,0,.8,.8,0,0])['kind']=='wall'
 assert longitudinal_geometry(np.arange(10)*.1,[0,0,.1,.1,0,0,.1,.1,0,0])['kind']=='wall'
 assert longitudinal_geometry(np.arange(6)*.1,[0,0,.1,.1,.2,.2])['kind']=='stairs'
