"""Numerical, censoring, frame, and leakage contracts for 2s evaluation."""
import importlib
import numpy as np
import pytest


def metrics():
    return importlib.import_module('visualnav_transformer.eat_nav.moving_pose2s_metrics')


def test_constant_twist_integrates_rotation_and_translation_in_full_body():
    m=metrics();p,r=m.constant_body_twist([[1,0,0],[0,0,1]],[[0,0,np.pi/4],[0,np.pi/4,0]])
    np.testing.assert_allclose(p[0],[4/np.pi,4/np.pi,0],atol=1e-10)
    np.testing.assert_allclose(p[1],[4/np.pi,0,4/np.pi],atol=1e-10)
    np.testing.assert_allclose(r[0],[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-10)
    p,_=m.constant_body_twist([[.1,.2,.3]],[[0,0,0]])
    np.testing.assert_allclose(p,[[.2,.4,.6]])


def test_candidate_arc_reverse_and_yaw_projection_do_not_mix_body_xy():
    m=metrics();from visualnav_transformer.eat_nav.bidirectional_bank import bank
    b=bank();p,y=m.candidate_arc(np.stack([b[0].waypoints,b[3].waypoints,b[10].waypoints]),[1,1,-1])
    np.testing.assert_allclose(p[0],[.6,0],atol=1e-7)
    np.testing.assert_allclose(y,[0,-np.pi*.8/3,-np.pi*.8/3],atol=1e-6)
    np.testing.assert_allclose(p[1],-p[2],atol=1e-7)
    q=np.array([[np.sqrt(.5),0,np.sqrt(.5),0]])
    np.testing.assert_allclose(m.body_to_initial_yaw([[1,0,0]],q),[[0,0,-1]],atol=1e-7)


def test_event_metrics_censoring_ties_calibration_and_single_class_na():
    m=metrics();y=np.array([1,0,1,np.nan]);p=np.array([.8,.2,.8,.99]);r=m.event_metrics(y,p,parents=['a','b','a','c'])
    assert (r['positive'],r['negative'],r['unknown'],r['independent_positive_parents'])==(2,1,1,1)
    assert r['ap']==1 and r['auroc']==1
    assert r['brier']==pytest.approx(.04)
    assert sum(b['count'] for b in r['reliability_10_bins'])==3
    assert r['threshold_0.5']['f1']==1
    tied=m.event_metrics([0,1],[.5,.5]);assert tied['auroc']==.5 and tied['ap']==.5
    single=m.event_metrics([0,0],[.1,.2]);assert single['auroc'] is None and single['ap'] is None
    assert single['na_reason']=='both classes required'
    with pytest.raises(ValueError):m.event_metrics([0,1],[.2,np.nan])


def test_pose_errors_mask_invalid_rotation_and_wrap_rpy():
    m=metrics();from visualnav_transformer.eat_nav.moving_pose2s_geometry import matrix_to_sixd
    angle=np.deg2rad(179);c,s=np.cos(angle),np.sin(angle)
    gt=np.array([[[c,-s,0],[s,c,0],[0,0,1]]]);pred=gt.transpose(0,2,1)
    e=m.pose_errors([[1,2,3]],[[0,0,0]],pred,gt,[[0,0,-1]],[True])
    assert e['so3_rad'][0]==pytest.approx(np.deg2rad(2))
    assert abs(e['rpy_rad'][0,2])==pytest.approx(np.deg2rad(2))
    report=m.pose_metrics(e)
    assert report['position_3d_m']['mean']==pytest.approx(np.sqrt(14))
    assert report['position_axis_m']['z']['rmse']==3
    assert report['rotation_so3_rad']['count']==1
    e=m.pose_errors([[1,2,3]],[[0,0,0]],pred,gt,[[0,0,-1]],[True],[False])
    assert m.pose_metrics(e)['rotation_so3_rad']['count']==0
    assert m.pose_metrics(e)['position_3d_m']['count']==1


def test_parent_bootstrap_uses_parents_not_windows_and_paired_differences():
    m=metrics();r=m.parent_bootstrap([0,0,10],['a','a','b'],resamples=200,seed=123)
    assert r['independent_parents']==2 and r['estimate']==pytest.approx(10/3)
    assert r['low']==0 and r['high']==10
    paired=m.parent_bootstrap([1,2,3],['a','a','b'],reference=[2,3,4]);assert paired['low']==-1 and paired['high']==-1


def test_stratification_uses_actual_body_velocity_not_track_speed():
    m=metrics();a={'body_state':np.array([[0,0,-1,.03,0,0,0,0,.3],[0,0,-1,.21,0,0,0,0,0]]),'current_quat_wxyz':np.array([[1,0,0,0],[1,0,0,0]]),'directions':np.ones((2,1,1)),'pose_mask':np.array([[False],[True]])}
    rows=[{'field':'x','level':4,'is_initial':True},{'field':'y','level':2,'is_initial':False}]
    g=m.state_groups(a,rows)
    assert g['speed:<0.05'].tolist()==[True,False]
    assert g['abs_wz:>=0.2'].tolist()==[True,False]
    assert g['pose:early_terminal'].tolist()==[True,False]


def test_evaluation_retains_early_event_and_train_only_prevalence():
    m=metrics();n=3
    a=dict(position_targets=np.zeros((n,1,3)),rotation_targets=np.tile([1,0,0,0,1,0],(n,1,1)),pose_mask=np.array([[False],[True],[True]]),event_targets=np.array([[[1,0,np.nan,0]],[[0,0,0,0]],[[1,0,1,0]]]),event_mask=np.array([[[1,1,0,1]],[[1,1,1,1]],[[1,1,1,1]]],bool),candidate_mask=np.ones((n,1),bool),body_state=np.tile([0,0,-1,0,0,0,0,0,0],(n,1)),current_quat_wxyz=np.tile([1,0,0,0],(n,1)),paths=np.zeros((n,1,8,2)),directions=np.ones((n,1,1)))
    pred=dict(schema='eatnav_moving_pose2s_prediction_v1',horizon_sec=2.,delta_position_body_m=np.zeros((2,1,3)),delta_quaternion_wxyz=np.tile([1,0,0,0],(2,1,1)),event_probability=np.full((2,1,4),.2),validity={'rotation_valid':np.ones((2,1),bool)})
    index=[{'parent_id':str(i),'field':'x','level':4,'is_initial':i==0} for i in range(n)]
    r=m.evaluate_predictions(a,index,[0,1],pred,train_indices=[2],bootstrap_resamples=20)
    assert r['model']['pose']['position_3d_m']['count']==1
    assert r['model']['events']['tip_over']['positive']==1
    assert r['baseline_event_train_prevalence']==[1.,0.,1.,0.]
    assert r['model']['events']['linear_stall']['unknown']==1
    assert r['candidate_arc']['scope']=='initial_yaw_horizontal_XY_and_world_yaw_only'
    pred['horizon_sec']=10
    with pytest.raises(ValueError,match='schema|horizon'):m.evaluate_predictions(a,index,[0,1],pred,train_indices=[2])


def test_multi_candidate_strata_keep_correct_parent_denominators():
    m=metrics();n=2;k=2
    a=dict(position_targets=np.zeros((n,k,3)),rotation_targets=np.tile([1,0,0,0,1,0],(n,k,1)),pose_mask=np.ones((n,k),bool),event_targets=np.zeros((n,k,4)),event_mask=np.ones((n,k,4),bool),candidate_mask=np.ones((n,k),bool),body_state=np.tile([0,0,-1,0,0,0,0,0,0],(n,1)),paths=np.tile(np.stack([np.linspace(0,.75,8),np.zeros(8)],-1),(n,k,1,1)),directions=np.ones((n,k,1)))
    a['event_targets'][0,:,0]=1
    p=dict(schema='eatnav_moving_pose2s_prediction_v1',horizon_sec=2.,delta_position_body_m=np.zeros((n,k,3)),delta_quaternion_wxyz=np.tile([1,0,0,0],(n,k,1)),event_probability=np.zeros((n,k,4)))
    r=m.evaluate_predictions(a,[{'parent_id':'a','field':'x'},{'parent_id':'b','field':'y'}],[0,1],p,train_indices=[1],bootstrap_resamples=10)
    assert r['state_stratified']['all']['models']['model']['events']['tip_over']['independent_positive_parents']==1
    assert r['state_stratified']['field:x']['models']['model']['events']['tip_over']['positive']==2


def test_cli_synthetic_output_provenance_and_14_candidate_benchmark(tmp_path):
    import importlib.util,json
    from pathlib import Path
    from visualnav_transformer.eat_nav import moving_pose2s_model as model
    from test_moving_pose2s_model import arrays
    a=arrays();a['event_targets'][0,0,0]=0
    dataset=tmp_path/'dataset';dataset.mkdir();np.savez_compressed(dataset/'samples.npz',**a)
    (dataset/'sample-index.json').write_text(json.dumps([dict(field='synthetic',level=0,parent_id=str(i),is_initial=i==0) for i in range(4)]))
    (dataset/'split.json').write_text(json.dumps(dict(roles=dict(train=[0,1],validation=[2,3],audit_eval=[]))))
    (dataset/'manifest.json').write_text(json.dumps(dict(schema='eatnav_moving_pose2s_dataset_v1',horizon_sec=2.)))
    checkpoint=tmp_path/'model.pt';norm=model.fit_normalization(a,[0,1]);support=model.training_support(a,[0,1]);model.save_checkpoint(checkpoint,model.MovingPose2sEvaluator(),norm,support,{})
    path=Path(__file__).resolve().parents[2]/'scripts/eat_nav_moving_pose2s_evaluate.py'
    spec=importlib.util.spec_from_file_location('pose_evaluate',path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    result=cli.evaluate(checkpoint,dataset,tmp_path/'output',role='validation',benchmark_warmup=1,benchmark_repetitions=2,bootstrap_resamples=10,compare_checkpoints=[checkpoint])
    assert result['metadata']['audit_evaluation_opened'] is False
    assert len(result['metadata']['checkpoint_sha256'])==64
    assert result['inference_benchmark']['candidates']==14
    assert result['inference_benchmark']['candidate_permutation_verified']
    assert result['inference_benchmark']['single_candidate_consistency_verified']
    assert (tmp_path/'output'/'state-stratified.csv').exists()
    assert (tmp_path/'output'/'report.md').exists()
    assert 'Moving-only same-holdout model comparison' in (tmp_path/'output'/'report.md').read_text()
    assert any(row['stratum']=='reference:moving' for row in __import__('csv').DictReader((tmp_path/'output'/'state-stratified.csv').open()))
    with pytest.raises(ValueError,match='role'):cli.evaluate(checkpoint,dataset,tmp_path/'bad',role='train')


def cli_module():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'scripts/eat_nav_moving_pose2s_evaluate.py'
    spec=importlib.util.spec_from_file_location('pose_eval_contract',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_split_audit_reports_shared_geometry_and_rejects_parent_leakage():
    c=cli_module();index=[dict(parent_id='a',geometry_sha256='same'),dict(parent_id='b',geometry_sha256='same'),dict(parent_id='c',geometry_sha256=None)]
    split=dict(roles=dict(train=[0],validation=[1],audit_eval=[2]))
    r=c.split_diagnostics(index,split)
    assert r['geometry_hash_overlap']['train__validation']['count']==1
    assert r['geometry_hash_missing_by_role']['audit_eval']==1
    index[1]['parent_id']='a'
    with pytest.raises(ValueError,match='parent'):c.verify_roles(index,split)


def test_batched_inference_current_inputs_only_and_compare_identical_models():
    c=cli_module();from visualnav_transformer.eat_nav import moving_pose2s_model as model
    from test_moving_pose2s_model import arrays
    a=arrays();norm=model.fit_normalization(a,[0,1]);net=model.MovingPose2sEvaluator();support=model.training_support(a,[0,1]);p=c.predict_batched(net,norm,a,[2,3],support,batch_size=1)
    # Future labels and trajectory are not passed across the model boundary.
    a['position_targets'][:]=np.nan;a['event_targets'][:]=np.nan;a['trajectory_position']=np.full((4,1,40,3),np.nan)
    other=c.predict_batched(net,norm,a,[2,3],support,batch_size=1)
    np.testing.assert_array_equal(p['delta_position_body_m'],other['delta_position_body_m'])
    a=arrays();r=c.paired_model_comparison(a,[dict(parent_id=str(i)) for i in range(4)],[2,3],p,p,resamples=10)
    assert r['position_3d_m']['estimate']==0 and r['tip_over_brier']['low']==0


def test_unknown_censored_heads_not_counted_as_known_negative():
    r=metrics().event_metrics([0,1,np.nan],[.1,.9,.2],[False,True,False],['a','b','c'])
    assert r['negative']==0 and r['unknown']==2 and r['positive']==1
    assert r['auroc'] is None and r['ap'] is None
    assert r['threshold_0.5']['fpr'] is None


def test_paired_comparison_separates_moving_effect_from_initial_errors():
    import copy
    from test_moving_pose2s_model import arrays
    c=cli_module();a=arrays(n=3);a['position_targets'][:]=0;a['event_targets'][:]=0;a['body_state'][1:,3]=.1
    index=[dict(parent_id='a',field='synthetic',is_initial=True),dict(parent_id='a',field='synthetic',is_initial=False),dict(parent_id='b',field='synthetic',is_initial=False)]
    current=dict(delta_position_body_m=np.array([[[8.,0,0]],[[1.,0,0]],[[1.,0,0]]]),delta_quaternion_wxyz=np.array([[[np.cos(.05),0,0,np.sin(.05)]],[[np.cos(.1),0,0,np.sin(.1)]],[[np.cos(.1),0,0,np.sin(.1)]]]),event_probability=np.full((3,1,4),.2),validity=dict(rotation_valid=np.ones((3,1),bool)))
    reference=copy.deepcopy(current);reference['delta_position_body_m']=np.array([[[0.,0,0]],[[3.,0,0]],[[3.,0,0]]]);reference['delta_quaternion_wxyz']=np.array([[[1.,0,0,0]],[[np.cos(.3),0,0,np.sin(.3)]],[[np.cos(.3),0,0,np.sin(.3)]]]);reference['event_probability'][:]=.4
    result=c.paired_model_comparison(a,index,[0,1,2],current,reference,resamples=20)
    assert result['all_windows']['window_count']==3 and result['all_windows']['independent_parents']==2
    assert result['all_windows']['current']['pose']['position_3d_m']['mean']==pytest.approx(10/3)
    assert result['all_windows']['reference']['pose']['position_3d_m']['mean']==2
    assert result['position_3d_m']['estimate']==pytest.approx(4/3)
    moving=result['moving_only']
    assert moving['window_count']==2 and moving['independent_parents']==2
    assert moving['current']['pose']['position_3d_m']['mean']==1
    assert moving['reference']['pose']['position_3d_m']['mean']==3
    assert moving['current']['pose']['rotation_so3_rad']['mean']==pytest.approx(.2)
    assert moving['reference']['pose']['rotation_so3_rad']['mean']==pytest.approx(.6)
    assert moving['paired_difference_95ci']['position_3d_m']['low']==-2
    assert moving['paired_difference_95ci']['so3_rad']['estimate']==pytest.approx(-.4)
    assert moving['paired_difference_95ci']['tip_over_brier']['estimate']==pytest.approx(-.12)
    assert result['state_stratified']['speed:0.05-0.2']['reference']['pose']['position_3d_m']['mean']==3
    empty=c.paired_model_comparison(a,index,[0],{k:(v[:1] if isinstance(v,np.ndarray) else {'rotation_valid':np.ones((1,1),bool)}) for k,v in current.items()},{k:(v[:1] if isinstance(v,np.ndarray) else {'rotation_valid':np.ones((1,1),bool)}) for k,v in reference.items()},resamples=20)
    assert empty['moving_only']['window_count']==0
    assert empty['moving_only']['paired_difference_95ci']['position_3d_m']['estimate'] is None


def test_fresh_audited_collection_states_form_seven_groups_without_pitch_inference():
    """Dataset carries explicit cohort identity even when order gates are reduced."""
    states=('moving_flat','turning','uphill_before','uphill_mid','step_before','descent','slow')
    rows=[]
    for state in states:
        for branch in range(2):
            rows.append(dict(kind='fresh',parent_id='fresh/'+state,collection_state=state,candidate_id='candidate-'+str(branch),is_initial=False,counterfactual_complete=state=='moving_flat',restoration_gate_passed=state=='moving_flat'))
    # A reused row with a steep pitch has no explicit fresh cohort identity.
    rows.append(dict(kind='r4',parent_id='reused',is_initial=False,qualified_state='legacy_qualified'))
    n=len(rows);q=np.tile([1.,0,0,0],(n,1));q[-1]=[np.cos(np.deg2rad(15)),0,np.sin(np.deg2rad(15)),0]
    arrays=dict(body_state=np.tile([0,0,-1,.1,0,0,0,0,0],(n,1)),current_quat_wxyz=q,pose_mask=np.ones((n,1),bool),directions=np.ones((n,1,1)))
    arrays['body_state'][-1,:3]=[np.sin(np.deg2rad(30)),0,-np.cos(np.deg2rad(30))]
    groups=metrics().state_groups(arrays,rows)
    assert {key for key in groups if key.startswith('collection_state:')}=={'collection_state:'+state for state in states}
    for ordinal,state in enumerate(states):
        mask=groups['collection_state:'+state]
        assert np.flatnonzero(mask).tolist()==[2*ordinal,2*ordinal+1]
        assert not mask[-1]
    assert np.flatnonzero(groups['qualified_state:legacy_qualified']).tolist()==[n-1]
