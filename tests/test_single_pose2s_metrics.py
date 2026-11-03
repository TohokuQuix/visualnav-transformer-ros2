import numpy as np
from visualnav_transformer.eat_nav.single_pose2s_metrics import evaluate_records


def record(i,**kw):
    return dict(parent_id=str(i),field='k_rail_field',level=4,candidate_id=0,stage=1,ground_truth_valid=True,prediction_valid=True,longitudinal_cm=1.,lateral_cm=1.,height_cm=99.,roll_deg=1.,pitch_deg=1.,yaw_deg=1.,euler_singular=False,rotation_deg=1.,distance_3d_cm=99.,**kw)


def test_490_rows_missing_null_not_zero_and_height_not_gate():
    result=evaluate_records([record(i) for i in range(5)])
    assert len(result['groups'])==490
    g=next(g for g in result['groups'] if g['field']=='k_rail_field' and g['level']==4 and g['candidate_id']==0)
    assert g['pass'] is True and g['height_cm_max']==99
    missing=next(g for g in result['groups'] if g['valid_intervals']==0)
    assert missing['status']=='評価データなし' and missing['lateral_cm_mean'] is None
    assert result['accuracy_achieved'] is False


def test_maximum_not_mean_and_invalid_predictions_stay_denominator():
    rows=[record(i) for i in range(5)];rows[0]['lateral_cm']=11
    result=evaluate_records(rows);g=next(g for g in result['groups'] if g['valid_intervals'])
    assert g['lateral_cm_mean']<10 and not g['pass'] and g['joint_within_fraction']==.8
    rows[0]['prediction_valid']=False
    g=next(g for g in evaluate_records(rows)['groups'] if g['valid_intervals'])
    assert g['valid_intervals']==5 and g['valid_predictions']==4 and g['joint_within_fraction']==.8 and not g['pass']
