"""Full490-cell reporting; unsupported/missing/invalid never become zero error."""
from collections import defaultdict
from itertools import product
import csv,json
from pathlib import Path
import numpy as np
from .demo_terrain import TERRAIN_TYPES
from .bidirectional_bank import bank

METRICS=['longitudinal_cm','lateral_cm','height_cm','roll_deg','pitch_deg','yaw_deg','rotation_deg','distance_3d_cm']
LIMITS=dict(longitudinal_cm=30.,lateral_cm=10.,roll_deg=10.,pitch_deg=10.,yaw_deg=10.)


def summarize(rows):
    valid=[r for r in rows if r['ground_truth_valid']];pred=[r for r in valid if r['prediction_valid'] and not r.get('euler_singular',False)]
    n=len(valid);parents=len({r['parent_id'] for r in valid})
    result=dict(planned_intervals=len(rows),valid_intervals=n,independent_trials=parents,valid_predictions=len(pred),invalid_predictions=n-len(pred))
    for metric in METRICS:
        values=np.array([abs(r[metric]) for r in pred if r.get(metric) is not None and np.isfinite(r[metric])])
        result[metric+'_count']=len(values)
        for name,value in [('mean',np.mean),('median',np.median),('p95',lambda x:np.percentile(x,95)),('max',np.max)]:
            result[metric+'_'+name]=float(value(values)) if len(values) else None
    for metric,limit in LIMITS.items():
        result[metric+'_within_fraction']=sum(r.get(metric) is not None and np.isfinite(r[metric]) and abs(r[metric])<=limit for r in pred)/n if n else None
    result['joint_within_fraction']=sum(all(r.get(m) is not None and np.isfinite(r[m]) and abs(r[m])<=lim for m,lim in LIMITS.items()) for r in pred)/n if n else None
    values=[r['longitudinal_cm'] for r in pred if r.get('longitudinal_cm') is not None and np.isfinite(r['longitudinal_cm'])]
    result['longitudinal_signed_bias_cm']=float(np.mean(values)) if values else None
    result['euler_singular_count']=sum(bool(r.get('euler_singular',False)) for r in valid)
    result['pass']=bool(n>=5 and parents>=5 and len(pred)==n and all(result[m+'_max'] is not None and result[m+'_max']<=lim for m,lim in LIMITS.items()))
    result['status']='評価データなし' if n==0 else '評価件数不足' if n<5 or parents<5 else '合格' if result['pass'] else '未達'
    return result


def evaluate_records(records):
    by=defaultdict(list)
    for row in records:
        key=(row['candidate_id'],row['field'],row['level'])
        if key[0] not in range(14) or key[1] not in TERRAIN_TYPES or key[2] not in range(5):raise ValueError('unexpected group')
        by[key].append(row)
    groups=[]
    for cid,field,level in product(range(14),TERRAIN_TYPES,range(5)):
        c=bank()[cid];groups.append(dict(candidate_id=cid,direction=c.direction,angle_deg=c.arc_angle_deg,field=field,level=level,**summarize(by[(cid,field,level)])))
    strata=[]
    for kind,fn in [('direction',lambda r:bank()[r['candidate_id']].direction),('turn',lambda r:'straight' if bank()[r['candidate_id']].arc_angle_deg==0 else 'left' if bank()[r['candidate_id']].arc_angle_deg>0 else 'right'),('stage',lambda r:r['stage']),('start_phase',lambda r:'initial' if r['stage']==0 else 'continuing'),('start_motion',lambda r:r.get('start_motion','unknown')),('field',lambda r:r['field']),('level',lambda r:r['level'])]:
        bins=defaultdict(list)
        for row in records:bins[str(fn(row))].append(row)
        for key,rows in sorted(bins.items()):strata.append(dict(stratum=kind,value=key,**summarize(rows)))
    return dict(schema='eatnav_single_dem_pose2s_evaluation_v1',pass_statistic='maximum_absolute_error_per_group',required_groups=490,groups=groups,strata=strata,overall=summarize(records),passed_groups=sum(g['pass'] for g in groups),missing_groups=sum(g['valid_intervals']==0 for g in groups),insufficient_groups=sum(g['status']=='評価件数不足' for g in groups),accuracy_achieved=all(g['pass'] for g in groups),thresholds=LIMITS)


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def save_evaluation(output,records,report):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    csv_write(output/'candidate_field_level.csv',report['groups']);csv_write(output/'per_window.csv',records);csv_write(output/'strata.csv',report['strata'])
    coverage=[{k:r[k] for k in ['candidate_id','field','level','planned_intervals','valid_intervals','independent_trials','valid_predictions','status']} for r in report['groups']]
    csv_write(output/'coverage.csv',coverage)
    trials=defaultdict(list)
    for r in records:trials[r['parent_id']].append(r)
    csv_write(output/'trial_summary.csv',[dict(parent_id=p,**summarize(rows)) for p,rows in sorted(trials.items())])
    csv_write(output/'missing_reasons.csv',[{k:r.get(k) for k in ['parent_id','candidate_id','field','level','stage','reason']} for r in records if not r['ground_truth_valid']])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    metrics=['longitudinal_cm','lateral_cm','height_cm','roll_deg','pitch_deg','yaw_deg']
    def table(rows):
        columns=['candidate_id','field','level','valid_intervals','independent_trials','status']+[m+'_'+s for m in metrics for s in ['mean','median','p95','max']]
        lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        for r in rows:lines.append('| '+' | '.join('—' if r.get(k) is None else f'{r[k]:.3f}' if isinstance(r[k],float) else str(r[k]) for k in columns)+' |')
        return '\n'.join(lines)+'\n'
    for kind,values in [('candidate_id',range(14)),('field',TERRAIN_TYPES),('level',range(5))]:
        directory=output/kind;directory.mkdir(exist_ok=True)
        for v in values:(directory/(str(v)+'.md')).write_text(table([r for r in report['groups'] if r[kind]==v]))
    (output/'README.md').write_text('# 独立予測精度評価\n\n490組全体の精度達成：'+str(report['accuracy_achieved'])+'。合格'+str(report['passed_groups'])+'/490、評価なし'+str(report['missing_groups'])+'、件数不足'+str(report['insufficient_groups'])+'。\n\n最大絶対誤差を組別に判定。高さには合否閾値なし。CSVとcandidate_id/field/level別Markdownを参照。\n')
