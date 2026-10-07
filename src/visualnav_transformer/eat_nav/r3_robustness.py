"""Outcome-blind, descriptive DEM class robustness for fixed local pose offsets."""
from __future__ import annotations

from collections import Counter
import math

OFFSETS_LOCAL_M=((0.,0.),(.02,0.),(-.02,0.),(0.,.02),(0.,-.02))
YAW_OFFSETS_DEG=(0.,2.,-2.)
INTERACTION_CLASSES=frozenset(('transition','differential','graded'))


def perturb_pose(pose: tuple[float,float,float], offset: tuple[float,float], yaw_deg: float):
    x,y,yaw=pose;dx,dy=offset;c,s=math.cos(yaw),math.sin(yaw)
    return (x+c*dx-s*dy,y+s*dx+c*dy,yaw+math.radians(yaw_deg))


def summarize_perturbations(rows: list[dict]) -> dict:
    if len(rows)!=15 or not rows[0]['valid']:
        raise ValueError('15 fixed perturbations with valid nominal first required')
    nominal=rows[0];cls=nominal['class'];interaction=cls in INTERACTION_CLASSES
    onsets=[x['onset_m'] for x in rows if x['valid'] and x['onset_m'] is not None]
    ranges={}
    for key in ('known_fraction','slope','roughness','step'):
        values=[x[key] for x in rows if x['valid'] and x.get(key) is not None]
        ranges[key]={'min':min(values) if values else None,'max':max(values) if values else None,
                     'range':max(values)-min(values) if values else None}
    return {'schema':'eatnav_r3_robustness_summary_v1','nominal_class':cls,
        'class_retention_fraction':sum(x['valid'] and x['class']==cls for x in rows)/15,
        'interaction_retention_fraction':sum(x['valid'] and ((x['class'] in INTERACTION_CLASSES)==interaction) for x in rows)/15,
        'perturbation_valid_count':sum(x['valid'] for x in rows),
        'unknown_only_count':sum(x.get('unknown_only',False) for x in rows),
        'class_switch_counts':dict(Counter(x['class'] for x in rows if x['class']!=cls)),
        'onset_mean_m':sum(onsets)/len(onsets) if onsets else None,
        'onset_min_m':min(onsets) if onsets else None,
        'onset_max_m':max(onsets) if onsets else None,
        'onset_range_m':round(max(onsets)-min(onsets),12) if onsets else None,
        'metric_ranges':ranges,
        'nominal_threshold_margin':{
            'graded_slope':None if nominal.get('slope') is None else nominal['slope']-.12,
            'transition_step':None if nominal.get('step') is None else nominal['step']-.05,
            'height_change':None if nominal.get('height_change') is None else abs(nominal['height_change'])-.04,
            'differential':None if nominal.get('lateral') is None else nominal['lateral']-.04}}
