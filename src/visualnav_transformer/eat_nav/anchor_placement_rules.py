"""Outcome-blind deterministic selection and support summaries for radial poses."""
from __future__ import annotations

from collections import Counter,defaultdict
import math
import statistics

CLASS_ORDER=('transition','differential','graded','flat-only')


def _valid(item):
    return bool(item.get('geometry_valid') and item.get('dem_complete') and
                item.get('classification') in CLASS_ORDER and
                not item.get('unknown_only_difference'))


def select_r2(candidates: list[dict]) -> list[dict]:
    """Minimum sampled radius for a known 0.10–0.50 m onset in each direction."""
    groups=defaultdict(list)
    for item in candidates:
        groups[(item['terrain'],item['direction_index'])].append(item)
    selected=[]
    for key in sorted(groups):
        feasible=[x for x in groups[key] if _valid(x) and x.get('centerline_known') is True and
                  x.get('corridor_known_fraction',0)>=.70 and
                  x.get('interaction_onset_m') is not None and
                  .10<=x['interaction_onset_m']<=.50]
        if feasible:selected.append(min(feasible,key=lambda x:(x['radius_m'],x['pose_id'])))
    return selected


def _profile_distance(left,right):
    a=left['profile_vector_m'];b=right['profile_vector_m']
    if len(a)!=len(b) or not a or not all(math.isfinite(v) for v in a+b):
        raise ValueError('invalid fixed-length terrain profile')
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))


def select_r3(candidates: list[dict], *, budget: int = 112) -> list[dict]:
    """Four-class quota followed by max-min profile selection, no terrain ranking."""
    if budget<1:raise ValueError('positive pose budget required')
    pools={cls:[x for x in candidates if _valid(x) and x['classification']==cls]
           for cls in CLASS_ORDER}
    selected=[];selected_ids=set();quota=budget//len(CLASS_ORDER)
    def choose(cls):
        pool=[x for x in pools[cls] if x['pose_id'] not in selected_ids]
        if not pool:return False
        same=[x for x in selected if x['classification']==cls]
        def score(x):
            profile=x['profile_vector_m']
            if len(profile)!=18 or not all(math.isfinite(v) for v in profile):
                raise ValueError('18-component known terrain profile required')
            distance=min((_profile_distance(x,y) for y in same),default=math.sqrt(sum(v*v for v in profile)/18))
            return (distance,-x['radius_m'],-x['direction_index'])
        item=max(pool,key=score)
        selected.append(item);selected_ids.add(item['pose_id'])
        return True
    for cls in CLASS_ORDER:
        for _ in range(min(quota,len(pools[cls]))):choose(cls)
    while len(selected)<budget:
        changed=False
        for cls in CLASS_ORDER:
            if len(selected)>=budget:break
            changed=choose(cls) or changed
        if not changed:break
    return selected


def summarize_support(candidates: list[dict]) -> dict:
    classes=Counter(x.get('classification','unknown') for x in candidates)
    sources={x['source_family'] for x in candidates}
    footprint=[x['footprint_known_fraction'] for x in candidates if isinstance(x.get('footprint_known_fraction'),(int,float))]
    onsets=[x['interaction_onset_m'] for x in candidates if x.get('interaction_onset_m') is not None]
    return {'candidate_pose_count':len(candidates),
        'geometry_valid_count':sum(bool(x.get('geometry_valid')) for x in candidates),
        'dem_complete_count':sum(bool(x.get('dem_complete')) for x in candidates),
        'class_counts':dict(classes),'insufficient_or_unknown_count':sum(
            x.get('classification') in ('unknown','insufficient') for x in candidates),
        'interaction_within_two_second_proxy_count':sum(
            x.get('interaction_onset_m') is not None and x['interaction_onset_m']<=.6 for x in candidates),
        'source_family_count':len(sources),
        'class_source_family_count':{cls:len({x['source_family'] for x in candidates if x.get('classification')==cls})
                                     for cls in CLASS_ORDER},
        'terrain_counts':dict(Counter(x['terrain'] for x in candidates)),
        'radius_counts':{str(k):v for k,v in sorted(Counter(x['radius_m'] for x in candidates).items())},
        'direction_counts':{str(k):v for k,v in sorted(Counter(x['direction_index'] for x in candidates).items())},
        'related_poses_per_source_family':dict(Counter(x['source_family'] for x in candidates)),
        'footprint_known_fraction':{'min':min(footprint) if footprint else None,
                                    'mean':statistics.mean(footprint) if footprint else None,
                                    'max':max(footprint) if footprint else None},
        'interaction_onset_m':{'min':min(onsets) if onsets else None,
                               'median':statistics.median(onsets) if onsets else None,
                               'max':max(onsets) if onsets else None}}
