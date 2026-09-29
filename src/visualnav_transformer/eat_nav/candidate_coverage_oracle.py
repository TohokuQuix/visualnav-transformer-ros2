"""Descriptive oracle coverage on completed simulator rollouts only."""
from __future__ import annotations
import math
import numpy as np


def oracle_best(rows,candidate_ids,metric):
    wanted=set(candidate_ids)
    values=[float(row[metric]) for row in rows if row['candidate_id'] in wanted
            and row.get(metric) is not None and math.isfinite(float(row[metric]))]
    return max(values) if values else None


def recall_curve(rows,ordered_ids,*,metric,epsilon,ks):
    if epsilon<0:raise ValueError('epsilon must be fixed and nonnegative')
    pool=oracle_best(rows,[row['candidate_id'] for row in rows],metric)
    result=[]
    for k in ks:
        if k<1 or k>len(ordered_ids):raise ValueError('K outside evaluated proposal set')
        best=oracle_best(rows,ordered_ids[:k],metric)
        result.append({'k':k,'best':best,'pool_best':pool,'epsilon':epsilon,
                       'recalled':bool(best is not None and pool is not None and best>=pool-epsilon)})
    return result


def outcome_spread(rows,metric):
    values=np.asarray([float(r[metric]) for r in rows if r.get(metric) is not None
                       and math.isfinite(float(r[metric]))],dtype=float)
    return {'valid_count':int(len(values)),
            'p90_minus_p10':float(np.quantile(values,.9)-np.quantile(values,.1)) if len(values) else None,
            'range':float(values.max()-values.min()) if len(values) else None,
            'min':float(values.min()) if len(values) else None,
            'max':float(values.max()) if len(values) else None}
