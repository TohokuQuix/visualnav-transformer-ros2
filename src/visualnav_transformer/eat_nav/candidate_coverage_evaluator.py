"""Secondary Evaluator ranking diagnostics on completed oracle candidates."""
from __future__ import annotations
import math
import numpy as np


def ranking_diagnostics(predictions,oracle_values,*,epsilon):
    """Compare one Evaluator head's ensemble [seed,candidate,1] to oracle."""
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=3 or p.shape[2]!=1 or p.shape[1]!=len(oracle_values):
        raise ValueError('Evaluator predictions must be [seed,candidate,1]')
    if not np.isfinite(p).all() or epsilon<0:
        raise ValueError('nonfinite predictions or invalid epsilon')
    mean=p[:,:,0].mean(axis=0)
    valid=[i for i,value in enumerate(oracle_values)
           if value is not None and math.isfinite(float(value))]
    if not valid:
        return {'top1_recall':None,'top3_recall':None,'pairwise_accuracy':None,
                'pairwise_support':0,'calibration_rmse':None,
                'ensemble_spread_mean':float(np.std(p[:,:,0],axis=0).mean())}
    best=max(float(oracle_values[i]) for i in valid)
    oracle_winners={i for i in valid if float(oracle_values[i])>=best-epsilon}
    order=np.argsort(-mean,kind='stable')
    correct=[]
    for position,i in enumerate(valid):
        for j in valid[position+1:]:
            gap=float(oracle_values[i])-float(oracle_values[j])
            if abs(gap)<=epsilon:continue
            correct.append(bool((mean[i]-mean[j])*gap>0))
    errors=[(mean[i]-float(oracle_values[i]))**2 for i in valid]
    return {'top1_recall':bool(order[0] in oracle_winners),
            'top3_recall':bool(any(i in oracle_winners for i in order[:3])),
            'pairwise_accuracy':float(np.mean(correct)) if correct else None,
            'pairwise_support':len(correct),
            'calibration_rmse':float(np.sqrt(np.mean(errors))),
            'ensemble_spread_mean':float(np.std(p[:,:,0],axis=0).mean()),
            'predicted_order':order.tolist(),
            'oracle_best':best}
