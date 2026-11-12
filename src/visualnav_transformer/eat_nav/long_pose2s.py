"""30s experiment-only analysis; no planner integration or history features."""
import numpy as np
from .moving_pose2s_geometry import quat_to_matrix

def horizontal_overprediction(pred_body, actual_body, q0):
    R=quat_to_matrix(np.asarray(q0,float))
    return float(np.linalg.norm((R@np.asarray(pred_body))[:2])-np.linalg.norm((R@np.asarray(actual_body))[:2]))


def rate_summary(rows):
    good=[r for r in rows if r['valid']]
    if any(r['excess_m'] is None or not np.isfinite(r['excess_m']) for r in good):raise ValueError('valid window requires finite signed horizontal excess')
    n=len(good);positive=sum(r['excess_m']>=.4 for r in good)
    return dict(valid_windows=n,positive_windows=positive,rate=positive/n if n else None,independent_trials=len({r['parent_id'] for r in good}),excluded_windows=len(rows)-n)
