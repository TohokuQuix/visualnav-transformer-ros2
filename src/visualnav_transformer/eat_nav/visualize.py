"""Shape-preserving trajectory visualization inputs."""

import numpy as np


def trajectory_plot_data(xy):
    points=np.asarray(xy,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=2:
        raise ValueError('trajectory must be [len_traj_pred,2]')
    return [tuple(map(float,row)) for row in points]
