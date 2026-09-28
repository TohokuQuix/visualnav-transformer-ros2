"""Train-only empirical Quality and valid action-delta statistics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EmpiricalCDF:
    values: tuple[float, ...]

    @classmethod
    def fit(cls, raw_values):
        values=tuple(sorted(float(v) for v in raw_values))
        if not values or not np.isfinite(values).all():
            raise ValueError('train Quality values must be finite and nonempty')
        return cls(values)

    def transform(self, raw):
        return float(np.searchsorted(self.values, float(raw), side='right') / len(self.values))


@dataclass(frozen=True)
class ActionStatistics:
    mean: np.ndarray
    std: np.ndarray
    count: int

    def normalize(self, action):
        return (np.asarray(action, dtype=np.float32)-self.mean)/self.std

    def denormalize(self, action):
        return np.asarray(action, dtype=np.float32)*self.std+self.mean


def p10(raw_values):
    values=np.asarray(raw_values,dtype=np.float32)
    if values.size==0 or not np.isfinite(values).all():
        raise ValueError('raw Critic sequence must be finite and nonempty')
    return float(np.percentile(values,10))


def fit_train_statistics(train_windows):
    windows=list(train_windows)
    if not windows:
        raise ValueError('train windows are empty')
    valid=[]
    raw=[]
    for window in windows:
        actions=np.asarray(window['action'],dtype=np.float32)
        mask=np.asarray(window['action_mask'],dtype=bool)
        if actions.ndim!=2 or actions.shape[1]!=2 or mask.shape!=(len(actions),):
            raise ValueError('action/mask shape mismatch')
        valid.append(actions[mask])
        raw.append(p10(window['critic_raw']))
    actions=np.concatenate(valid,axis=0)
    if not len(actions):
        raise ValueError('no valid train actions')
    mean=actions.mean(axis=0)
    std=np.maximum(actions.std(axis=0),1e-6)
    return ActionStatistics(mean.astype(np.float32),std.astype(np.float32),len(actions)),EmpiricalCDF.fit(raw)
