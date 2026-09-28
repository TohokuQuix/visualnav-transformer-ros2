"""Complete-only reference training view and train-only normalization."""

from __future__ import annotations

import numpy as np

from .quality import fit_train_statistics
from .quality_followup import is_complete_window, window_identity


class CompleteOnlyView:
    def __init__(self, dataset, point_count: int):
        self.dataset = dataset
        self.indices = [i for i, window in enumerate(dataset.windows)
                        if is_complete_window(window, point_count)]
        self.identities = [window_identity(dataset.windows[i]) for i in self.indices]
        if len(set(self.identities)) != len(self.identities):
            raise ValueError('duplicate complete window identity')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        window = self.dataset.windows[self.indices[index]]
        return {'dem': window['dem'], 'goal': window['goal'],
                'quality': np.asarray([window['quality_raw']], dtype=np.float32),
                'action': window['action'], 'action_mask': window['action_mask']}


def fit_complete_train_statistics(view: CompleteOnlyView):
    windows = []
    for index in view.indices:
        window = view.dataset.windows[index]
        windows.append(dict(window, critic_raw=np.asarray(window['critic_raw'])[window['critic_mask']]))
    return fit_train_statistics(windows)
