"""Episode-local, timestamp-strict DEM history used before inference."""

from collections import deque

import numpy as np

from .wire import actor_from_message, dem_from_message


class DemHistory:
    def __init__(self):
        self.frames = deque(maxlen=4)
        self.episode_id = None
        self.last_valid_sensor_ns = None

    def clear(self):
        self.frames.clear()
        self.episode_id = None
        self.last_valid_sensor_ns = None

    def push(self, message):
        frame, meta = dem_from_message(message)
        if self.episode_id != meta['episode_id']:
            self.clear()
            self.episode_id = meta['episode_id']
        if self.frames and meta['stamp_ns'] <= self.frames[-1][1]['stamp_ns']:
            raise ValueError('DEM stamp duplicate/reversed')
        if self.last_valid_sensor_ns is not None and \
                meta['sensor_stamp_ns'] <= self.last_valid_sensor_ns:
            raise ValueError('DEM sensor update did not advance')
        self.frames.append((frame.copy(), meta))
        self.last_valid_sensor_ns = meta['sensor_stamp_ns']

    def assemble(self, actor_message, now_ns: int):
        state, actor = actor_from_message(actor_message)
        if len(self.frames) != 4 or actor['episode_id'] != self.episode_id:
            raise ValueError('DEM history incomplete/episode mismatch')
        stamps = [meta['stamp_ns'] for _, meta in self.frames]
        if any(b-a != 250_000_000 for a, b in zip(stamps, stamps[1:])) or \
                stamps[-1] != actor['stamp_ns'] or \
                not 0 <= now_ns-stamps[-1] <= 300_000_000 or \
                not 0 <= stamps[-1]-self.last_valid_sensor_ns <= 300_000_000 or \
                any(not 0 <= stamps[-1]-source <= 300_000_000
                    for source in actor['sources'].values()):
            raise ValueError('DEM/Actor timing mismatch/stale')
        return np.stack([frame for frame, _ in self.frames]), state, \
            {'episode_id': self.episode_id, 'dem_stamps_ns': stamps,
             'anchor_stamp_ns': stamps[-1], 'actor_sources': actor['sources'],
             'latest_sensor_stamp_ns': self.last_valid_sensor_ns}
