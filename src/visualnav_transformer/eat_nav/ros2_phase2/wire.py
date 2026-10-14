"""Strict ROS2 codecs for the frozen model input, without model dependencies."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from eat_nav_interfaces.msg import ActorState22, CandidateBank, DemFrame
from visualnav_transformer.eat_nav.bidirectional_bank import SCHEMA, bank
from visualnav_transformer.eat_nav.motion_14_stage_a import candidate_role


BODY_FRAME = 'chassis/body'
DEM_PREPROCESS = 'robot_dem_scalar_median_v1'
ACTOR_CONTRACT = 'onix_policy_pre_noise_22d_v1'
BANK_SOURCE = Path(__file__).resolve().parents[1] / 'bidirectional_bank.py'
CANDIDATES = bank()
CANDIDATE_IDS = tuple(candidate_role(c.direction, c.arc_angle_deg) for c in CANDIDATES)
PATHS = np.asarray([c.waypoints for c in CANDIDATES], dtype=np.float32)
BANK_SHA = hashlib.sha256(BANK_SOURCE.read_bytes()).hexdigest()
WIRE_SHA = hashlib.sha256(np.ascontiguousarray(PATHS).tobytes()).hexdigest()


def set_stamp(field, ns: int) -> None:
    if not isinstance(ns, int) or ns < 0:
        raise ValueError('stamp must be nonnegative integer nanoseconds')
    field.sec, field.nanosec = divmod(ns, 1_000_000_000)


def stamp_ns(field) -> int:
    value = int(field.sec)*1_000_000_000 + int(field.nanosec)
    if value < 0 or not 0 <= int(field.nanosec) < 1_000_000_000:
        raise ValueError('invalid ROS timestamp')
    return value


def _dem_array(value) -> np.ndarray:
    dem = np.asarray(value, dtype=np.float32)
    if dem.shape != (2, 20, 20) or not np.isfinite(dem).all() or \
            not np.isin(dem[1], (0., 1.)).all() or \
            np.any(dem[0][dem[1] == 0] != 0) or dem[1].mean() < .25:
        raise ValueError('DEM shape/finite/mask/valid-fraction contract')
    return np.ascontiguousarray(dem)


def dem_to_message(value, stamp: int, sensor_stamp: int, episode_id: int,
                   source_id: str = 'sealed_replay') -> DemFrame:
    dem = _dem_array(value)
    if sensor_stamp > stamp or episode_id < 0 or not source_id:
        raise ValueError('DEM provenance')
    msg = DemFrame()
    msg.header.frame_id = BODY_FRAME
    set_stamp(msg.header.stamp, stamp)
    set_stamp(msg.sensor_stamp, sensor_stamp)
    msg.episode_id = episode_id
    msg.resolution_m = .1
    msg.preprocess_id = DEM_PREPROCESS
    msg.source_id = source_id
    msg.height_m = dem[0].reshape(-1).tolist()
    msg.valid = dem[1].astype(np.uint8).reshape(-1).tolist()
    return msg


def dem_from_message(msg: DemFrame):
    if msg.header.frame_id != BODY_FRAME or msg.preprocess_id != DEM_PREPROCESS or \
            abs(float(msg.resolution_m)-.1) > 1e-8 or not msg.source_id:
        raise ValueError('DEM frame/resolution/preprocess/source contract')
    t, sensor = stamp_ns(msg.header.stamp), stamp_ns(msg.sensor_stamp)
    if sensor > t or len(msg.height_m) != 400 or len(msg.valid) != 400:
        raise ValueError('DEM stamp/length contract')
    dem = _dem_array(np.stack((np.asarray(msg.height_m, np.float32).reshape(20, 20),
                              np.asarray(msg.valid, np.float32).reshape(20, 20))))
    return dem, {'stamp_ns': t, 'sensor_stamp_ns': sensor,
                 'episode_id': int(msg.episode_id), 'source_id': msg.source_id}


def actor_to_message(value, stamp: int, episode_id: int, sources: dict) -> ActorState22:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (22,) or not np.isfinite(state).all() or episode_id < 0:
        raise ValueError('Actor state contract')
    msg = ActorState22()
    msg.header.frame_id = BODY_FRAME
    set_stamp(msg.header.stamp, stamp)
    for key in ('imu', 'joint', 'command'):
        sample = int(sources[key])
        if sample > stamp:
            raise ValueError('future Actor source')
        set_stamp(getattr(msg, key+'_stamp'), sample)
    msg.episode_id = episode_id
    msg.contract_id = ACTOR_CONTRACT
    msg.values = state.tolist()
    return msg


def actor_from_message(msg: ActorState22):
    stamp = stamp_ns(msg.header.stamp)
    sources = {key: stamp_ns(getattr(msg, key+'_stamp')) for key in
               ('imu', 'joint', 'command')}
    state = np.asarray(msg.values, dtype=np.float32)
    if msg.header.frame_id != BODY_FRAME or msg.contract_id != ACTOR_CONTRACT or \
            state.shape != (22,) or not np.isfinite(state).all() or \
            any(value > stamp for value in sources.values()):
        raise ValueError('Actor wire contract')
    return np.ascontiguousarray(state), {'stamp_ns': stamp,
            'episode_id': int(msg.episode_id), 'contract_id': msg.contract_id,
            'sources': sources}


def bank_to_message(stamp: int) -> CandidateBank:
    msg = CandidateBank()
    msg.header.frame_id = BODY_FRAME
    set_stamp(msg.header.stamp, stamp)
    msg.bank_version = SCHEMA
    msg.bank_source_sha256 = BANK_SHA
    msg.wire_path_sha256 = WIRE_SHA
    msg.candidate_ids = list(CANDIDATE_IDS)
    msg.direction = [c.direction for c in CANDIDATES]
    msg.mask = [True]*14
    msg.body_xy_m = PATHS.reshape(-1).tolist()
    return msg


def bank_from_message(msg: CandidateBank):
    paths = np.asarray(msg.body_xy_m, dtype=np.float32).reshape(14, 8, 2)
    direction = np.asarray(msg.direction, dtype=np.float32).reshape(14, 1)
    mask = np.asarray(msg.mask, dtype=np.bool_)
    if msg.header.frame_id != BODY_FRAME or msg.bank_version != SCHEMA or \
            msg.bank_source_sha256 != BANK_SHA or msg.wire_path_sha256 != WIRE_SHA or \
            tuple(msg.candidate_ids) != CANDIDATE_IDS or \
            not np.array_equal(direction[:, 0], [1]*7+[-1]*7) or \
            not np.array_equal(mask, np.ones(14, dtype=bool)) or \
            not np.array_equal(paths, PATHS) or \
            hashlib.sha256(np.ascontiguousarray(paths).tobytes()).hexdigest() != WIRE_SHA:
        raise ValueError('frozen 14-Path bank mismatch')
    return {'paths': paths, 'direction': direction, 'mask': mask}, \
        {'stamp_ns': stamp_ns(msg.header.stamp), 'ids': CANDIDATE_IDS,
         'bank_source_sha256': BANK_SHA, 'wire_path_sha256': WIRE_SHA}
