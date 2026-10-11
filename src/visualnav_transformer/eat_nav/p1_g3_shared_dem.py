"""G3 v2 common robot-relative DEM input; never a native sensor measurement."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .p1_dem_capture import _durable_json, _sha

STEPS = (4, 9, 14, 19)
DT_NS = 50_000_000
ROLE = 'shared_experimental_history'
FRAME = 'robot_base'


def _frame_sha(frame: dict) -> str:
    metadata = {key: frame[key] for key in (
        'identity', 'timestamp_ns', 'shape', 'frame', 'resolution_m',
        'sensor_update_timestamp_s', 'raw_sha256', 'source_env', 'history_role')}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode())
    digest.update(np.asarray(frame['transformed']).tobytes(order='C'))
    return digest.hexdigest()


def build_shared_history(native: list[list[dict]], source_env: int = 0) -> dict:
    if len(native) != 4 or any(len(history) != 4 for history in native):
        raise ValueError('four-env four-frame native DEM required')
    if source_env != 0:
        raise ValueError('G3 v2 source_env must be 0')
    source = []
    for index, step in enumerate(STEPS):
        original = native[source_env][index]
        expected_ns = (step + 1) * DT_NS
        if (original['identity'] != f'settling_step_{step}' or
                original['timestamp_ns'] != expected_ns or
                original['shape'] != [2, 20, 20] or original['frame'] != FRAME):
            raise ValueError('source DEM identity/order/timestamp/shape/frame mismatch')
        array = np.array(original['transformed'], dtype=np.float32, order='C', copy=True)
        if array.shape != (2, 20, 20) or not np.isfinite(array).all():
            raise ValueError('source DEM shape/nonfinite')
        frame = {key: original[key] for key in ('identity', 'timestamp_ns', 'shape',
                  'frame', 'resolution_m', 'sensor_update_timestamp_s', 'raw_sha256')}
        frame.update({'source_env': 0, 'history_role': ROLE, 'transformed': array})
        frame['source_frame_sha256'] = _frame_sha(frame)
        source.append(frame)
    branches = []
    for _ in range(4):
        history = []
        for frame in source:
            replica = {key: value for key, value in frame.items() if key != 'transformed'}
            replica['transformed'] = frame['transformed'].copy(order='C')
            replica['transformed'].setflags(write=False)
            history.append(replica)
        branches.append(history)
    return {'schema': 'eatnav_p1_g3_shared_dem_v2', 'source_env': 0,
            'history_role': ROLE, 'branches': branches}


def validate_shared_history(shared: dict, frozen_gate) -> dict:
    if shared.get('source_env') != 0 or shared.get('history_role') != ROLE:
        raise ValueError('shared DEM provenance mismatch')
    branches = shared.get('branches')
    if not isinstance(branches, list) or len(branches) != 4 or any(len(b) != 4 for b in branches):
        raise ValueError('shared DEM four-env four-frame history incomplete')
    age = []
    for index, step in enumerate(STEPS):
        canonical = branches[0][index]
        for branch in branches:
            frame = branch[index]
            if (frame['identity'] != f'settling_step_{step}' or
                    frame['timestamp_ns'] != (step + 1) * DT_NS):
                raise ValueError('shared DEM timestamp/order/identity mismatch')
            if (frame['shape'] != [2, 20, 20] or frame['frame'] != FRAME or
                    frame['source_env'] != 0 or frame['history_role'] != ROLE or
                    not np.isfinite(frame['resolution_m']) or frame['resolution_m'] <= 0):
                raise ValueError('shared DEM shape/frame/resolution/provenance mismatch')
            if any(frame[key] != canonical[key] for key in ('resolution_m',
                   'sensor_update_timestamp_s', 'raw_sha256', 'source_frame_sha256')):
                raise ValueError('shared DEM metadata/source SHA mismatch')
            array = np.asarray(frame['transformed'])
            if (array.shape != (2, 20, 20) or array.dtype != np.float32 or
                    not array.flags.c_contiguous or not np.isfinite(array).all()):
                raise ValueError('shared DEM bytes/finite mismatch')
            if _frame_sha(frame) != frame['source_frame_sha256'] or \
                    array.tobytes() != np.asarray(canonical['transformed']).tobytes():
                raise ValueError('shared DEM byte integrity mismatch')
        sensor = float(canonical['sensor_update_timestamp_s'])
        delta = (step + 1) * .05 - sensor
        if not np.isfinite(sensor) or delta < -1e-5 or delta > .30 + 1e-5:
            raise ValueError('shared DEM sensor update age invalid')
        age.append(float(round(delta, 9)))
    diagnostics = frozen_gate(branches)
    return {'pass': True, 'meaning': 'shared_experimental_input_equality',
            'native_sensor_equality_claimed': False, 'source_env': 0,
            'source_frame_sha256': [f['source_frame_sha256'] for f in branches[0]],
            'sensor_update_age_s': age, 'frozen_numeric_diagnostics': diagnostics}


def _durable_npy(path: Path, array: np.ndarray) -> None:
    temp = path.with_suffix('.tmp')
    with temp.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def persist_shared_history(output: Path, shared: dict, *, native_diagnostic: dict) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = {}
    metadata = []
    for env, branch in enumerate(shared['branches']):
        for index, frame in enumerate(branch):
            name = f'prepath-shared-dem-env-{env}-frame-{index}.npy'
            _durable_npy(output/name, frame['transformed'])
            ledger[name] = _sha(output/name)
            metadata.append({'branch_env': env, 'array_file': name,
                'array_sha256': ledger[name],
                **{key: frame[key] for key in ('identity', 'timestamp_ns', 'shape',
                    'frame', 'resolution_m', 'sensor_update_timestamp_s', 'source_env',
                    'history_role', 'raw_sha256', 'source_frame_sha256')}})
    _durable_json(output/'prepath-shared-dem-metadata.json', metadata)
    ledger['prepath-shared-dem-metadata.json'] = _sha(output/'prepath-shared-dem-metadata.json')
    _durable_json(output/'prepath-native-dem-diagnostic.json', native_diagnostic)
    ledger['prepath-native-dem-diagnostic.json'] = _sha(output/'prepath-native-dem-diagnostic.json')
    _durable_json(output/'prepath-shared-dem-sha-ledger.json', ledger)
    return {'ledger_sha256': _sha(output/'prepath-shared-dem-sha-ledger.json'),
            'source_frame_sha256': [f['source_frame_sha256'] for f in shared['branches'][0]]}


def verify_shared_artifacts(output: Path, *, require_native: bool = False) -> dict:
    output = Path(output)
    ledger = json.loads((output/'prepath-shared-dem-sha-ledger.json').read_text())
    expected = {f'prepath-shared-dem-env-{env}-frame-{frame}.npy'
                for env in range(4) for frame in range(4)} | {
                'prepath-shared-dem-metadata.json', 'prepath-native-dem-diagnostic.json'}
    if set(ledger) != expected:
        raise ValueError('shared DEM ledger file set mismatch')
    for name, digest in ledger.items():
        if _sha(output/name) != digest:
            raise ValueError('shared DEM artifact SHA mismatch:' + name)
    rows = json.loads((output/'prepath-shared-dem-metadata.json').read_text())
    if len(rows) != 16:
        raise ValueError('shared DEM metadata count mismatch')
    branches = [[] for _ in range(4)]
    for row in rows:
        env = row['branch_env']; name = row['array_file']
        if not 0 <= env < 4 or name not in ledger or row['array_sha256'] != ledger[name]:
            raise ValueError('shared DEM metadata file/SHA mismatch')
        array = np.load(output/name, allow_pickle=False)
        branches[env].append({**row, 'transformed': array})
    # File and metadata validation uses the same source digest and time contract as runtime.
    validate_shared_history({'source_env': 0, 'history_role': ROLE, 'branches': branches},
                            lambda _: [])
    if require_native:
        native_ledger = json.loads((output/'prepath-dem-sha-ledger.json').read_text())
        native_metadata = json.loads((output/'prepath-dem-frame-metadata.json').read_text())
        source_rows = {row['frame_index']: row for row in native_metadata if row['env'] == 0}
        if set(source_rows) != set(range(4)):
            raise ValueError('shared DEM native source metadata incomplete')
        for index in range(4):
            name = f'prepath-dem-env-0-frame-{index}.npz'
            if native_ledger.get(name) != _sha(output/name):
                raise ValueError('shared DEM native source SHA mismatch')
            with np.load(output/name, allow_pickle=False) as source:
                if source['transformed'].tobytes() != branches[0][index]['transformed'].tobytes():
                    raise ValueError('shared DEM native source byte mismatch')
                raw_digest = hashlib.sha256(source['source_height'].tobytes()).hexdigest()
                if raw_digest != branches[0][index]['raw_sha256']:
                    raise ValueError('shared DEM native raw SHA mismatch')
            for key in ('identity', 'timestamp_ns', 'shape', 'frame',
                        'resolution_m', 'sensor_update_timestamp_s'):
                if source_rows[index][key] != branches[0][index][key]:
                    raise ValueError('shared DEM native source metadata mismatch:' + key)
    return {'pass': True, 'raw_shared_frame_count': 16,
            'ledger_sha256': _sha(output/'prepath-shared-dem-sha-ledger.json')}


def diagnose_branch_inputs(actor_pre_noise, actor_post_noise_pre_delay,
                           actor_actual_post_delay, critic_observation,
                           contact_current, contact_history) -> dict:
    result = {}
    for name, values in (('actor_pre_noise', actor_pre_noise),
                         ('actor_post_noise_pre_delay', actor_post_noise_pre_delay),
                         ('actor_actual_post_delay', actor_actual_post_delay),
                         ('critic_observation', critic_observation),
                         ('contact_current', contact_current),
                         ('contact_history', contact_history)):
        array = np.asarray(values)
        if array.shape[0] != 4 or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(name + ' shape/nonfinite')
        difference = np.abs(array[0] - array[1])
        result[name] = {'shape': list(array.shape),
                        'direct_max_abs': float(np.max(difference)),
                        'direct_rms': float(np.sqrt(np.mean(difference ** 2))),
                        'all_branch_max_abs': [float(np.max(np.abs(array[0]-array[i])))
                                               for i in range(4)]}
    return result
