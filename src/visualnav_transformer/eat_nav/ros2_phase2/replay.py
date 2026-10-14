"""Read-only sealed pre-Path replay for CPU transport parity."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .wire import dem_to_message, actor_to_message


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sealed_prepath(root, *, env_id, episode_id):
    root=Path(root).resolve()
    ledger_path=root/'outer-sha-ledger.json'
    seal=json.loads((root/'outer-seal.json').read_text())
    if _sha(ledger_path) != seal['ledger_sha256']:
        raise ValueError('outer seal ledger mismatch')
    ledger=json.loads(ledger_path.read_text())
    name='sim-output/prepath.npz'
    path=root/name
    if _sha(path) != ledger[name] or _sha(root/'decision.json') != ledger['decision.json']:
        raise ValueError('sealed source hash mismatch')
    decision=json.loads((root/'decision.json').read_text())
    if decision.get('pass') is not True:
        raise ValueError('source run not valid')
    with np.load(path,allow_pickle=False) as arrays:
        if arrays['dem'].shape != (4,4,2,20,20) or \
                arrays['actor_pre_noise'].shape != (4,22) or not 0 <= env_id < 4:
            raise ValueError('source prepath arrays')
        dem=arrays['dem'][env_id].copy()
        actor=arrays['actor_pre_noise'][env_id].copy()
        root_tile_local=arrays['root_tile_local'][env_id].copy()
    stamps=(250,500,750,1000)
    source_id=f'sealed:{ledger[name]}'
    messages=[dem_to_message(frame,stamp*1_000_000,stamp*1_000_000,
                             episode_id,source_id)
              for frame,stamp in zip(dem,stamps)]
    actor_msg=actor_to_message(actor,1_000_000_000,episode_id,
                               {key:1_000_000_000 for key in
                                ('imu','joint','command')})
    return {'dem_original':dem,'actor_original':actor,
            'root_tile_local':root_tile_local,
            'dem_messages':messages,'actor_message':actor_msg,
            'dem_sha256':_sha(path),'ledger_dem_sha256':ledger[name],
            'seal_sha256':_sha(root/'outer-seal.json'),
            'source_stamp_status':'logical_replay; source sensor stamps unavailable'}
