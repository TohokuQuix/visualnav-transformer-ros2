"""Freeze one limited pilot process from passing candidate-free measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from visualnav_transformer.eat_nav.p1_statistical_freeze import sha


REPO=Path(__file__).resolve().parents[1]
BASE=Path('/tmp/eatnav-p1-statistical-20261009')


def atomic(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def prepare(anchor_id,batch_index,root):
    reference=BASE/f'preflight-{anchor_id}-b{batch_index}-v4'
    seal=json.loads((reference/'host-seal.json').read_text())
    complete=reference/'output/complete.json'
    if sha(complete)!=seal['complete_sha256']:
        raise ValueError('candidate-free seal mismatch')
    gate=json.loads((reference/'output/prepath-gate.json').read_text())
    if not gate['pass']:
        raise ValueError('candidate-free initialization failed')
    other=BASE/f'preflight-{anchor_id}-b{1-batch_index}-v4'
    terrain=json.loads((reference/'output/terrain.json').read_text())
    other_terrain=json.loads((other/'output/terrain.json').read_text())
    if (terrain['fingerprints'][0]['canonical_mesh_sha256']!=
            other_terrain['fingerprints'][0]['canonical_mesh_sha256'] or
            terrain['collision']['sha256']!=other_terrain['collision']['sha256']):
        raise ValueError('cross-batch terrain identity failed')
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copytree(reference/'input',root/'input')
    shutil.copytree(REPO/'src/visualnav_transformer',root/'src/visualnav_transformer')
    shutil.copy2(REPO/'scripts/eat_nav_p1_statistical_pilot_sim.py',root/'source.py')
    config=json.loads((REPO/'config/eat_nav_p1_statistical_v2.json').read_text())
    if config['primary_control_steps']!=40 or config['diagnostic_control_steps']!=120:
        raise ValueError('P1 primary/diagnostic window contract')
    config.update({
        'rl_checkpoint':'/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
        'rl_checkpoint_sha256':'b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d',
        'follower_sha256':'616097e44d199b8d33f6534aedd299f2d20127ed8e100c18777f9ca7a2da389d',
        'follower_params_sha256':'ddedff40d13e20416a43309669ad19e7f3a06e2df8cc8c9d3248369e14d40c90',
        'bridge_sha256':sha(REPO/'scripts/eat_nav_command_delivery_bridge.py'),
        'logical_t0_ns':2_000_000_000,
    })
    atomic(root/'input/statistical-config.json',config)
    freeze=json.loads((root/'input/freeze.json').read_text())
    freeze.update({'schema':'p1_statistical_pilot_freeze_v1',
                   'config_path':str((REPO/'config/eat_nav_p1_statistical_v2.json').resolve()),
                   'config_sha256':sha(root/'input/statistical-config.json'),
                   'pilot_simulator_source_sha256':sha(root/'source.py'),
                   'analysis_source_sha256':sha(REPO/'src/visualnav_transformer/eat_nav/p1_statistical_analysis.py'),
                   'candidate_free_seal_sha256':sha(reference/'host-seal.json'),
                   'candidate_free_terrain_sha256':sha(reference/'output/terrain.json'),
                   'router_log_directory':'/home/kojima/eatnav-router-logs/p1-r5-20261008-attempt2'})
    atomic(root/'input/freeze.json',freeze)
    atomic(root/'input/pilot-preflight.json',{
        'candidate_free_root':str(reference),'candidate_free_seal_sha256':sha(reference/'host-seal.json'),
        'same_terrain_across_batches':True,'terrain_sha256':sha(reference/'output/terrain.json'),
        'prepath_gate_sha256':sha(reference/'output/prepath-gate.json')})
    return {'root':str(root),'source_sha256':sha(root/'source.py'),
            'freeze_sha256':sha(root/'input/freeze.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--anchor-id',required=True)
    parser.add_argument('--batch-index',type=int,choices=(0,1),required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.anchor_id,args.batch_index,args.root)))
