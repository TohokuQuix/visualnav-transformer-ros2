"""One-shot bounded v2 preflight, 56-trial pilot, and read-only analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from visualnav_transformer.eat_nav.p1_statistical_freeze import sha
from visualnav_transformer.eat_nav.p1_statistical_protocol import validate_assignment
from eat_nav_p1_statistical_analyze import verify_candidate_free, GROUPS
from eat_nav_p1_statistical_host import run as run_candidate_free
from eat_nav_p1_statistical_pilot_host import run as run_pilot, router_audit
from eat_nav_p1_statistical_pilot_prepare import prepare as prepare_pilot


REPO=Path(__file__).resolve().parents[1]
OLD=Path('/tmp/eatnav-p1-statistical-20261009')
ROOT=Path('/tmp/eatnav-p1-statistical-v2-20261009')
ORDER=tuple((group,batch) for group in GROUPS for batch in (0,1))
CHECKPOINT='/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'
CHECKPOINT_SHA='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d'


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def prepare_candidate_free(group,batch,config):
    previous=OLD/f'preflight-{group}-b{batch}-v4'
    old_seal=json.loads((previous/'host-seal.json').read_text())
    if sha(previous/'output/complete.json')!=old_seal['complete_sha256']:
        raise ValueError('historical preflight seal drift')
    root=ROOT/f'preflight-{group}-b{batch}'
    if root.exists():raise FileExistsError(root)
    root.mkdir()
    shutil.copytree(previous/'input',root/'input')
    shutil.copytree(REPO/'src/visualnav_transformer',root/'src/visualnav_transformer')
    shutil.copy2(REPO/'scripts/eat_nav_p1_statistical_sim.py',root/'source.py')
    runtime={**config,'rl_checkpoint':CHECKPOINT,'rl_checkpoint_sha256':CHECKPOINT_SHA}
    atomic(root/'input/statistical-config.json',runtime)
    freeze=json.loads((root/'input/freeze.json').read_text())
    freeze.update({'schema':'p1_statistical_candidate_free_v2',
        'config_path':str((REPO/'config/eat_nav_p1_statistical_v2.json').resolve()),
        'config_sha256':sha(root/'input/statistical-config.json'),
        'simulator_source_sha256':sha(root/'source.py'),
        'historical_preflight_seal_sha256':sha(previous/'host-seal.json')})
    atomic(root/'input/freeze.json',freeze)
    return root


def gate_candidate_free(group):
    measurements=[]
    for batch in (0,1):
        root=ROOT/f'preflight-{group}-b{batch}'
        result=verify_candidate_free(root)
        terrain=json.loads((root/'output/terrain.json').read_text())
        measurements.append((terrain['fingerprints'][0]['canonical_mesh_sha256'],
                             terrain['collision']['sha256']))
        if result['trace_count']!=7 or not result['analysis_primary_window_accepted']:
            raise ValueError('candidate-free 120-step analysis acceptance')
    if measurements[0]!=measurements[1]:
        raise ValueError('cross-batch canonical terrain/collision mismatch: '+group)
    return {'group':group,'mesh_sha256':measurements[0][0],
            'collision_sha256':measurements[0][1]}


def main():
    if ROOT.exists():raise FileExistsError('v2 root is one-shot: '+str(ROOT))
    config=json.loads((REPO/'config/eat_nav_p1_statistical_v2.json').read_text())
    assignment=json.loads((OLD/'preflight-first-differential-b0-v4/input/assignment.json').read_text())
    validate_assignment(assignment)
    if (config['primary_control_steps'],config['diagnostic_control_steps'])!=(40,120):
        raise ValueError('P1 window mismatch before physics')
    if config['batch_seeds']!=assignment['batch_seeds'] or len(ORDER)!=8 or len(assignment['trials'])!=56:
        raise ValueError('frozen assignment/count mismatch')
    for group,batch in ORDER:
        rows=[r for r in assignment['trials'] if r['anchor_id']==group and r['batch_index']==batch]
        if len(rows)!=7 or {r['slot'] for r in rows}!=set(range(7)):
            raise ValueError('seven-slot assignment mismatch')
    router=router_audit()
    noise_sources={}
    for name,path in {
        'onix_rough_env_cfg':'/workspace/isaaclab_onix/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py',
        'eat_collection_cfg':'/tmp/eatnav-extended/isaac-source/onix_isaac_lab/tasks/locomotion/velocity/config/onix/eat_nav_env_cfg.py',
        'isaaclab_noise_model':'/workspace/isaaclab/source/isaaclab/isaaclab/utils/noise/noise_model.py',
    }.items():
        digest=subprocess.run(['docker','exec','isaac-lab-onix','sha256sum',path],
                              capture_output=True,text=True,check=True).stdout.split()[0]
        noise_sources[name]={'path':path,'sha256':digest}
    ROOT.mkdir()
    static={'schema':'p1_statistical_v2_static_gate_v1','pass':True,
            'primary_steps':40,'total_candidate_steps':120,'diagnostic_last_step':119,
            'state_groups':4,'batches_per_group':2,'slots_per_batch':7,'trial_count':56,
            'assignment_sha256':sha(OLD/'preflight-first-differential-b0-v4/input/assignment.json'),
            'config_sha256':sha(REPO/'config/eat_nav_p1_statistical_v2.json'),
            'candidate_free_source_sha256':sha(REPO/'scripts/eat_nav_p1_statistical_sim.py'),
            'pilot_source_sha256':sha(REPO/'scripts/eat_nav_p1_statistical_pilot_sim.py'),
            'analysis_source_sha256':sha(REPO/'scripts/eat_nav_p1_statistical_analyze.py'),
            'policy_action_mode':'act_inference','paired_policy_noise':False,
            'noise_verification':'Actor policy corruption enabled; additive uniform on base angular velocity ±0.2, joint position ±0.01, joint velocity ±1.5, projected gravity ±0.05; torch.rand_like global RNG; terrain generation fork_rng restores CPU/CUDA RNG; no Path-dependent draw observed',
            'noise_remaining_uncertainty':'global observation RNG is not paired per env or Path',
            'noise_source_files':noise_sources,
            'router':router}
    atomic(ROOT/'static-preflight.json',static)
    started=time.time();completed=[]
    def status(stage,**details):
        atomic(ROOT/'status.json',{'stage':stage,'started_wall':started,
             'completed':completed,**details})
    try:
        status('candidate_free')
        for group,batch in ORDER:
            root=prepare_candidate_free(group,batch,config)
            status('candidate_free',current=str(root))
            run_candidate_free(root,group,batch)
            accepted=verify_candidate_free(root)
            completed.append({'stage':'candidate_free','group':group,'batch':batch,
                'seal_sha256':sha(root/'host-seal.json'),
                'trace_count':accepted['trace_count']})
            status('candidate_free')
        geometry=[gate_candidate_free(group) for group in GROUPS]
        atomic(ROOT/'candidate-free-gate.json',{'pass':True,'process_count':8,
            'trace_count':56,'control_steps_per_trace':120,'Path_publications':0,
            'terrain':geometry})
        # Freeze every pilot input before exposing the first Path.
        for group,batch in ORDER:
            pilot_root=ROOT/f'pilot-{group}-b{batch}-v1'
            prepare_pilot(group,batch,pilot_root,reference_base=ROOT,reference_suffix='')
        atomic(ROOT/'pilot-freeze.json',{'schema':'p1_statistical_v2_pilot_freeze_v1',
            'roots':[{'group':group,'batch':batch,
                      'freeze_sha256':sha(ROOT/f'pilot-{group}-b{batch}-v1/input/freeze.json')}
                     for group,batch in ORDER],
            'assignment_sha256':static['assignment_sha256'],
            'config_sha256':static['config_sha256']})
        status('pilot')
        for group,batch in ORDER:
            root=ROOT/f'pilot-{group}-b{batch}-v1'
            status('pilot',current=str(root))
            run_pilot(root,group,batch)
            completed.append({'stage':'pilot','group':group,'batch':batch,
                'seal_sha256':sha(root/'host-seal.json')})
            status('pilot')
        status('analysis')
        result=subprocess.run([sys.executable,str(REPO/'scripts/eat_nav_p1_statistical_analyze.py'),
            '--pilot-root',str(ROOT),'--output',str(ROOT/'analysis')],
            env={**os.environ,'PYTHONPATH':str(REPO/'src')},
            capture_output=True,text=True)
        (ROOT/'analysis-command.log').write_text(result.stdout+result.stderr)
        if result.returncode or not (ROOT/'analysis/seal.json').is_file():
            raise RuntimeError('read-only pilot analysis failed: '+result.stderr[-1000:])
        status('complete',finished_wall=time.time(),analysis_seal_sha256=sha(ROOT/'analysis/seal.json'))
        return 0
    except BaseException as exc:
        atomic(ROOT/'failure.json',{'type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'completed':completed})
        status('stopped',failed_stage=json.loads((ROOT/'status.json').read_text())['stage'],
               failure_sha256=sha(ROOT/'failure.json'),finished_wall=time.time())
        return 1


if __name__=='__main__':
    raise SystemExit(main())
