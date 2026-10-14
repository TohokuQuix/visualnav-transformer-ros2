"""Checkpoint and tensor contract before the isolated CPU model worker."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from visualnav_transformer.eat_nav.bidirectional_bank import bank
from visualnav_transformer.eat_nav.motion_14_stage_a import candidate_role

_BANK_SOURCE=Path(__file__).resolve().parents[1]/'bidirectional_bank.py'
_CANDIDATES=bank()
CANDIDATE_IDS=tuple(candidate_role(p.direction,p.arc_angle_deg) for p in _CANDIDATES)
PATHS=np.asarray([p.waypoints for p in _CANDIDATES],dtype=np.float32)
BANK_SHA=hashlib.sha256(_BANK_SOURCE.read_bytes()).hexdigest()
WIRE_SHA=hashlib.sha256(np.ascontiguousarray(PATHS).tobytes()).hexdigest()


MODES = ('mock14_transport', 'legacy6_demo', 'learned14_adopted')
SOURCE_KEYS = ('model_sha256', 'runner_sha256', 'contract_sha256',
               'dataset_ledger_sha256')


def sha_file(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_adoption_artifacts(results_path, checkpoint_path):
    """Verify the sealed Stage C gate and checkpoint provenance before loading."""
    results_path=Path(results_path).resolve()
    root=results_path.parent
    if results_path.name!='results.json':
        raise ValueError('Stage C result name')
    result_ledger=json.loads((root/'results-ledger.json').read_text())
    dataset_ledger=json.loads((root/'dataset-ledger.json').read_text())
    if set(result_ledger)!={'results.json','normalization.json','test-predictions.npz'} or \
            set(dataset_ledger)!={'grouped.npz','metadata.json','audit.json',
                                  'contract.json','provenance.json'}:
        raise ValueError('Stage C ledger incomplete')
    for name,digest in result_ledger.items():
        if Path(name).name!=name or sha_file(root/name)!=digest:
            raise ValueError('Stage C results ledger mismatch')
    for name,digest in dataset_ledger.items():
        if Path(name).name!=name or sha_file(root/name)!=digest:
            raise ValueError('Stage C dataset ledger mismatch')
    if result_ledger.get('results.json')!=sha_file(results_path) or \
            result_ledger.get('normalization.json')!=sha_file(root/'normalization.json'):
        raise ValueError('Stage C result/normalization unsealed')
    results=json.loads(results_path.read_text())
    source=json.loads((root/'training/source.json').read_text())
    if results.get('demo_adoption',{}).get('decision')!='pass' or \
            results.get('source')!=source:
        raise ValueError('Stage C adoption/source mismatch')
    contract=json.loads((root/'contract.json').read_text())
    if contract.get('candidate_count')!=14 or \
            tuple(contract.get('candidate_order',()))!=CANDIDATE_IDS:
        raise ValueError('Stage C contract candidate order')
    with np.load(root/'grouped.npz',allow_pickle=False) as grouped:
        paths=grouped['paths']
        direction=grouped['direction']
        mask=grouped['mask']
        if paths.ndim!=4 or paths.shape[1:]!=(14,8,2) or len(paths)==0 or \
                not np.array_equal(paths,np.broadcast_to(PATHS,paths.shape)) or \
                direction.shape!=(len(paths),14,1) or \
                not np.array_equal(direction[:,:,0],np.broadcast_to(
                    np.asarray([1]*7+[-1]*7), (len(paths),14))) or \
                mask.shape!=(len(paths),14) or not mask.all():
            raise ValueError('Stage C grouped Path bank differs from frozen bank')
    repo=Path(__file__).resolve().parents[4]
    expected={
        'model_sha256':sha_file(repo/'src/visualnav_transformer/eat_nav/motion_14_predictor.py'),
        'runner_sha256':sha_file(repo/'scripts/eat_nav_motion_14_train.py'),
        'contract_sha256':sha_file(root/'contract.json'),
        'dataset_ledger_sha256':sha_file(root/'dataset-ledger.json'),
    }
    if source!=expected or results['demo_adoption']['contract_sha256'] != \
            sha_file(repo/'docs/eat-nav/motion-predictor-vnext/motion_14_training_contract.json'):
        raise ValueError('Stage C source/contract SHA mismatch')
    selected=results['selected_full_seed']
    expected_checkpoint=(root/'training'/f'full-seed{selected}'/'best.pt').resolve()
    if Path(checkpoint_path).resolve()!=expected_checkpoint or \
            results['variants']['full'][str(selected)]['checkpoint_sha256'] != \
            sha_file(expected_checkpoint):
        raise ValueError('Stage C selected checkpoint mismatch')
    return results,json.loads((root/'normalization.json').read_text())


def validate_mode(mode, candidate_count, checkpoint_path):
    if mode not in MODES:
        raise ValueError('unknown predictor mode')
    if mode == 'mock14_transport':
        if candidate_count != 14 or checkpoint_path is not None:
            raise ValueError('mock14 is checkpoint-free and requires 14 candidates')
    elif mode == 'legacy6_demo':
        if candidate_count != 7 or checkpoint_path is None:
            raise ValueError('legacy6 accepts the seven-forward GUI bank only')
    elif candidate_count != 14 or checkpoint_path is None:
        raise ValueError('adopted model requires 14 candidates and checkpoint')


def _vec(norm, key, n):
    value = np.asarray(norm[key], dtype=np.float32)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f'invalid normalization {key}')
    return value


def validate_normalization(norm):
    if not isinstance(norm, dict) or norm.get('fitted_from') != 'train_only':
        raise ValueError('normalization must be train-only')
    for key in ('dem_height_mean', 'dem_height_std'):
        if not np.isfinite(float(norm[key])):
            raise ValueError('DEM normalization finite')
    if float(norm['dem_height_std']) <= 0 or np.any(_vec(norm,'state_std',22) <= 0) or \
            np.any(_vec(norm,'target_std',2) <= 0):
        raise ValueError('normalization positive scale')
    _vec(norm,'state_mean',22)
    _vec(norm,'target_mean',2)


def validate_checkpoint_metadata(payload, results, *, expected_source):
    if payload.get('variant') != 'full' or payload.get('candidate_count') != 14 or \
            tuple(payload.get('candidate_order',())) != CANDIDATE_IDS or \
            results.get('demo_adoption',{}).get('decision') != 'pass':
        raise ValueError('checkpoint candidate/adoption contract')
    source = payload.get('source')
    if not isinstance(source, dict) or source != expected_source or \
            any(not isinstance(source.get(key),str) or len(source[key]) != 64
                for key in SOURCE_KEYS):
        raise ValueError('checkpoint source mismatch')
    validate_normalization(payload.get('normalization'))


def prepare_model_input(dem, state, paths, direction, norm):
    validate_normalization(norm)
    d = np.asarray(dem, dtype=np.float32)
    s = np.asarray(state, dtype=np.float32)
    p = np.asarray(paths, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    if d.shape != (4,2,20,20) or s.shape != (22,) or p.shape != (14,8,2) or \
            direction.shape not in ((14,),(14,1)) or \
            not np.isfinite(d).all() or not np.isfinite(s).all() or \
            not np.isfinite(p).all() or not np.array_equal(p, PATHS) or \
            not np.array_equal(direction.reshape(-1), [1]*7+[-1]*7) or \
            not np.isin(d[:,1],(0,1)).all() or np.any(d[:,0][d[:,1]==0] != 0):
        raise ValueError('model input contract')
    d = d.copy()
    mask = d[:,1].astype(bool)
    d[:,0][mask] = (d[:,0][mask]-norm['dem_height_mean'])/norm['dem_height_std']
    d[:,0][~mask] = 0
    s = (s-_vec(norm,'state_mean',22))/_vec(norm,'state_std',22)
    return (np.ascontiguousarray(d[None]), np.ascontiguousarray(s[None]),
            np.ascontiguousarray(p[None]),
            np.ascontiguousarray(direction.reshape(1,14,1)),
            np.ones((1,14), dtype=bool))


def inverse_target(scaled, norm):
    validate_normalization(norm)
    value = np.asarray(scaled,dtype=np.float32)
    if value.shape != (14,2) or not np.isfinite(value).all():
        raise ValueError('model output shape/nonfinite')
    # Match the Stage C evaluator: float64 JSON statistics, then float32 label.
    return np.ascontiguousarray((value.astype(np.float64)*
        np.asarray(norm['target_std'],np.float64)+
        np.asarray(norm['target_mean'],np.float64)).astype(np.float32))


def mock14_predict(paths):
    paths = np.asarray(paths,dtype=np.float32)
    if paths.shape != (14,8,2) or not np.array_equal(paths, PATHS):
        raise ValueError('mock requires frozen 14 bank')
    return np.ascontiguousarray(paths[:,-1,:]*.5)


def worker_request(python, checkpoint, results, expected_source, values,
                   normalization, timeout_s=10):
    """One bounded request; no ROS or GPU imports in the transport process."""
    request = {'schema':'eatnav_ros2_predictor_request_v1',
               'mode':'learned14_adopted', 'checkpoint':str(checkpoint),
               'results':str(results), 'expected_source':expected_source,
               'bank_sha256':BANK_SHA, 'wire_sha256':WIRE_SHA,
               'normalization':normalization,
               'arrays':[v.tolist() for v in values]}
    payload = json.dumps(request,separators=(',',':'))
    if len(payload) > 1_000_000:
        raise ValueError('worker request too large')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[3])
    process = subprocess.run([str(python),'-m',
        'visualnav_transformer.eat_nav.ros2_phase2.model_worker'],
        input=payload+'\n',text=True,capture_output=True,timeout=timeout_s,env=env)
    if process.returncode or len(process.stdout) > 100_000:
        raise RuntimeError('model worker failed')
    result = json.loads(process.stdout)
    if result['normalization'] != normalization:
        raise ValueError('worker normalization mismatch')
    return inverse_target(result['scaled_xy'], normalization), result['checkpoint_sha256']


def legacy_worker_request(python, checkpoint, dem, state, robot_pose_map,
                          goal_map, timeout_s=10):
    """Explicit seven-forward legacy adapter; never sends 14 slots to old model."""
    from visualnav_transformer.eat_nav.demo_arc_candidates import DEMO_ARC_ROLES, DEMO_ARC_ANGLES
    from visualnav_transformer.eat_nav.arc_bank import arc_waypoints
    paths=np.asarray([arc_waypoints(DEMO_ARC_ANGLES[role])
                      for role in DEMO_ARC_ROLES],dtype=np.float32)
    validate_mode('legacy6_demo',len(paths),checkpoint)
    request={'schema':'eatnav_ros2_predictor_request_v1','mode':'legacy6_demo',
             'checkpoint':str(checkpoint),'dem':np.asarray(dem,dtype=np.float32).tolist(),
             'state':np.asarray(state,dtype=np.float32).tolist(),
             'paths':paths.tolist(),'robot_pose_map':list(robot_pose_map),
             'goal_map':list(goal_map)}
    payload=json.dumps(request,separators=(',',':'))
    if len(payload)>1_000_000:
        raise ValueError('legacy request too large')
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=''
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[3])
    process=subprocess.run([str(python),'-m',
        'visualnav_transformer.eat_nav.ros2_phase2.model_worker'],
        input=payload+'\n',text=True,capture_output=True,timeout=timeout_s,env=env)
    if process.returncode or len(process.stdout)>100_000:
        raise RuntimeError('legacy worker failed: '+process.stderr[-500:])
    result=json.loads(process.stdout)
    motion=np.asarray(result['motion_xy'],dtype=np.float32)
    scores=np.asarray(result['scores_m'],dtype=np.float32)
    if motion.shape!=(7,2) or scores.shape!=(7,) or \
            not np.isfinite(motion).all() or not np.isfinite(scores).all():
        raise ValueError('legacy worker output contract')
    return motion,scores,result['checkpoint_sha256']
