"""Batch02 coordinator, isolated from the sealed historical batch01 implementation."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from .fall_stall_audit import read,sha,write,seal_derived,verify_derived,require
from .fall_stall_batch02_contract import contained_path,verify_sources
from .fall_stall_blind_verification import BlindVerificationError,verify_attempt_blind

SCHEMA='eatnav_batch02_execution_binding_v1'
ADDED_SOURCES=tuple('src/visualnav_transformer/eat_nav/'+name+'.py' for name in (
    'fall_stall_collection_contract','fall_stall_blind_verification','fall_stall_collection_runner',
    'fall_stall_collection_runtime','fall_stall_additional_targets'))+('scripts/eat_nav_additional_collection.py',)+tuple('src/visualnav_transformer/eat_nav/fall_stall_batch02_'+name+'.py' for name in ('contract','runner','runtime','targets'))+('scripts/eat_nav_additional_batch02_collection.py',)+(
    'src/visualnav_transformer/deployment/deployment_environment.yaml',
    'src/visualnav_transformer/deployment/src/create_topomap.py',
    'src/visualnav_transformer/deployment/src/explore.py',
    'src/visualnav_transformer/deployment/src/joy_teleop.py',
    'src/visualnav_transformer/deployment/src/navigate.py',
    'src/visualnav_transformer/deployment/src/navigate_robot.py',
    'src/visualnav_transformer/deployment/src/pd_controller.py',
    'src/visualnav_transformer/deployment/src/robot_inference.py',
    'src/visualnav_transformer/deployment/src/robot_interface.py',
    'src/visualnav_transformer/deployment/src/ros_data.py',
    'src/visualnav_transformer/deployment/src/topic_names.py',
    'src/visualnav_transformer/deployment/src/utils.py',
    'src/visualnav_transformer/train/config/defaults.yaml',
    'src/visualnav_transformer/train/config/gnm.yaml',
    'src/visualnav_transformer/train/config/late_fusion.yaml',
    'src/visualnav_transformer/train/config/nomad.yaml',
    'src/visualnav_transformer/train/config/vint.yaml',
    'src/visualnav_transformer/train/data_split.py',
    'src/visualnav_transformer/train/process_bag_diff.py',
    'src/visualnav_transformer/train/process_bags.py',
    'src/visualnav_transformer/train/process_recon.py',
    'src/visualnav_transformer/train/setup.py',
    'src/visualnav_transformer/train/train.py',
    'src/visualnav_transformer/train/train_eat_nav.py',
    'src/visualnav_transformer/train/vint_train/__init__.py',
    'src/visualnav_transformer/train/vint_train/data/__init__.py',
    'src/visualnav_transformer/train/vint_train/data/data_config.yaml',
    'src/visualnav_transformer/train/vint_train/data/data_utils.py',
    'src/visualnav_transformer/train/vint_train/data/vint_dataset.py',
    'src/visualnav_transformer/train/vint_train/models/__init__.py',
    'src/visualnav_transformer/train/vint_train/models/base_model.py',
    'src/visualnav_transformer/train/vint_train/models/gnm/__init__.py',
    'src/visualnav_transformer/train/vint_train/models/gnm/gnm.py',
    'src/visualnav_transformer/train/vint_train/models/gnm/modified_mobilenetv2.py',
    'src/visualnav_transformer/train/vint_train/models/nomad/__init__.py',
    'src/visualnav_transformer/train/vint_train/models/nomad/nomad.py',
    'src/visualnav_transformer/train/vint_train/models/nomad/nomad_vint.py',
    'src/visualnav_transformer/train/vint_train/models/nomad/vib_placeholder.py',
    'src/visualnav_transformer/train/vint_train/models/vint/__init__.py',
    'src/visualnav_transformer/train/vint_train/models/vint/self_attention.py',
    'src/visualnav_transformer/train/vint_train/models/vint/vint.py',
    'src/visualnav_transformer/train/vint_train/models/vint/vit.py',
    'src/visualnav_transformer/train/vint_train/process_data/__init__.py',
    'src/visualnav_transformer/train/vint_train/process_data/process_bags_config.yaml',
    'src/visualnav_transformer/train/vint_train/process_data/process_data_utils.py',
    'src/visualnav_transformer/train/vint_train/training/__init__.py',
    'src/visualnav_transformer/train/vint_train/training/logger.py',
    'src/visualnav_transformer/train/vint_train/training/train_eval_loop.py',
    'src/visualnav_transformer/train/vint_train/training/train_utils.py',
    'src/visualnav_transformer/train/vint_train/visualizing/__init__.py',
    'src/visualnav_transformer/train/vint_train/visualizing/action_utils.py',
    'src/visualnav_transformer/train/vint_train/visualizing/distance_utils.py',
    'src/visualnav_transformer/train/vint_train/visualizing/visualize_utils.py',
)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.status-',delete=False) as stream:
        name=Path(stream.name);json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(name,path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def create_binding(contract,runtime):
    from .fall_stall_batch02_runtime import immutable_runtime
    return {'schema':SCHEMA,'created_unix_ns':time.time_ns(),'batch_id':contract['manifest']['batch_id'],
            'output_root':contract['manifest']['output_root'],'runtime_identity_sha256':digest(immutable_runtime(runtime)),'freeze_ledger_sha256':contract['ledger_sha256'],
            'protocol_sha256':contract['manifest']['protocol_sha256'],'assignment_sha256':digest(contract['rows']),
            'source_sha256':{p:sha(contract['repo']/p) for p in ADDED_SOURCES},'runtime':runtime,
            'test_access':'integrity_only_unopened','failure_induction':False}


def check_binding(contract,binding):
    try:
        from .fall_stall_batch02_runtime import check_runtime_proof,immutable_runtime
        require(binding['batch_id']==contract['manifest']['batch_id'] and binding['output_root']==contract['manifest']['output_root'],'binding batch/root')
        require(binding['runtime_identity_sha256']==digest(immutable_runtime(binding['runtime'])),'binding runtime identity')
        require(binding['schema']==SCHEMA and binding['freeze_ledger_sha256']==contract['ledger_sha256'],'binding identity')
        require(binding['assignment_sha256']==digest(contract['rows']) and binding['protocol_sha256']==contract['manifest']['protocol_sha256'],'binding schedule')
        require(binding['test_access']=='integrity_only_unopened' and binding['failure_induction'] is False,'binding semantics')
        require(set(binding['source_sha256'])==set(ADDED_SOURCES),'binding source membership')
        for p,d in binding['source_sha256'].items():require(sha(contract['repo']/p)==d,'execution source changed')
        verify_sources(contract)
        check_runtime_proof(contract,binding['runtime'],binding['source_sha256'])
    except Exception:raise BlindVerificationError('BINDING_MISMATCH') from None


def save_preparation(contract,output,binding):
    """Seal a separate binding/snapshot bundle without initializing collection."""
    output=Path(output)
    check_binding(contract,binding)
    require(not output.exists() and not output.resolve().is_relative_to(Path(contract['manifest']['output_root']).resolve()),
            'new separate preparation output required')
    require(not Path(contract['manifest']['output_root']).exists(),'raw collection already exists')
    output.mkdir(parents=True)
    shutil.copytree(contract['freeze'],output/'freeze')
    for relative in ADDED_SOURCES:
        target=output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(contract['repo']/relative,target)
    write(output/'execution-binding.json',binding)
    write(output/'preparation.json',{'schema':'eatnav_batch02_runtime_preparation_v1',
        'state':'runtime_bound_collection_not_started','collection_started':False,'ready_to_collect':True,
        'startup_runtime_and_resources_recheck_required':True,'reserved_test':'unopened',
        'freeze_ledger_sha256':contract['ledger_sha256'],
        'execution_binding_sha256':sha(output/'execution-binding.json')})
    seal_derived(output)
    load_preparation(contract,output,expected_binding_sha256=sha(output/'execution-binding.json'))


def load_preparation(contract,output,*,expected_binding_sha256=None):
    """No simulator access; validate saved sources, freeze, and binding identity."""
    try:
        output=Path(output);verify_derived(output)
        actual=sha(output/'execution-binding.json')
        if expected_binding_sha256 is not None:require(actual==expected_binding_sha256,'expected binding identity')
        preparation=read(output/'preparation.json')
        require(preparation['schema']=='eatnav_batch02_runtime_preparation_v1'
                and preparation['execution_binding_sha256']==actual
                and preparation['freeze_ledger_sha256']==contract['ledger_sha256']
                and preparation['collection_started'] is False and preparation['reserved_test']=='unopened',
                'preparation identity')
        binding=read(output/'execution-binding.json');check_binding(contract,binding)
        verify_derived(output/'freeze')
        require(sha(output/'freeze/sha-ledger.json')==contract['ledger_sha256'],'prepared freeze identity')
        for relative,digest_value in binding['source_sha256'].items():
            require(sha(contained_path(output/'sources',relative))==digest_value,'prepared source identity')
        return binding
    except Exception:
        raise BlindVerificationError('BINDING_MISMATCH') from None


def check_run_snapshot(contract,root,binding):
    try:
        for name in ('receipts','starts','private','sources','freeze','execution-binding.json','assignment.json','status.json','.collection.lock'):
            contained_path(root,name)
        require(read(root/'execution-binding.json')==binding,'saved binding mismatch')
        require(read(root/'assignment.json')['rows']==contract['rows'],'saved assignment mismatch')
        verify_derived(root/'freeze')
        require(sha(root/'freeze/sha-ledger.json')==contract['ledger_sha256'],'freeze snapshot identity')
        for p,d in binding['source_sha256'].items():require(sha(contained_path(root/'sources',p))==d,'execution snapshot identity')
        flags=[(root/'receipts'/f'{row["index"]:03d}.json').is_file() for row in contract['rows']]
        require(flags==sorted(flags,reverse=True),'noncontiguous completion receipts')
        prefix=sum(flags)
        for row in contract['rows'][prefix+1:]:
            require(not contained_path(root,row['output_relative']).exists(),'future attempt beyond completion prefix')
            require(not contained_path(root,f'starts/{row["index"]:03d}.json').exists(),'future start beyond completion prefix')
            require(not contained_path(root,f'private/{row["index"]:03d}.log').exists(),'future log beyond completion prefix')
        names={f'{row["index"]:03d}.json' for row in contract['rows']}
        for directory in ('receipts','starts'):
            require({p.name for p in (root/directory).iterdir()}<=names,'unexpected row metadata')
            for p in (root/directory).iterdir():contained_path(root,str(p.relative_to(root)))
    except Exception:raise BlindVerificationError('BINDING_MISMATCH') from None


def initialize_collection(contract,root,binding):
    root=Path(root)
    if root!=Path(contract['manifest']['output_root']) or root.is_symlink():raise BlindVerificationError('ROOT_MISMATCH')
    check_binding(contract,binding);root.mkdir(parents=True,exist_ok=False)
    for name in ('receipts','starts','private','sources'): (root/name).mkdir(mode=0o700 if name=='private' else 0o755)
    shutil.copytree(contract['freeze'],root/'freeze')
    write(root/'execution-binding.json',binding);write(root/'assignment.json',{'rows':contract['rows']})
    for p in ADDED_SOURCES:
        target=root/'sources'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(contract['repo']/p,target)
    atomic_json(root/'status.json',{'state':'initialized','completed_processes':0,'total_processes':len(contract['rows']),
                                   'reserved_test':'unopened','failure_induction':False})


def verify_collection(contract,root,*,require_complete=True):
    root=Path(root)
    try:
        require(root==Path(contract['manifest']['output_root']) and not root.is_symlink(),'root identity')
        saved_binding=read(root/'execution-binding.json');check_binding(contract,saved_binding)
        check_run_snapshot(contract,root,saved_binding)
        require(read(root/'assignment.json')['rows']==contract['rows'],'saved schedule identity')
        count=0;roles={k:0 for k in ('train','validation','reserved_test')}
        for row in contract['rows']:
            receipt=root/'receipts'/f'{row["index"]:03d}.json'
            if not receipt.exists():
                require(not require_complete,'missing receipt');break
            attempt=contained_path(root,row['output_relative'])
            reference=contained_path(root,row['reference_relative']) if row['reference_relative'] else None
            actual=verify_attempt_blind(attempt,row,contract,reference)
            require(read(receipt)==actual,'receipt/evidence mismatch');count+=1;roles[row['role']]+=1
        if require_complete:
            require(read(root/'status.json')['state']=='completed' and count==len(contract['rows']),'collection incomplete')
            verify_derived(root)
        return {'state':'completed' if count==len(contract['rows']) else 'incomplete','completed_processes':count,
                'total_processes':len(contract['rows']),'role_process_counts':roles,'reserved_test':'unopened','integrity':'pass'}
    except BlindVerificationError:raise
    except Exception:raise BlindVerificationError('EVIDENCE_INTEGRITY') from None


def run_collection(contract,root,binding,*,executor=None,resume=False):
    root=Path(root)
    if root!=Path(contract['manifest']['output_root']) or root.is_symlink():raise BlindVerificationError('ROOT_MISMATCH')
    if executor is None:
        from .fall_stall_batch02_runtime import recheck_runtime
        recheck_runtime(contract,binding)
    if not resume:initialize_collection(contract,root,binding)
    elif not root.is_dir():raise BlindVerificationError('ROOT_MISMATCH')
    check_binding(contract,binding)
    check_run_snapshot(contract,root,binding)
    if executor is None:
        from .fall_stall_batch02_runtime import execute_frozen_host
        executor=lambda target,**kwargs:execute_frozen_host(target,contract,binding,**kwargs)
    with (root/'.collection.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise BlindVerificationError('LOCKED') from None
        if read(root/'status.json')['state']=='completed':return verify_collection(contract,root)
        count=0;roles={k:0 for k in ('train','validation','reserved_test')}
        for row in contract['rows']:
            index=row['index'];attempt=contained_path(root,row['output_relative'])
            reference=contained_path(root,row['reference_relative']) if row['reference_relative'] else None
            receipt=contained_path(root,f'receipts/{index:03d}.json');start=contained_path(root,f'starts/{index:03d}.json')
            try:
                check_binding(contract,binding)
                if receipt.exists():
                    actual=verify_attempt_blind(attempt,row,contract,reference)
                    require(read(receipt)==actual,'receipt mismatch')
                else:
                    if not attempt.exists():
                        if start.exists():raise BlindVerificationError('INCOMPLETE_ATTEMPT',index)
                        write(start,{'row_index':index,'role':row['role'],'started_unix_ns':time.time_ns()})
                        atomic_json(root/'status.json',{'state':'running','completed_processes':count,'total_processes':len(contract['rows']),
                            'row_index':index,'role':row['role'],'reserved_test':'unopened','role_process_counts':roles})
                        print(json.dumps({'row_index':index,'role':row['role'],'state':'started','completed_processes':count}),flush=True)
                        log_path=contained_path(root,f'private/{index:03d}.log')
                        with log_path.open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                            try:
                                executor(attempt,direction=row['direction'],angle=row['angle'],seed=row['terrain_seed'],
                                    terrain_type=row['terrain_type'],terrain_level=row['terrain_level'],condition='front_folded',repeat=0,
                                    protocol=contract['freeze']/'protocol.md',reference=reference,prepath_only=row['stage']=='prepath',
                                    rl_checkpoint=Path(contract['sources']['actor']['path']),actor_env_yaml=Path(contract['sources']['actor_env']['path']),
                                    control_steps=200,replan_every_steps=40)
                            except Exception:
                                traceback.print_exc(file=log)
                                raise BlindVerificationError('EXECUTION_FAILED',index) from None
                    try:actual=verify_attempt_blind(attempt,row,contract,reference)
                    except BlindVerificationError:
                        if not (attempt/'outer-seal.json').is_file():raise BlindVerificationError('INCOMPLETE_ATTEMPT',index) from None
                        raise
                    write(receipt,actual)
                count+=1;roles[row['role']]+=1
                atomic_json(root/'status.json',{'state':'running','completed_processes':count,'total_processes':len(contract['rows']),
                    'role_process_counts':roles,'reserved_test':'unopened'})
                print(json.dumps({'row_index':index,'role':row['role'],'pass':True,'completed_processes':count,'total_processes':len(contract['rows'])}),flush=True)
            except Exception as error:
                code=error.code if isinstance(error,BlindVerificationError) else 'EVIDENCE_INTEGRITY'
                atomic_json(root/'status.json',{'state':'stopped','completed_processes':count,'total_processes':len(contract['rows']),
                    'row_index':index,'role':row['role'],'error_code':code,'reserved_test':'unopened','role_process_counts':roles})
                raise BlindVerificationError(code,index) from None
        atomic_json(root/'status.json',{'state':'completed','completed_processes':count,'total_processes':len(contract['rows']),
            'role_process_counts':roles,'reserved_test':'unopened','failure_induction':False})
        seal_derived(root)
        return verify_collection(contract,root)
