"""Manifest-ordered passive collection, append-only receipts and blind resume."""
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
from .fall_stall_collection_contract import contained_path,verify_sources
from .fall_stall_blind_verification import BlindVerificationError,verify_attempt_blind

SCHEMA='eatnav_additional_execution_binding_v1'
ADDED_SOURCES=tuple('src/visualnav_transformer/eat_nav/'+name+'.py' for name in (
    'fall_stall_collection_contract','fall_stall_blind_verification','fall_stall_collection_runner',
    'fall_stall_collection_runtime','fall_stall_additional_targets'))+('scripts/eat_nav_additional_collection.py',)


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
    return {'schema':SCHEMA,'created_unix_ns':time.time_ns(),'freeze_ledger_sha256':contract['ledger_sha256'],
            'protocol_sha256':contract['manifest']['protocol_sha256'],'assignment_sha256':digest(contract['rows']),
            'source_sha256':{p:sha(contract['repo']/p) for p in ADDED_SOURCES},'runtime':runtime,
            'test_access':'integrity_only_unopened','failure_induction':False}


def check_binding(contract,binding):
    try:
        require(binding['schema']==SCHEMA and binding['freeze_ledger_sha256']==contract['ledger_sha256'],'binding identity')
        require(binding['assignment_sha256']==digest(contract['rows']) and binding['protocol_sha256']==contract['manifest']['protocol_sha256'],'binding schedule')
        require(binding['test_access']=='integrity_only_unopened' and binding['failure_induction'] is False,'binding semantics')
        require(set(binding['source_sha256'])==set(ADDED_SOURCES),'binding source membership')
        for p,d in binding['source_sha256'].items():require(sha(contract['repo']/p)==d,'execution source changed')
        verify_sources(contract)
    except Exception:raise BlindVerificationError('BINDING_MISMATCH') from None


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
    if not resume:initialize_collection(contract,root,binding)
    elif not root.is_dir():raise BlindVerificationError('ROOT_MISMATCH')
    check_binding(contract,binding)
    check_run_snapshot(contract,root,binding)
    if executor is None:
        from .fall_stall_collection_runtime import execute_frozen_host
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
