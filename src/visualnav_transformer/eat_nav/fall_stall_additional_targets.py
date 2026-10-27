"""Role-aware initial input/v2 join; reserved-test semantics are inaccessible here."""
from pathlib import Path
import json
import numpy as np
from .fall_stall_audit import read,write,seal_derived,sha,require
from .fall_stall_blind_verification import BlindVerificationError,verify_attempt_blind
from .fall_stall_collection_contract import contained_path
from .fall_stall_collection_runner import check_binding
from .fall_stall_dataset_tensors import pack_groups
from .fall_stall_targets import build_targets


def build_development_dataset(root,contract,output,*,roles=('train','validation')):
    if not roles or len(set(roles))!=len(roles) or any(role not in ('train','validation') for role in roles):
        raise BlindVerificationError('RESERVED_TEST_DENIED')
    root=Path(root);output=Path(output)
    if root!=Path(contract['manifest']['output_root']) or root.is_symlink():raise BlindVerificationError('ROOT_MISMATCH')
    require(not output.exists() and not output.resolve().is_relative_to(root.resolve()),'new separate development output required')
    check_binding(contract,read(root/'execution-binding.json'))
    groups=[];records=[];index=[];role_indices={role:[] for role in roles}
    for baseline in contract['rows']:
        if baseline['role'] not in roles or baseline['stage']!='prepath':continue
        selected=[row for row in contract['rows'] if row['base_group']==baseline['base_group']]
        require(len(selected)==5,'incomplete paired group')
        prepath=contained_path(root,baseline['output_relative'])/'sim-output'
        for row in selected:
            attempt=contained_path(root,row['output_relative'])
            receipt=verify_attempt_blind(attempt,row,contract,None if row['stage']=='prepath' else prepath)
            require(read(root/'receipts'/f'{row["index"]:03d}.json')==receipt,'missing/mismatched completion receipt')
        with np.load(prepath/'prepath.npz',allow_pickle=False) as initial:
            dem=initial['dem'].copy();actor=initial['actor_pre_noise'].copy();roots=initial['root'].copy()
        for env in range(4):
            sample_id=f'{baseline["base_group"]}/env{env}'
            group={'initial':{'dem':dem[env],'actor_pre_noise':actor[env],'root':roots[env]},'candidates':{}}
            for row in selected[1:]:
                sim=contained_path(root,row['output_relative'])/'sim-output'
                rows=[json.loads(line) for line in (sim/f'trace-env-{env}.jsonl').read_text().splitlines()]
                path=read(sim/'input-manifest.json')['path_body_xy_m'];activation=read(sim/'path-activation.json')
                target=build_targets(rows,path,decision_stamp_ns=activation['activation_stamp_ns'])
                group['candidates'][row['candidate_id']]={'path':path,'direction':row['direction'],'targets':target}
                records.append({'sample_id':sample_id,'role':row['role'],'seed':row['terrain_seed'],'candidate':row['candidate_id'],
                                'trace_sha256':sha(sim/f'trace-env-{env}.jsonl'),'targets':target})
            role_indices[baseline['role']].append(len(groups));index.append({'sample_id':sample_id,'role':baseline['role'],'seed':baseline['terrain_seed'],'env':env})
            groups.append(group)
    arrays=pack_groups(groups);output.mkdir(parents=True)
    np.savez_compressed(output/'samples.npz',**arrays);write(output/'sample-index.json',index)
    write(output/'split.json',{'schema':'eatnav_additional_fixed_development_split_v1','roles':role_indices,'reserved_test_included':False})
    with (output/'targets.jsonl').open('x') as stream:
        for record in records:stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    write(output/'dataset-manifest.json',{'schema':'eatnav_additional_development_dataset_v1','roles':list(roles),
        'freeze_ledger_sha256':contract['ledger_sha256'],'execution_binding_sha256':sha(root/'execution-binding.json'),
        'state_groups':len(groups),'candidate_outcomes':len(records),'reserved_test_included':False})
    seal_derived(output)
    return {'state_groups':len(groups),'candidate_outcomes':len(records),'reserved_test_included':False}
