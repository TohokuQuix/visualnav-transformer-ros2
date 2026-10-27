"""Integrity-only boundary: public receipts/errors never contain trajectory semantics."""
import contextlib
import io
from pathlib import Path
import numpy as np
from scripts.eat_nav_motion_14_collection_host import verify_sim_seal
from .fall_stall_audit import read,require,sha,verify_attempt,verify_path_activation
from .fall_stall_collection_contract import contained_path
from .fall_stall_evidence import validate_initial
from .bidirectional_bank import candidate

ERROR_CODES=frozenset(('EVIDENCE_INTEGRITY','REFERENCE_MISMATCH','INITIAL_INPUT_INVALID','TRACE_INVALID','PATH_INVALID','ROW_MISMATCH','SOURCE_MISMATCH','EXECUTION_FAILED','INCOMPLETE_ATTEMPT','BINDING_MISMATCH','ROOT_MISMATCH','LOCKED','RESERVED_TEST_DENIED'))
RECEIPT_SCHEMA='eatnav_additional_blind_receipt_v1'


class BlindVerificationError(ValueError):
    def __init__(self,code,row_index=None):
        self.code=code if code in ERROR_CODES else 'EVIDENCE_INTEGRITY'
        self.row_index=row_index if type(row_index) is int and 0<=row_index<120 else None
        super().__init__(f'{self.code} row={self.row_index}')


def _finite_tree(value):
    if isinstance(value,dict):return all(_finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(_finite_tree(v) for v in value)
    return bool(np.isfinite(value)) if isinstance(value,(int,float)) else True


def _finite_root(value):
    root=np.asarray(value,dtype=float)
    return root.shape==(13,) and np.isfinite(root).all() and np.linalg.norm(root[3:7])>0


def _trace_integrity(sim,complete,activation,info,initial):
    all_terminated=[];all_censored=[]
    lengths=complete['trace_lengths']
    require(complete['control_steps']==max(lengths),'recorded control step mismatch')
    for env in range(4):
        rows=[__import__('json').loads(line) for line in (sim/f'trace-env-{env}.jsonl').read_text().splitlines()]
        require(len(rows)==lengths[env],'trace length mismatch')
        for i,row in enumerate(rows):
            require(row.get('control_step')==i and row.get('logical_sim_time_ns')==activation['activation_stamp_ns']+i*50_000_000,'noncontiguous trace')
            require(row.get('candidate_cycle')==i//40 and row.get('candidate_replan')==bool(i and i%40==0),'cycle schedule mismatch')
            require(_finite_root(row['root_state']) and _finite_root(row['post_root_state']),'invalid/nonfinite pre/post root')
            require(not row.get('terminal') or i==len(rows)-1,'trace continues after terminal')
        verify_path_activation(activation,info['path_body_xy_m'],rows,env,initial['root'][env],initial['root_tile_local'][env])
        terminal=rows[-1].get('terminal') or {}
        if terminal.get('pre_reset_state'):
            snapshot=terminal['pre_reset_state']
            require(terminal.get('termination_reason')=='tilt_over','nonphysical/unknown termination')
            require(_finite_tree(snapshot) and _finite_root(snapshot.get('root_state')),'invalid terminal snapshot')
            require(np.allclose(snapshot['root_state'],rows[-1]['post_root_state'],atol=1e-6,rtol=0),'post/terminal root mismatch')
            if 'termination_reasons' in terminal:require(terminal['termination_reasons']==['tilt_over'],'unexpected termination terms')
            if 'termination_reasons' in snapshot:require(snapshot['termination_reasons']==['tilt_over'],'unexpected snapshot terms')
            all_terminated.append(env)
        elif terminal.get('termination_reason')=='parallel_env_censored_by_other_env':all_censored.append(env)
        else:require(not terminal and len(rows)==200,'partial/unknown terminal')
    require(all_terminated==sorted(complete['terminated_envs']) and all_censored==sorted(complete['censored_envs']),'terminal identity mismatch')
    require(min(lengths)==200 or (bool(all_terminated) and len(all_terminated)+len(all_censored)==4),'partial batch without terminal evidence')


def verify_attempt_blind(attempt,row,contract,reference=None):
    """Inspect necessary protocol fields privately; return only allowlisted identities."""
    code='ROW_MISMATCH'
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            require(row in contract['rows'],'row outside fixed assignment')
            attempt=Path(attempt);require(not attempt.is_symlink(),'symlink attempt')
            frozen={**contract['sources']['pilot_critical_sha256'],'protocol':contract['manifest']['protocol_sha256']}
            assignment={**row,'state_group_id':row.get('base_group',row.get('state_group_id'))}
            pilot={'condition':'front_folded','repeat':0,'source_sha256':frozen}
            code='EVIDENCE_INTEGRITY'
            host,info,complete,_=verify_attempt(attempt,assignment,pilot)
            sim=attempt/'sim-output'
            verify_sim_seal(sim,expected_control_steps=0 if row['stage']=='prepath' else 200,expected_replan_every_steps=40)
            code='INITIAL_INPUT_INVALID'
            initial=validate_initial(sim,row['terrain_seed'],contract['repo'],read,sha,require,contract.setdefault('_initial_cache',{}))
            code='REFERENCE_MISMATCH'
            if row['stage']=='prepath':
                require(reference is None and host.get('reference_prepath_sha256') is None,'unexpected baseline reference')
            else:
                require(reference is not None,'missing prepath reference')
                reference=Path(reference);require(not reference.is_symlink(),'symlink prepath reference')
                digest=sha(reference/'prepath.npz')
                require(host.get('reference_prepath_sha256')==info.get('reference_prepath_sha256')==sha(sim/'prepath.npz')==digest,'paired prepath mismatch')
                code='PATH_INVALID'
                activation=read(sim/'path-activation.json')
                require(activation['direction']==row['direction'] and activation['arc_angle_deg']==row['angle'],'activation candidate mismatch')
                require(activation['activation_stamp_ns']==1_000_000_000,'activation timing mismatch')
                expected=candidate(row['direction'],row['angle']).waypoints
                require(np.asarray(info['path_body_xy_m']).shape==(8,2) and np.allclose(info['path_body_xy_m'],expected,atol=1e-7,rtol=0),'candidate bank geometry mismatch')
                code='TRACE_INVALID';_trace_integrity(sim,complete,activation,info,initial)
            return {'schema':RECEIPT_SCHEMA,'row_index':row['index'],'role':row['role'],'pass':True,
                    'outer_seal_sha256':sha(attempt/'outer-seal.json'),'inner_seal_sha256':sha(sim/'host-seal.json')}
    except Exception:
        raise BlindVerificationError(code,row.get('index')) from None
