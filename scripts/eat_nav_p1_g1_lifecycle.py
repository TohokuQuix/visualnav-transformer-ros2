"""Pure ownership and finalization gates for candidate-free P1 G1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eat_nav_p1_owner_host import resolve_owner


GAP_CAP_BYTES = 256 * 1024 * 1024
WRITERS = frozenset(('simulator', 'heartbeat', 'resource', 'log'))


def diagnose_owner_resolution(ready: dict, table: dict, container_id: str) -> str:
    """Classify a failed lookup from actual /proc rows; never infer an owner."""
    rows=[row for row in table.values()
          if row.get('container_pid') == ready.get('container_pid')]
    if not rows:return 'container_pid_not_found'
    rows=[row for row in rows if container_id in row.get('cgroup','')]
    if not rows:return 'cgroup_mismatch'
    rows=[row for row in rows if row.get('start_ticks') == ready.get('start_ticks')]
    if not rows:return 'start_tick_mismatch'
    rows=[row for row in rows if (row.get('namespace_pid_chain') or [None])[-1]
          == (ready.get('nspid') or [None])[-1]]
    if not rows:return 'pid_namespace_mismatch'
    token=ready.get('run_token')
    rows=[row for row in rows if token in row.get('command_line','').split()]
    if not rows:return 'token_or_command_mismatch'
    if len(rows)==1:return 'matched'
    return 'ambiguous_or_other_identity_mismatch'


def residual_identities(table: dict, *, owner: dict | None, known: dict[int,int],
                        token: str, container_id: str) -> list[dict]:
    residual=[]
    identities=dict(known)
    if owner is not None:identities[owner['host_pid']]=owner['start_ticks']
    for pid,tick in sorted(identities.items()):
        row=table.get(pid)
        if row is not None and row.get('start_ticks')==tick:
            residual.append({'host_pid':pid,'start_ticks':tick,'reason':'known_owned_process'})
    for row in table.values():
        if (container_id in row.get('cgroup','') and token in row.get('command_line','').split()
                and not any(item['host_pid']==row['host_pid'] for item in residual)):
            residual.append({'host_pid':row['host_pid'],'start_ticks':row['start_ticks'],
                             'reason':'unresolved_run_token_claimant'})
    return residual


def resolve_ready_owner(ready: dict, table: dict, *, token: str, container_id: str,
                        source_sha: str, config_sha: str, output_root: str) -> dict:
    expected = {'schema':'eatnav_p1_owner_v1','run_token':token,
                'container_id':container_id,'source_sha256':source_sha,
                'config_sha256':config_sha,'output_root':output_root}
    if any(ready.get(k) != v for k,v in expected.items()):
        raise ValueError('owner_ready_provenance_mismatch')
    return resolve_owner(table,ready,container_id)


class G1Lifecycle:
    def __init__(self, *, token: str, container_id: str, source_sha: str,
                 config_sha: str, output_root: str, gpu_uuid: str):
        self.token=token;self.container_id=container_id;self.source_sha=source_sha
        self.config_sha=config_sha;self.output_root=output_root;self.gpu_uuid=gpu_uuid
        self.owner=None;self.ready=None;self.phase='WAITING_OWNER';self.live_gpu_samples=0

    def register_ready(self, ready: dict, table: dict) -> dict:
        if self.phase != 'WAITING_OWNER':
            raise ValueError('owner_registration_order')
        self.owner=resolve_ready_owner(ready,table,token=self.token,
            container_id=self.container_id,source_sha=self.source_sha,
            config_sha=self.config_sha,output_root=self.output_root)
        self.ready=ready
        self.phase='OWNER_REGISTERED'
        return self.owner

    def _check_bracket(self, sample: dict) -> None:
        if sample.get('gpu_uuid') != self.gpu_uuid or sample.get('failures') or not sample.get('pass'):
            raise ValueError('GPU_bracket_invalid')
        gap=sample.get('bracket_gap_bytes')
        if not isinstance(gap,int) or gap > GAP_CAP_BYTES:
            raise ValueError('attribution_gap_over_256_mib')
        for key in ('process_table_before','process_table_after'):
            table=sample.get(key)
            row=table.get(self.owner['host_pid']) if isinstance(table,dict) else None
            if row is None or row.get('start_ticks') != self.owner['start_ticks']:
                raise ValueError('owner_disappeared_or_reused')

    def pre_authorization(self, sample: dict) -> None:
        if self.phase != 'OWNER_REGISTERED':
            raise ValueError('preauthorization_order')
        self._check_bracket(sample)
        if sample.get('owned_gpu_pids'):
            raise ValueError('GPU_before_authorization')
        self.phase='MONITOR_READY'

    def authorize(self) -> dict:
        if self.phase != 'MONITOR_READY':
            raise ValueError('owner_not_registered_or_monitor_not_ready')
        self.phase='AUTHORIZED'
        return {'schema':'eatnav_p1_owner_v1','run_token':self.token,
                'container_pid':self.ready['container_pid'],
                'start_ticks':self.owner['start_ticks'],
                'output_root':self.output_root,'source_sha256':self.source_sha,
                'config_sha256':self.config_sha,'host_pid':self.owner['host_pid'],
                'monitor_ready':True}

    def observe(self, sample: dict) -> None:
        if self.phase not in ('AUTHORIZED','RUNNING'):
            raise ValueError('live_sample_order')
        self._check_bracket(sample)
        known_tree=set(sample.get('owned_process_tree_host_pids') or [self.owner['host_pid']])
        for row in sample.get('processes',[]):
            if row.get('classification')=='owned_simulator_tree':
                pid=row.get('host_pid')
                proc=sample['process_table_before'].get(pid)
                if pid not in known_tree or proc is None or proc.get('start_ticks') != row.get('start_ticks'):
                    raise ValueError('NVML_owner_mismatch')
        owned=set(sample.get('owned_gpu_pids') or [])
        if owned and not owned.issubset(known_tree):
            raise ValueError('NVML_owner_mismatch')
        if owned:self.live_gpu_samples += 1
        self.phase='RUNNING'

    def complete(self, *, native_exit: int, owner_alive: bool, writers_closed: bool) -> bool:
        if self.phase not in ('AUTHORIZED','RUNNING'):
            raise ValueError('completion_order')
        if owner_alive:raise ValueError('owned_residual')
        if not writers_closed:raise ValueError('writers_not_closed')
        if native_exit != 0:raise ValueError('native_exit_nonzero')
        if self.live_gpu_samples < 1:raise ValueError('owned_GPU_sample_missing')
        self.phase='COMPLETE'
        return True


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_new(path: Path, value: dict) -> None:
    if path.exists():raise FileExistsError(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def seal_after_writers(root: Path, *, writer_state: dict, stage: str,
                       reason: str | None = None) -> dict:
    """Seal only after all writer processes/fds have closed and ownership ended."""
    root=Path(root)
    if set(writer_state) != WRITERS or not all(writer_state.values()):
        raise ValueError('writers_not_closed')
    ledger_path=root/'sha-ledger.json';seal_path=root/'host-seal.json'
    if ledger_path.exists() or seal_path.exists():raise FileExistsError('seal_already_exists')
    files={str(p.relative_to(root)):_sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    _atomic_new(ledger_path,{'schema':'eatnav_g1_final_ledger_v2','file_sha256':files})
    _atomic_new(seal_path,{'schema':'eatnav_g1_final_seal_v2','stage':stage,'reason':reason,
                          'sha_ledger_sha256':_sha(ledger_path),'writer_state':writer_state})
    return verify_seal(root)


def verify_seal(root: Path) -> dict:
    root=Path(root)
    ledger_path=root/'sha-ledger.json';seal_path=root/'host-seal.json'
    seal=json.loads(seal_path.read_text());ledger=json.loads(ledger_path.read_text())
    if _sha(ledger_path)!=seal['sha_ledger_sha256']:
        raise ValueError('post_seal_mutation:ledger')
    expected=ledger['file_sha256']
    current={str(p.relative_to(root)):_sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p not in (ledger_path,seal_path)}
    if current != expected:
        raise ValueError('post_seal_mutation:files')
    return {'valid':True,'file_count':len(expected),'stage':seal['stage']}
