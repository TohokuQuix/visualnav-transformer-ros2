"""Per-run PID/start-tick ownership; commands are diagnostic only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from eat_nav_g0e_gpu_probe import proc_info

MEASUREMENT={'docker_exec','simulator','bridge'}|{f'follower_{i}' for i in range(8)}
CONTROL={'runner','runtime_monitor','finalization_verifier'}
EXTERNAL={'zenoh_router'}
ALL_ROLES=MEASUREMENT|CONTROL|EXTERNAL


def ownership_spec():
    return {role:{'ownership':('measurement' if role in MEASUREMENT else
        'control' if role in CONTROL else 'external'),
        'expected_lifetime':'one_run' if role not in EXTERNAL else 'whole_schedule',
        'cleanup_responsibility':'runner' if role in MEASUREMENT else
            'self_or_runner' if role in CONTROL else 'external_router_owner'}
        for role in sorted(ALL_ROLES)}


def process_state(pid):
    try:
        raw=(Path('/proc')/str(pid)/'stat').read_text()
        return raw[raw.rfind(')')+2:].split()[0]
    except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):
        return None


def process_record(pid):
    value=proc_info(int(pid))
    if value is not None:value['state']=process_state(pid)
    return value


class ProcessRegistry:
    def __init__(self,run_id,run_start_ns):
        self.run_id=str(run_id);self.run_start_ns=int(run_start_ns);self.entries=[]

    def assert_run(self,run_id,run_start_ns):
        if (self.run_id,self.run_start_ns)!=(str(run_id),int(run_start_ns)):
            raise ValueError('process registry run identity mismatch')

    def register(self,role,info,*,ownership,expected_lifetime=None,cleanup_responsibility=None,
                 launch_monotonic_ns=None):
        if role not in ALL_ROLES:raise ValueError('unknown role '+role)
        correct='measurement' if role in MEASUREMENT else 'control' if role in CONTROL else 'external'
        if ownership!=correct:raise ValueError('role ownership mismatch '+role)
        if not isinstance(info,dict) or not info.get('host_pid') or info.get('start_ticks') is None:
            raise ValueError('missing PID/start tick for '+role)
        if any(x['role']==role and x['host_pid']==info['host_pid'] and
               x['start_ticks']==info['start_ticks'] for x in self.entries):return
        self.entries.append({'run_id':self.run_id,'run_start_ns':self.run_start_ns,
            'role':role,'ownership':ownership,'host_pid':int(info['host_pid']),
            'container_pid':info.get('container_pid'),'start_ticks':int(info['start_ticks']),
            'parent_pid':info.get('parent_host_pid'),'process_group':info.get('process_group'),
            'session_id':info.get('session_id'),'cgroup':info.get('cgroup'),
            'state_at_registration':info.get('state'),
            'command_line_diagnostic':info.get('command_line'),
            'launch_monotonic_ns':int(launch_monotonic_ns or time.monotonic_ns()),
            'expected_lifetime':expected_lifetime or ('one_run' if ownership!='external' else 'whole_schedule'),
            'cleanup_responsibility':cleanup_responsibility or ('runner' if ownership=='measurement' else
                'self_or_runner' if ownership=='control' else 'external_router_owner')})

    def register_pid(self,role,pid,*,ownership,**kwargs):
        value=process_record(pid)
        if value is None:raise ValueError('process exited before registration '+role)
        self.register(role,value,ownership=ownership,**kwargs)

    def by_role(self,role):return [x for x in self.entries if x['role']==role]

    def require_complete(self):
        found={x['role'] for x in self.entries}
        missing=ALL_ROLES-found
        if missing:raise ValueError('missing process registry role '+','.join(sorted(missing)))

    def residuals(self,lookup=process_record):
        result=[]
        for entry in self.entries:
            if entry['ownership']!='measurement':continue
            observed=lookup(entry['host_pid'])
            if observed is None or observed.get('start_ticks')!=entry['start_ticks']:continue
            result.append({'role':entry['role'],'host_pid':entry['host_pid'],
                'start_ticks':entry['start_ticks'],'state':observed.get('state'),
                'container_pid':entry['container_pid']})
        return result

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.tmp')
        tmp.write_text(json.dumps({'run_id':self.run_id,'run_start_ns':self.run_start_ns,
            'entries':self.entries},indent=2,sort_keys=True)+'\n')
        tmp.replace(path)

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        value=cls(data['run_id'],data['run_start_ns'])
        value.entries=data['entries']
        return value


def register_followers(registry,journal_path):
    events=[json.loads(line) for line in Path(journal_path).read_text().splitlines()]
    for event in events:
        if event.get('event')!='follower_spawn':continue
        branch=int(event['branch'])
        if branch not in range(8):raise ValueError('follower branch out of range')
        registry.register(f'follower_{branch}',event['process'],ownership='measurement',
                          launch_monotonic_ns=event['steady_ns'])
    if len({x['role'] for x in registry.entries if x['role'].startswith('follower_')})!=8:
        raise ValueError('eight registered follower launches required')


def container_owned_roots(table,container_id,baseline):
    baseline={tuple(x) for x in baseline}
    return {pid for pid,p in table.items() if container_id in p['cgroup']
        and (pid,p['start_ticks']) not in baseline}


def registered_alive(entry,lookup=process_record):
    observed=lookup(entry['host_pid'])
    return observed is not None and observed.get('start_ticks')==entry['start_ticks']


def stop_and_join_monitor(monitor,stop_path,final_sample,*,timeout):
    """Final resource snapshot, stop request, then join; no residual scan here."""
    final_sample()
    stop_path=Path(stop_path)
    tmp=stop_path.with_name(stop_path.name+'.tmp')
    tmp.write_text(json.dumps({'monotonic_ns':time.monotonic_ns(),'request':'finalize'})+'\n')
    tmp.replace(stop_path)
    try:monitor.wait(timeout=timeout)
    except (TimeoutError,Exception) as exc:
        raise RuntimeError('control_cleanup_failure: runtime monitor join') from exc
    if monitor.returncode!=0:raise RuntimeError('control_cleanup_failure: runtime monitor exit')
