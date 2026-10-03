"""One-shot, host-PID NVML attribution for G0-E startup only."""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import time

MIB = 1024 * 1024
UNKNOWN = (1 << 64) - 1


class _MemoryV2(C.Structure):
    _fields_ = [('version', C.c_uint), ('total', C.c_ulonglong),
                ('reserved', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]


class _Util(C.Structure):
    _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]


class _ProcessV3(C.Structure):
    _fields_ = [('pid', C.c_uint), ('usedGpuMemory', C.c_ulonglong),
                ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]


class Nvml:
    def __init__(self):
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self._call('nvmlInit_v2')
        self.count = C.c_uint()
        self._call('nvmlDeviceGetCount_v2', C.byref(self.count))
        if self.count.value < 1:
            raise RuntimeError('no NVML GPU')
        self.handles = []
        for i in range(self.count.value):
            handle = C.c_void_p()
            self._call('nvmlDeviceGetHandleByIndex_v2', i, C.byref(handle))
            self.handles.append(handle)

    def _call(self, symbol, *args):
        fn = getattr(self.lib, symbol)
        result = fn(*args)
        if result != 0:
            raise RuntimeError(f'{symbol} NVML error {result}')

    def _processes(self, handle, kind):
        symbol = f'nvmlDeviceGet{kind}RunningProcesses_v3'
        count = C.c_uint(256)
        table = (_ProcessV3 * count.value)()
        self._call(symbol, handle, C.byref(count), table)
        return [{'host_pid': int(x.pid), 'used_gpu_memory_bytes': int(x.usedGpuMemory),
                 'kind': kind.lower()} for x in table[:count.value]]

    def sample(self):
        devices = []
        processes = []
        for i, handle in enumerate(self.handles):
            uuid = C.create_string_buffer(96)
            self._call('nvmlDeviceGetUUID', handle, uuid, len(uuid))
            memory = _MemoryV2(); memory.version = C.sizeof(_MemoryV2) | (2 << 24)
            util = _Util(); temperature = C.c_uint(); power = C.c_uint(); mode = C.c_uint()
            self._call('nvmlDeviceGetMemoryInfo_v2', handle, C.byref(memory))
            self._call('nvmlDeviceGetUtilizationRates', handle, C.byref(util))
            self._call('nvmlDeviceGetTemperature', handle, 0, C.byref(temperature))
            self._call('nvmlDeviceGetPowerUsage', handle, C.byref(power))
            self._call('nvmlDeviceGetComputeMode', handle, C.byref(mode))
            uid = uuid.value.decode()
            devices.append({'index': i, 'uuid': uid, 'total_bytes': memory.total,
                            'used_bytes': memory.used, 'free_bytes': memory.free,
                            'driver_reserved_bytes': memory.reserved,
                            'utilization_percent': util.gpu, 'temperature_c': temperature.value,
                            'power_mw': power.value, 'compute_mode': mode.value})
            for kind in ('Compute', 'Graphics'):
                for item in self._processes(handle, kind):
                    processes.append({**item, 'gpu_index': i, 'gpu_uuid': uid})
        return devices, processes


def proc_info(pid):
    """Host /proc identity includes start ticks, namespace PID, cgroup and ancestry."""
    root = Path('/proc') / str(pid)
    try:
        stat = (root / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        status = (root / 'status').read_text().splitlines()
        get = lambda key: next((x.split(':', 1)[1].strip() for x in status if x.startswith(key + ':')), None)
        cmdline = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        uid = int(get('Uid').split()[0])
        try: executable = os.readlink(root / 'exe')
        except PermissionError: executable = None
        try: fd_count = len(list((root/'fd').iterdir()))
        except PermissionError: fd_count = None
        return {'host_pid': pid, 'container_pid': int(get('NSpid').split()[-1]),
                'namespace_pid_chain': [int(x) for x in get('NSpid').split()],
                'parent_host_pid': int(fields[1]), 'process_group': int(fields[2]),
                'session_id': int(fields[3]), 'start_ticks': int(fields[19]),
                'user': pwd.getpwuid(uid).pw_name, 'uid': uid,
                'executable': executable, 'command_line': cmdline,
                'cgroup': (root / 'cgroup').read_text().strip(),
                'rss_bytes': int((root/'status').read_text().split('VmRSS:')[1].split()[0])*1024 if 'VmRSS:' in (root/'status').read_text() else 0,
                'fd_count': fd_count,
                'thread_count': int(get('Threads')), 'cpu_ticks': int(fields[11])+int(fields[12])}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError, StopIteration):
        return None


def proc_table():
    result = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            info = proc_info(int(entry.name))
            if info: result[info['host_pid']] = info
    return result


def descendants(table, roots):
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, item in table.items():
            if item['parent_host_pid'] in selected and pid not in selected:
                selected.add(pid); changed = True
    return selected


def deduplicate_gpu(rows, table):
    merged = {}
    for row in rows:
        proc = table.get(row['host_pid'])
        identity = (row['gpu_uuid'], row['host_pid'], proc['start_ticks'] if proc else None)
        if row['used_gpu_memory_bytes'] == UNKNOWN:
            raise RuntimeError('NVML per-PID GPU memory unavailable')
        if identity not in merged:
            merged[identity] = {**row, 'kinds': [row['kind']], 'proc': proc}
        else:
            old = merged[identity]
            old['used_gpu_memory_bytes'] = max(old['used_gpu_memory_bytes'], row['used_gpu_memory_bytes'])
            old['kinds'] = sorted(set(old['kinds'] + [row['kind']]))
    return list(merged.values())


def account(devices, gpu_rows, table, roots, baseline_identities, container_id):
    tree = descendants(table, roots)
    processes = deduplicate_gpu(gpu_rows, table)
    sim = 0; other = 0; unknown = 0
    for row in processes:
        proc = row['proc']
        row['simulator_tree'] = row['host_pid'] in tree
        row['container_id'] = container_id if proc and container_id in proc['cgroup'] else None
        row['existed_before_monitor'] = bool(proc and (row['host_pid'], proc['start_ticks']) in baseline_identities)
        row['created_after_monitor'] = bool(proc and not row['existed_before_monitor'])
        size = row['used_gpu_memory_bytes']
        if proc is None: unknown += size
        elif row['simulator_tree']: sim += size
        else: other += size
    device = sum(d['used_bytes'] for d in devices)
    unattributed = device - sim - other - unknown
    return {'device_total_used_bytes':device, 'simulator_tree_used_bytes':sim,
            'other_processes_used_bytes':other, 'unresolved_pid_bytes':unknown,
            'unattributed_bytes':unattributed, 'processes':processes,
            'simulator_tree_host_pids':sorted(tree)}


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def nvidia_smi_processes():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
    return {'wall_ns':time.time_ns(),'stdout':result.stdout,'stderr':result.stderr}


def host_available_bytes():
    line=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return int(line.split()[1])*1024


def cgroup_memory(container_id):
    path=Path(f'/sys/fs/cgroup/system.slice/docker-{container_id}.scope/memory.stat')
    values={line.split()[0]:int(line.split()[1]) for line in path.read_text().splitlines()}
    events_path=path.with_name('memory.events')
    events={line.split()[0]:int(line.split()[1]) for line in events_path.read_text().splitlines()}
    return {'anon_bytes':values['anon'],'shmem_bytes':values['shmem'],'file_cache_bytes':values['file'],
            'events':events}


def safe_signal(owned_identities, sig):
    """Signal only current processes whose (host PID,start time) was recorded as owned."""
    sent=[]; skipped=[]
    for pid, start in sorted(owned_identities, reverse=True):
        current=proc_info(pid)
        if current is None or current['start_ticks']!=start:
            skipped.append({'pid':pid,'reason':'exited_or_reused'});continue
        os.kill(pid,sig);sent.append({'pid':pid,'signal':sig})
    return {'sent':sent,'skipped':skipped}
