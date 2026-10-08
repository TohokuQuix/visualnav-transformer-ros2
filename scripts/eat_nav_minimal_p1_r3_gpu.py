#!/usr/bin/env python3
"""Candidate-free, bracketed NVML baseline for the P1 r3 resource contract."""

from __future__ import annotations

import ctypes
import json
import subprocess
import time
from pathlib import Path

from scripts.eat_nav_g0e_gpu_probe import MIB,Nvml,account,proc_table
from scripts.eat_nav_minimal_p1_freeze import read,sha,write
from visualnav_transformer.eat_nav.minimal_p1_r3 import evaluate_gpu_baseline

ROOT=Path('/tmp/eatnav-local-primitive-p1-r3-20261008-r2')


def capture(nvml):
    started=time.monotonic_ns()
    before,rows=nvml.sample()
    table_started=time.monotonic_ns()
    table=proc_table()
    table_finished=time.monotonic_ns()
    after=nvml.memory_snapshot()
    accounting=account(before,rows,table,set(),set(),'no_simulator_container_owned')
    sums={d['uuid']:0 for d in before}
    for process in accounting['processes']:
        sums[process['gpu_uuid']]+=process['used_gpu_memory_bytes']
    after_by_uuid={d['uuid']:d['used_bytes'] for d in after}
    brackets=[]
    for device in before:
        uuid=device['uuid'];left=device['used_bytes'];right=after_by_uuid[uuid];summed=sums[uuid]
        gap=(min(left,right)-summed if summed<min(left,right) else
             max(left,right)-summed if summed>max(left,right) else 0)
        brackets.append({'uuid':uuid,'device_before_bytes':left,'device_after_bytes':right,
                         'process_sum_bytes':summed,'gap_bytes':gap})
    nvml_version=ctypes.create_string_buffer(96)
    nvml._call('nvmlSystemGetNVMLVersion',nvml_version,len(nvml_version))
    identity=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,temperature.gpu,driver_version,mig.mode.current',
                             '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
    identity_rows=[line.split(', ') for line in identity.stdout.strip().splitlines()]
    if len(identity_rows)!=1 or len(identity_rows[0])!=8:
        raise ValueError('unexpected nvidia-smi hardware identity table')
    ident=identity_rows[0]
    sample={'schema':'eatnav_p1_r3_gpu_bracket_baseline_v1','wall_ns':time.time_ns(),
            'monotonic_ns':time.monotonic_ns(),'sample_started_monotonic_ns':started,
            'devices_before':before,'devices_after':after,'raw_gpu_process_rows':rows,
            'deduplicated_processes':accounting['processes'],
            'process_table_sample_started_ns':table_started,'process_table_sample_finished_ns':table_finished,
            'brackets':brackets,'query_timing':{**nvml.last_query_timing,
                **nvml.last_memory_snapshot_timing},
            'nvml_version':nvml_version.value.decode(),
            'nvidia_smi_identity_stdout':identity.stdout,
            'nvidia_smi_identity_stderr':identity.stderr,
            'nvidia_smi_identity':{'index':ident[0],'uuid':ident[1],'name':ident[2],
                'total_mib':float(ident[3]),'used_mib':float(ident[4]),
                'temperature_c':float(ident[5]),'driver_version':ident[6],
                'mig_mode':ident[7]},
            'simulator_processes_started':0}
    if len(before)!=1 or len(after)!=1 or before[0]['uuid']!=after[0]['uuid']:
        raise ValueError('GPU identity changed inside baseline bracket')
    d=before[0]
    if ident[1]!=d['uuid'] or abs(float(ident[3])-d['total_bytes']/MIB)>1:
        raise ValueError('NVML/nvidia-smi device identity mismatch')
    summary={'uuid':d['uuid'],'name':ident[2],'mig_mode':ident[7],
             'driver_version':ident[6], 'nvml_version':sample['nvml_version'],
             'total_mib':d['total_bytes']/MIB,
             'device_used_mib':max(d['used_bytes'],after[0]['used_bytes'])/MIB,
             'other_processes_used_mib':accounting['other_processes_used_bytes']/MIB,
             'unresolved_pid_bytes':accounting['unresolved_pid_bytes'],
             'bracket_gap_mib':max(brackets,key=lambda x:abs(x['gap_bytes']))['gap_bytes']/MIB,
             'temperature_c':d['temperature_c']}
    return sample,summary


def run(root=ROOT):
    root=Path(root)
    contract=read(root/'p1-resource-contract-v1.json')
    forecast=read(root/'runtime-resource-forecast-r3.json')
    if not forecast['pass'] or forecast['resource_contract_sha256']!=sha(root/'p1-resource-contract-v1.json'):
        raise ValueError('r3 resource forecast/config freeze invalid')
    nvml=Nvml()
    raw,summary=capture(nvml)
    result=evaluate_gpu_baseline(summary,contract)
    result.update({'stage':'immediately_before_X1','raw_sample_sha256':None,
                   'historical_uuid_match':summary['uuid']==contract['required_gpu_uuid'],
                   'gpu_cap_fraction_of_total':contract['gpu_device_total_hard_cap_mib']/summary['total_mib'],
                   'forecast_fraction_of_total':contract['forecast_gpu_peak_mib']/summary['total_mib']})
    write(root/'gpu-device-baseline.json',raw)
    result['raw_sample_sha256']=sha(root/'gpu-device-baseline.json')
    write(root/'gpu-baseline-gate.json',result)
    if not result['pass']:
        write(root/'final-decision.json',{'decision':'p1_resource_contract_invalid_stop',
            'failed_baseline_gates':result['failures'],
            'gpu_baseline_sha256':sha(root/'gpu-device-baseline.json'),
            'candidate_processes_started':0,'router_or_x2_started':False,
            'unrelated_processes_managed':False})
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
