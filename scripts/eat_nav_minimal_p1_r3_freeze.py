#!/usr/bin/env python3
"""Freeze P1 r3 anchors, 28 rows and the versioned resource forecast without Outcome."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.eat_nav_local_primitive_review import verify_inputs
from scripts.eat_nav_minimal_p1_freeze import (BACKUP, PREVIOUS, ROLE, merge_anchor_evidence,
                                                read, sha, source_map, verify_ledger, write)
from visualnav_transformer.eat_nav.local_motion_primitives import BEARINGS_DEG, DISTANCE_M, constant_curvature_primitive
from visualnav_transformer.eat_nav.minimal_p1 import build_manifest, forecast_resource
from visualnav_transformer.eat_nav.minimal_p1_r3 import select_r3

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/eat-nav/local-primitive-p1-r3-20261008'
R2=ROOT/'docs/eat-nav/local-primitive-p1-minimal-20261008'
OUTPUT=Path('/tmp/eatnav-local-primitive-p1-r3-20261008-r2')
GATE_SHA='d14c598d4824df5c22bb815148efaf0030afcf7a60b9f7990d6c22db615fcd57'
RESOURCE_SHA='978eb72ecc31a2be23a426f05c80e19b1f5d829965a7030bb4429381c7582356'
HISTORICAL_GPU_SAMPLE=BACKUP/'eatnav-expanded-feasibility-ros-continuation-20261005/baselines/expanded-train-004-right_1.jsonl'
PRIOR_MONITOR=BACKUP/'eatnav-vnt-expanded-feasibility/scripts/eat_nav_e1_monitor.py'
PRIOR_NVML=BACKUP/'eatnav-vnt-expanded-feasibility/scripts/eat_nav_g0e_gpu_probe.py'


def run():
    if sha(DOC/'minimal-p1-gate-v2.json')!=GATE_SHA or sha(OUTPUT/'minimal-p1-gate-v2.json')!=GATE_SHA:
        raise ValueError('r3 selection config SHA changed')
    if sha(DOC/'p1-resource-contract-v1.json')!=RESOURCE_SHA or sha(OUTPUT/'p1-resource-contract-v1.json')!=RESOURCE_SHA:
        raise ValueError('r3 resource contract SHA changed')
    if sorted(p.name for p in OUTPUT.iterdir())!=['minimal-p1-gate-v2.json','p1-resource-contract-v1.json']:
        raise FileExistsError('fresh r3 output root required')
    old=verify_ledger(PREVIOUS)
    role=verify_ledger(ROLE)
    r2=verify_ledger(R2)
    if read(R2/'final-decision.json')['subreason']!='prelaunch_resource_forecast_exceeds_frozen_cap':
        raise ValueError('r2 decision changed')
    if sha(ROOT/'scripts/eat_nav_e1_monitor.py')!=sha(PRIOR_MONITOR) or sha(ROOT/'scripts/eat_nav_g0e_gpu_probe.py')!=sha(PRIOR_NVML):
        raise ValueError('corrected GPU monitor/NVML source SHA drift')
    old_sample=json.loads(HISTORICAL_GPU_SAMPLE.open().readline())
    device=old_sample['devices'][0]
    resource=read(DOC/'p1-resource-contract-v1.json')
    if device['uuid']!=resource['required_gpu_uuid'] or round(device['total_bytes']/2**20)!=resource['audited_total_memory_mib']:
        raise ValueError('historical GPU identity differs')
    sources=source_map()
    sources.update({
        'r3_gate_source':{'path':str(ROOT/'src/visualnav_transformer/eat_nav/minimal_p1_r3.py'),
                          'sha256':sha(ROOT/'src/visualnav_transformer/eat_nav/minimal_p1_r3.py')},
        'r3_freeze_runner':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
        'r3_gpu_baseline_runner':{'path':str(ROOT/'scripts/eat_nav_minimal_p1_r3_gpu.py'),
                                  'sha256':sha(ROOT/'scripts/eat_nav_minimal_p1_r3_gpu.py')},
        'gpu_monitor':{'path':str(ROOT/'scripts/eat_nav_e1_monitor.py'),'sha256':sha(PRIOR_MONITOR)},
        'nvml_accounting':{'path':str(ROOT/'scripts/eat_nav_g0e_gpu_probe.py'),'sha256':sha(PRIOR_NVML)},
    })
    provenance={'schema':'eatnav_minimal_p1_r3_source_provenance_v1',
                'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'worktree':str(ROOT),
                'worktree_status_before_selection':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
                'old_primitive_review':old,'role_review':role,'r2_review':r2,
                'r2_source_provenance_sha256':sha(R2/'source-provenance.json'),
                'r2_selected_sha256':sha(R2/'selected-four-anchors.json'),
                'r2_manifest_sha256':sha(R2/'p1-run-manifest.json'),
                'historical_gpu_sample_sha256':sha(HISTORICAL_GPU_SAMPLE),
                'historical_gpu_identity':device,
                'source_paths':sources,'candidate_outcome_read':False}
    write(OUTPUT/'source-provenance.json',provenance)
    frozen=verify_inputs(BACKUP,Path('/home/kojima/eatnav-backups'))
    roles=read(ROLE/'role-aware-terrain-gate.json')['anchors']
    raw={x['anchor_id']:x for x in frozen['anchors']}
    if len(roles)!=8 or len(raw)!=8:raise ValueError('eight physical anchor records required')
    merged=[merge_anchor_evidence(row,raw[row['anchor_id']]) for row in roles]
    selection=select_r3(merged)
    gate={'schema':'minimal_p1_gate_v2','config_sha256':GATE_SHA,'candidate_outcome_read':False,
          'prior_r2_gate_decision':'p1_measurement_or_protocol_invalid_stop',
          'graded_common':'not_tested_in_p1_r3','flat_control':'not_tested_in_p1_r3',
          'result':selection}
    write(OUTPUT/'minimal-p1-gate-v2-result.json',gate)
    if not selection['pass']:
        write(OUTPUT/'final-decision.json',{'decision':'source_or_artifact_integrity_invalid',
                                            'subreason':'revised_transition_anchor_gate_failed',
                                            'candidate_processes_started':0})
        return selection
    selected=[next(row for row in merged if row['anchor_id']==aid) for aid in selection['selected']]
    for row in selected[2:]:
        onset=min(float(p['onset_m']) for p in row['paths'] if p['class_proxy']=='transition'
                  and p['known'] and p['onset_m'] is not None and float(p['onset_m'])<.6)
        if abs(onset-.2)>1e-9:raise ValueError('revised transition onset is not frozen 0.20 m')
    r2_selected=read(R2/'selected-four-anchors.json')['selected']
    if [x['anchor_id'] for x in selected[:2]] != [x['anchor_id'] for x in r2_selected[:2]]:
        raise ValueError('differential anchors changed from r2')
    write(OUTPUT/'selected-four-anchors-r3.json',{'schema':'eatnav_minimal_p1_r3_selected_four_v1',
        'selected':selected,'selection_result_sha256':sha(OUTPUT/'minimal-p1-gate-v2-result.json'),
        'r2_selected_sha256':sha(R2/'selected-four-anchors.json'),'candidate_outcome_read':False})
    curves=[constant_curvature_primitive(DISTANCE_M,b) for b in BEARINGS_DEG]
    previous=read(PREVIOUS/'primitive-path-definitions.json')['candidates']
    if [(x['bearing_deg'],x['path_sha256']) for x in previous] != [(x['bearing_deg'],x['path_sha256']) for x in curves]:
        raise ValueError('five primitive Path SHA changed')
    compact=[{'bearing_deg':x['bearing_deg'],'path_sha256':x['path_sha256'],
              'endpoint':x['endpoint'].tolist(),'waypoints':x['waypoints'].tolist()} for x in curves]
    write(OUTPUT/'frozen-primitive-paths-r3.json',{'paths':compact,
        'previous_primitive_paths_sha256':sha(PREVIOUS/'primitive-path-definitions.json')})
    manifest=build_manifest(selected,compact,output_root=str(OUTPUT),
                            source_sha={k:v['sha256'] for k,v in sources.items()},seed=123)
    manifest['schema']='eatnav_minimal_p1_r3_manifest_v1'
    manifest['selected_four_sha256']=sha(OUTPUT/'selected-four-anchors-r3.json')
    manifest['frozen_primitive_paths_sha256']=sha(OUTPUT/'frozen-primitive-paths-r3.json')
    manifest['r2_manifest_sha256']=sha(R2/'p1-run-manifest.json')
    write(OUTPUT/'p1-run-manifest-r3.json',manifest)
    write(OUTPUT/'input-freeze.json',{
        'gate_config_sha256':GATE_SHA,'resource_contract_sha256':RESOURCE_SHA,
        'source_provenance_sha256':sha(OUTPUT/'source-provenance.json'),
        'selected_four_sha256':sha(OUTPUT/'selected-four-anchors-r3.json'),
        'manifest_sha256':sha(OUTPUT/'p1-run-manifest-r3.json'),
        'primitive_paths_sha256':sha(OUTPUT/'frozen-primitive-paths-r3.json'),
        'candidate_outcome_read':False,'candidate_retry_count':0})
    measured=read(PREVIOUS/'runtime-resource-forecast.json')
    measured['gpu_device_total_hard_cap_mib']=7680
    forecast=forecast_resource(measured,wall_cap_hours=4.,storage_cap_gb=1.,
                               container_anon_shmem_cap_gib=6.)
    if (abs(forecast['gpu_device_peak_with_margin_mib']-7138.8)>1e-6 or
            abs(7680-forecast['gpu_device_peak_with_margin_mib']-541.2)>1e-6):
        raise ValueError('prospective GPU margin arithmetic changed')
    forecast.update({'schema':'eatnav_minimal_p1_r3_forecast_v1',
                     'resource_contract_sha256':RESOURCE_SHA,
                     'previous_measured_forecast_sha256':sha(PREVIOUS/'runtime-resource-forecast.json'),
                     'baseline_gate_pending':True})
    write(OUTPUT/'runtime-resource-forecast-r3.json',forecast)
    return selection


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
