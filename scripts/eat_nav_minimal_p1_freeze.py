#!/usr/bin/env python3
"""Freeze outcome-blind four-anchor P1 inputs and reject unsafe prelaunch budgets."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

from scripts.eat_nav_local_primitive_review import verify_inputs, runtime_goal_dependency_audit
from visualnav_transformer.eat_nav.local_motion_primitives import BEARINGS_DEG, DISTANCE_M, constant_curvature_primitive
from visualnav_transformer.eat_nav.minimal_p1 import build_manifest, forecast_resource, select_four


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/eat-nav/local-primitive-p1-minimal-20261008'
PREVIOUS = ROOT/'docs/eat-nav/local-motion-primitives-20261008'
ROLE = ROOT/'docs/eat-nav/local-primitive-p1-role-aware-20261008'
BACKUP = Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown')
OUTPUT = Path('/tmp/eatnav-local-primitive-p1-minimal-20261008-r2')
GATE_SHA = 'd5db05e611eba28eeaf5c8e0dd8013bc4b1351bbceff71941ee71aed93b0a05f'
EXPANDED_SAFETY = BACKUP/'eatnav-expanded-feasibility-ros-continuation-20261005/e1-safety-stops.yaml'


def sha(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):
            hasher.update(part)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def verify_ledger(root):
    root=Path(root)
    ledger=read(root/'artifact-sha-ledger.json')['files']
    for name, item in ledger.items():
        path=root/name
        if (not path.is_file() or path.stat().st_size!=item['size_bytes'] or
                sha(path)!=item['sha256']):
            raise ValueError('artifact SHA/size mismatch: '+str(path))
    return {'root':str(root),'ledger_sha256':sha(root/'artifact-sha-ledger.json'),
            'verified_file_count':len(ledger)}


def merge_anchor_evidence(role, raw):
    if role['anchor_id']!=raw['anchor_id']:
        raise ValueError('anchor identity mismatch')
    if role['dem_history_sha256']!=raw['dem_history_sha256']:
        raise ValueError('DEM history SHA mismatch')
    if role['source_path_shas']!=raw['source_path_shas']:
        raise ValueError('anchor source SHA mismatch')
    if bool(role['physical_eligible'])!=bool(raw['physical_supply']):
        raise ValueError('physical eligibility mismatch')
    body=np.asarray(raw['body_state_actor'],dtype=float)
    if body.shape!=(22,) or not np.isfinite(body).all():
        raise ValueError('invalid Actor body-state source')
    return {**role,'body_state_actor':body.tolist(),
            'source_shard':raw['source_shard'],
            'actor_config_identity':raw['actor_config_identity']}


def source_map():
    primitive_provenance=read(PREVIOUS/'source-provenance.json')
    checkpoint=Path(primitive_provenance['RL_checkpoint']['read_only_audit_copy'])
    sources={
        'checkpoint_audit_copy':checkpoint,
        'reference_eat_config':ROOT/'docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
        'simulator':ROOT/'scripts/eat_nav_e1_sim.py',
        'bridge':ROOT/'scripts/eat_nav_command_delivery_bridge.py',
        'bridge_client':ROOT/'src/visualnav_transformer/eat_nav/phase_c_bridge_client.py',
        'follower_source':Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src/simple_trajectory_follower.cpp'),
        'follower_params':Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'),
        'follower_binary':BACKUP/'eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node',
        'expanded_e1_safety':EXPANDED_SAFETY,
        'original_e1_numeric_gates':ROOT/'docs/quality-identifiability-gate/candidate-coverage/e1-feasibility/e1-gates.yaml',
        'previous_primitive_ledger':PREVIOUS/'artifact-sha-ledger.json',
        'previous_role_ledger':ROLE/'artifact-sha-ledger.json',
        'role_gate_source':ROOT/'src/visualnav_transformer/eat_nav/role_aware_terrain.py',
        'new_gate_source':ROOT/'src/visualnav_transformer/eat_nav/minimal_p1.py',
        'new_freeze_runner':Path(__file__),
        'new_gate_config':DOC/'minimal-p1-gate-v1.json',
    }
    result={key:{'path':str(path),'sha256':sha(path)} for key,path in sources.items()}
    if result['checkpoint_audit_copy']['sha256']!=primitive_provenance['RL_checkpoint']['sha256']:
        raise ValueError('RL checkpoint SHA changed')
    follower_contract=primitive_provenance['source_exact_follower']
    if result['follower_source']['sha256']!=follower_contract['production_cpp_sha256']:
        raise ValueError('follower source SHA changed')
    if result['follower_params']['sha256']!=follower_contract['production_config_sha256']:
        raise ValueError('follower config SHA changed')
    if result['follower_binary']['sha256']!='616097e44d199b8d33f6534aedd299f2d20127ed8e100c18777f9ca7a2da389d':
        raise ValueError('follower binary SHA changed')
    exact=runtime_goal_dependency_audit()
    if not exact['relabeling_source_valid']:
        raise ValueError('frozen global goal source contract invalid')
    return result


def available_ram_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024/2**30
    raise ValueError('MemAvailable unavailable')


def run():
    if sha(DOC/'minimal-p1-gate-v1.json')!=GATE_SHA or sha(OUTPUT/'minimal-p1-gate-v1.json')!=GATE_SHA:
        raise ValueError('prospective minimal gate SHA changed')
    if sorted(OUTPUT.iterdir())!=[OUTPUT/'minimal-p1-gate-v1.json']:
        raise FileExistsError('new output root required')
    old=verify_ledger(PREVIOUS)
    role=verify_ledger(ROLE)
    if read(PREVIOUS/'final-decision.json')['decision']!='local_motion_primitive_terrain_support_fail':
        raise ValueError('previous primitive decision changed')
    if read(ROLE/'final-decision.json')['decision']!='role_aware_terrain_support_insufficient':
        raise ValueError('previous role decision changed')
    sources=source_map()
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines()
    provenance={'schema':'eatnav_minimal_p1_source_provenance_v1',
                'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'worktree':str(ROOT),'worktree_status_before_selection':status,
                'primitive_review':old,'role_review':role,
                'source_paths':sources,'candidate_outcome_read':False,
                'durable_anchor_backup_root':str(BACKUP)}
    write(OUTPUT/'source-provenance.json',provenance)
    # This routine independently validates the two radial seeds, settling
    # records, four DEM frames, body states, follower fixture and G0/G1 sources.
    frozen=verify_inputs(BACKUP,Path('/home/kojima/eatnav-backups'))
    roles=read(ROLE/'role-aware-terrain-gate.json')['anchors']
    raw={x['anchor_id']:x for x in frozen['anchors']}
    if len(roles)!=8 or len(raw)!=8:
        raise ValueError('eight anchored role records required')
    merged=[merge_anchor_evidence(row,raw[row['anchor_id']]) for row in roles]
    selection=select_four(merged)
    gate={'schema':'minimal_p1_gate_v1','gate_config_sha256':GATE_SHA,
          'prior_five_role_decision':'role_aware_terrain_support_insufficient',
          'graded_common':'not_required_for_minimal_p1',
          'flat_control':'not_required_for_minimal_p1',
          'candidate_outcome_read':False,'physical_anchor_count':len(merged),
          'result':selection}
    write(OUTPUT/'minimal-p1-gate.json',gate)
    if not selection['pass']:
        write(OUTPUT/'final-decision.json',{'decision':'minimal_p1_anchor_support_invalid',
                                            'rollout_started':False})
        return selection
    selected=[next(row for row in merged if row['anchor_id']==aid) for aid in selection['selected']]
    selected_doc={'schema':'eatnav_minimal_p1_selected_four_v1',
                  'selection_input_sha256':sha(OUTPUT/'minimal-p1-gate.json'),
                  'selected':selected,'alternatives_by_slot':selection['alternatives_by_slot'],
                  'candidate_outcome_read':False}
    write(OUTPUT/'selected-four-anchors.json',selected_doc)
    old_paths=read(PREVIOUS/'primitive-path-definitions.json')['candidates']
    curves=[constant_curvature_primitive(DISTANCE_M,b) for b in BEARINGS_DEG]
    if [(p['bearing_deg'],p['path_sha256']) for p in old_paths]!=[(c['bearing_deg'],c['path_sha256']) for c in curves]:
        raise ValueError('frozen five-Path SHA mismatch')
    geometry=read(PREVIOUS/'primitive-geometry-audit.json')
    if not geometry['geometry_gate'] or not geometry['follower_gate'] or geometry['hard_valid_count']!=40:
        raise ValueError('frozen Path hard/follower gate failed')
    compact=[{'bearing_deg':c['bearing_deg'],'path_sha256':c['path_sha256'],
              'endpoint':c['endpoint'].tolist(),
              'waypoints':c['waypoints'].tolist()} for c in curves]
    write(OUTPUT/'frozen-primitive-paths.json',{'paths':compact,
                                                'previous_path_definitions_sha256':sha(PREVIOUS/'primitive-path-definitions.json')})
    manifest=build_manifest(selected,compact,output_root=str(OUTPUT),source_sha={k:v['sha256'] for k,v in sources.items()},seed=123)
    manifest['selected_four_sha256']=sha(OUTPUT/'selected-four-anchors.json')
    manifest['frozen_primitive_paths_sha256']=sha(OUTPUT/'frozen-primitive-paths.json')
    write(OUTPUT/'p1-run-manifest.json',manifest)
    write(OUTPUT/'input-freeze.json',{'gate_config_sha256':GATE_SHA,
                                      'selected_four_sha256':sha(OUTPUT/'selected-four-anchors.json'),
                                      'run_manifest_sha256':sha(OUTPUT/'p1-run-manifest.json'),
                                      'frozen_primitive_paths_sha256':sha(OUTPUT/'frozen-primitive-paths.json'),
                                      'source_provenance_sha256':sha(OUTPUT/'source-provenance.json'),
                                      'retry_count':0,'candidate_outcome_read':False})
    measured=read(PREVIOUS/'runtime-resource-forecast.json')
    safety=yaml.safe_load(EXPANDED_SAFETY.read_text())['infrastructure_stop']
    if (safety['gpu_device_total_memory_over_mib']!=6656 or
            safety['schedule_wall_clock_over_sec']!=14400 or
            safety['storage_over_bytes']!=1000000000):
        raise ValueError('expanded E1 resource cap changed')
    forecast=forecast_resource(measured,wall_cap_hours=4.,storage_cap_gb=1.,
                               container_anon_shmem_cap_gib=safety['cgroup_anon_plus_shmem_over_bytes']/2**30,
                               host_available_ram_gib=available_ram_gib()-safety['host_mem_available_under_bytes']/2**30)
    forecast.update({'schema':'eatnav_minimal_p1_resource_preflight_v1',
                     'measured_forecast_sha256':sha(PREVIOUS/'runtime-resource-forecast.json'),
                     'expanded_e1_safety_sha256':sha(EXPANDED_SAFETY),
                     'hard_caps':safety,
                     'gpu_margin_contract':'measured device-total peak × 1.20 must not exceed frozen hard cap',
                     'pilot_started':False})
    write(OUTPUT/'runtime-resource-preflight.json',forecast)
    if not forecast['pass']:
        write(OUTPUT/'final-decision.json',{'decision':'p1_measurement_or_protocol_invalid_stop',
                                            'subreason':'prelaunch_resource_forecast_exceeds_frozen_cap',
                                            'failed_resource_gates':forecast['failures'],
                                            'rollout_started':False,'candidate_processes_started':0,
                                            'router_or_x2_started':False})
    return selection


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
