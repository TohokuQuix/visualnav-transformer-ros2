#!/usr/bin/env python3
"""Offline 0.75 m common-goal candidate-bank review; no live processes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from scripts.eat_nav_curvature_attainable import (
    verify_anchor_identity, verify_base_sources, _read_follower_config,
    _source_exact_audit, FOLLOWER_CONFIG, load_anchor_inputs, sha, write, CONFIG as PRIOR_CONFIG,
)
from visualnav_transformer.eat_nav.curvature_attainable import generate_goal_library
from visualnav_transformer.eat_nav.goal_bank_offline import goal_for_bearing
from visualnav_transformer.eat_nav.goal075_offline import contract075

BASE_COMMIT='bb85939e716376017b909e564938ec00b11e849b'
ARCHIVES={
    'goal060':('eatnav-goal-bearing-bank-review-20261008-r6.tar.gz',
               'eca4650875d4972827010bbbe4cd970cb3b06b4d0b9cfcdf2516d1a204de1e6b',
               'eatnav-goal-bearing-bank-review-20261008-r6','sha-ledger.json',
               ('eight-anchor-freeze.json','state-groups.json','runtime-storage-forecast.json','final-decision.json')),
    'redesign060':('eatnav-goal-bank-redesign-20261008-r4.tar.gz',
                   '36a6b011510a633c16b0bc9ba35b8c7d941e87739dd68cb9b758308b8e619c59',
                   'eatnav-goal-bank-redesign-20261008-r4','sha-ledger.json',
                   ('selected-candidate-definitions.json','runtime-storage-forecast.json','report.md','final-decision.json')),
    'curvature060':('eatnav-curvature-attainable-20261008-r9.tar.gz',
                     'b3f0dc4a102838a8a6092a1012d550414893ca498c88c61956e543f700f3d4ac',
                     'eatnav-curvature-attainable-20261008-r9','artifact-sha-ledger.json',
                     ('source-provenance.json','attainable-set-config.json','final-decision.json',
                      'candidate-attainable-counts.json','artifact-sha-ledger.json')),
}


def _archive_audit(backup_root:Path,spec:tuple) -> tuple[dict,dict]:
    name,expected,root,ledger_name,needed=spec
    archive=backup_root/name
    if sha(archive)!=expected:
        raise ValueError('prior archive SHA mismatch: '+name)
    saved={}
    with tarfile.open(archive,'r:gz') as bundle:
        ledger_bytes=bundle.extractfile(root+'/'+ledger_name).read()
        ledger=json.loads(ledger_bytes)['files']
        for relative,record in ledger.items():
            member=bundle.extractfile(root+'/'+relative)
            if member is None:
                raise ValueError('missing archived artifact: '+relative)
            body=member.read()
            digest=hashlib.sha256(body).hexdigest()
            want=record['sha256'] if isinstance(record,dict) else record
            if digest!=want or isinstance(record,dict) and len(body)!=record['size_bytes']:
                raise ValueError('archived artifact SHA/size mismatch: '+relative)
            if relative in needed:
                saved[relative]=json.loads(body) if relative.endswith('.json') else body.decode()
        if ledger_name in needed:saved[ledger_name]=json.loads(ledger_bytes)
    if set(needed)-set(saved):
        raise ValueError('archived required artifact missing: '+str(set(needed)-set(saved)))
    info={'archive':str(archive),'archive_sha256':expected,
          'ledger_sha256':hashlib.sha256(ledger_bytes).hexdigest(),
          'verified_artifact_count':len(ledger)}
    return info,saved


def verify_inputs(backup_root):
    backup_root=Path(backup_root)
    archives={};data={}
    for key,spec in ARCHIVES.items():
        archives[key],data[key]=_archive_audit(backup_root,spec)
    anchors,anchor_source=load_anchor_inputs(backup_root)
    anchor_check=verify_anchor_identity(anchors,data['goal060']['eight-anchor-freeze.json'])
    groups=data['goal060']['state-groups.json']
    if len(groups)!=40:
        raise ValueError('40 frozen StateGroups required')
    source_exact=_source_exact_audit(backup_root,_read_follower_config(FOLLOWER_CONFIG))
    root=Path(__file__).resolve().parents[1]
    previous_source=data['curvature060']['source-provenance.json']
    current_generator=sha(root/'src/visualnav_transformer/eat_nav/curvature_attainable.py')
    current_runner=sha(root/'scripts/eat_nav_curvature_attainable.py')
    generator_sha_match=(current_generator==previous_source['attainable_generator_source_sha256']
                         and current_runner==previous_source['attainable_runner_source_sha256'])
    if not generator_sha_match:
        raise ValueError('previous G0–G3 generator or runner source SHA changed')
    if data['curvature060']['final-decision.json']['decision']!='candidate_diversity_infeasible_under_frozen_follower_contract':
        raise ValueError('unexpected 0.60m curvature decision')
    base_sources=verify_base_sources(root)
    e1_gate=root/'docs/quality-identifiability-gate/candidate-coverage/e1-feasibility/e1-gates.yaml'
    provenance={'base_commit':BASE_COMMIT,'archives':archives,
                'prior_curvature_decision':data['curvature060']['final-decision.json']['decision'],
                'generator_sha_match':generator_sha_match,
                'generator_source_sha256':current_generator,
                'prior_runner_source_sha256':current_runner,
                'base_decision_source_sha256':base_sources,
                'frozen_e1_gate_sha256':sha(e1_gate),
                'source_exact_follower':source_exact,
                'anchor_identity':anchor_check,
                'anchor_source':anchor_source,
                'current_worktree_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                'current_worktree_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True).splitlines(),
                'input_roots_from_durable_backup_only':True}
    return {'anchors':anchors,'old_groups':groups,'prior':data,'provenance':provenance}


def freeze_contract(output_root):
    output_root=Path(output_root)
    if output_root.exists():
        raise FileExistsError('new output root required: '+str(output_root))
    output_root.mkdir(parents=True)
    contract={**contract075(),
              'mode_separation':PRIOR_CONFIG['mode_separation'],
              'known_terrain_rms_min_m':PRIOR_CONFIG['known_corridor_rms_min_m'],
              'known_terrain_common_bins_min':PRIOR_CONFIG['known_corridor_common_bins_min'],
              'footprint_proxies':['main_track_sprocket','initial_flipper_support',
                                   'articulated_reach','full_collision_union'],
              'primary_bank_fixed_five':True,
              'variable_count_diagnostic_only':True,
              'no_candidate_outcome_used':True}
    write(output_root/'goal-path-contract-075m.json',contract)
    return {'contract':contract,'contract_sha256':sha(output_root/'goal-path-contract-075m.json')}


def verify_g0_regression_archive(old_groups,definitions):
    if len(old_groups)!=40 or len(definitions)!=560:
        raise ValueError('unexpected frozen G0 regression count')
    by_group={row['state_group_id']:row for row in old_groups}
    libraries={bearing:generate_goal_library(goal_for_bearing(bearing))[0]
               for bearing in (-20,-10,0,10,20)}
    matched=0
    for item in definitions:
        group=by_group[item['state_group_id']]
        family='G0A' if item['bank']=='A' else 'G0B'
        variant=f"{item['bank']}-{item['side']}-{abs(float(item['signed_amplitude_or_offset_m'])):.2f}"
        curve=next((c for c in libraries[group['bearing_deg']]
                    if c['family']==family and c['variant_id']==variant),None)
        if curve is None or curve['path_sha256']!=item['path_sha256']:
            raise ValueError('prior 0.60m G0 Path SHA mismatch: '+item['state_group_id']+':'+variant)
        matched+=1
    return {'matched_selected_prior_paths':matched,'source_exact_path_sha':True}
