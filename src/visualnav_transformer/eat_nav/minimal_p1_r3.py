"""Prospectively versioned P1 r3 transition-anchor selection."""

from __future__ import annotations

import math


SLOTS=(('first','R-differential','first-differential'),
       ('second','R-differential','second-differential'),
       ('first','R-transition-common','first-graded'),
       ('second','R-transition-common','second-graded'))


def _transition_paths(row):
    return [p for p in row['paths'] if p['known'] and p['class_proxy']=='transition'
            and p['onset_m'] is not None and 0<=float(p['onset_m'])<.6]


def _eligible(row,role):
    if (row['primary_role']!=role or not row['physical_eligible'] or
            not row['hard_valid'] or not row['follower_compatible'] or
            len(row['dem_history_sha256'])!=4 or len(row['body_state_actor'])!=22 or
            not all(math.isfinite(float(v)) for v in row['body_state_actor']) or
            len(row['paths'])!=5 or not all(p['known'] for p in row['paths'])):
        return False
    if role=='R-differential':
        return row['pair_count']>0 and row['max_pair_rms_m']>=.01
    return bool(_transition_paths(row))


def select_r3(rows):
    options=[]
    for seed,role,required_id in SLOTS:
        candidates=[r for r in rows if r['seed_label']==seed and _eligible(r,role)]
        if role=='R-transition-common':
            candidates.sort(key=lambda r:(min(float(p['onset_m']) for p in _transition_paths(r)),
                                          -min(float(p['known_fraction']) for p in r['paths']),
                                          -max(float(p['class_diagnostic']['maximum_step_m'])-.05
                                               for p in _transition_paths(r)),
                                          r['selection_rank'],r['anchor_id']))
        else:
            candidates.sort(key=lambda r:(-float(r['max_pair_rms_m']),r['selection_rank'],r['anchor_id']))
        options.append({'seed_label':seed,'primary_role':role,'required_anchor_id':required_id,
                        'eligible_order':[r['anchor_id'] for r in candidates],
                        'known_onset_m_by_anchor':{r['anchor_id']:
                            min(float(p['onset_m']) for p in _transition_paths(r))
                            for r in candidates} if role=='R-transition-common' else {}})
    selected=[x['eligible_order'][0] if x['eligible_order'] else None for x in options]
    valid=(selected==[x[2] for x in SLOTS] and len(set(selected))==4)
    return {'pass':valid,'decision':None if valid else 'source_or_artifact_integrity_invalid',
            'selected':selected if valid else [],'slot_audit':options,
            'candidate_outcome_access':False}


def evaluate_gpu_baseline(sample, contract):
    failures=[]
    if sample['uuid']!=contract['required_gpu_uuid']:
        failures.append('gpu_uuid_changed')
    if sample['name']!=contract['required_gpu_name']:
        failures.append('gpu_name_changed')
    if abs(sample['total_mib']-contract['audited_total_memory_mib'])>1:
        failures.append('gpu_memory_profile_changed')
    if sample['mig_mode'] not in ('[N/A]', 'Disabled'):
        failures.append('gpu_mig_profile_changed')
    if sample['total_mib']<contract['required_total_memory_mib_min']:
        failures.append('gpu_total_below_minimum')
    if sample['device_used_mib']>contract['gpu_device_total_hard_cap_mib']:
        failures.append('gpu_device_over_cap')
    if sample['other_processes_used_mib']>contract['other_process_gpu_hard_cap_mib']:
        failures.append('other_process_gpu_over_cap')
    if sample['unresolved_pid_bytes']>contract['unresolved_gpu_bytes_hard_cap']:
        failures.append('unresolved_gpu_pid_memory')
    if abs(sample['bracket_gap_mib'])>contract['absolute_attribution_gap_hard_cap_mib']:
        failures.append('gpu_attribution_gap')
    if not math.isfinite(float(sample['temperature_c'])):
        failures.append('gpu_temperature_nonfinite')
    return {'pass':not failures,'failures':failures,'sample':sample,
            'contract_schema':contract['schema'] if 'schema' in contract else None}
