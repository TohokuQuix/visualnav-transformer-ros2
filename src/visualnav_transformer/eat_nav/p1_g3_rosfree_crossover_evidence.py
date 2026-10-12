"""Offline validation of direct-core G3 control traces; no ROS transport claims."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .p1_g3_assignment import validate_frozen_assignment
from .p1_g3_shared_dem import verify_shared_artifacts
from .p1_longhorizon_protocol import G3Exposure


SCHEMA = 'eatnav_p1_g3_rosfree_crossover_v1'
EVENTS = ['pose', 'core', 'command_injection', 'actor', 'physics', 'next_pose']
STEADY = ['pose_capture_steady_ns', 'core_call_steady_ns',
          'command_injection_steady_ns', 'actor_begin_steady_ns',
          'physics_begin_steady_ns', 'next_pose_capture_steady_ns']
STAMP = ['pose_stamp_ns', 'core_call_stamp_ns', 'command_injection_stamp_ns',
         'actor_input_stamp_ns', 'physics_input_ns']


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close(a, b, limit=1e-6):
    return len(a) == len(b) and all(math.isfinite(float(x)) and
        math.isfinite(float(y)) and abs(float(x)-float(y)) <= limit
        for x, y in zip(a, b))


def validate_control_trace(rows: list[dict], env_id: int) -> dict:
    if not rows or len(rows) > 640 or env_id not in range(4):
        raise ValueError('invalid four-env trace length/id')
    first_stamp = rows[0]['logical_sim_time_ns']
    initial_episode = rows[0]['timing_protocol']['episode_length_before']
    goal_seen = False
    exposure = G3Exposure()
    for step, row in enumerate(rows):
        stamp = first_stamp + step * 50_000_000
        timing = row['timing_protocol']
        core = row['core_evidence']
        if (row['control_step'] != step or row['logical_sim_time_ns'] != stamp or
                timing['step'] != step or timing['stamp_ns'] != stamp or
                any(timing[key] != stamp for key in STAMP) or
                timing['next_pose_stamp_ns'] != stamp + 50_000_000 or
                timing['events'] != EVENTS):
            raise ValueError('core/Actor/physics logical stamp or event order mismatch')
        points = [timing[key] for key in STEADY]
        if any(not isinstance(value, int) for value in points) or points != sorted(points):
            raise ValueError('causal steady-time order mismatch')
        if (core['env_id'] != env_id or core['pose_frame'] != 'tile_local' or
                core['stamp_ns'] != stamp or not _close(core['pose'], core['pose'])):
            raise ValueError('core instance/pose/stamp mismatch')
        if step and not _close(rows[step-1]['timing_protocol']['next_pose'], core['pose']):
            raise ValueError('next pose was not used at the next core call')
        if (timing['episode_length_before'] != initial_episode + step or
                timing['episode_length_after'] != initial_episode + step + 1 or
                timing['terminated'] or timing['truncated'] or timing['reset_called'] or
                row['terminal'] is not None):
            raise ValueError('episode continuity failure')
        command = row['follower_command']
        applied = row['applied_command']
        if row['label_exposure'] != exposure.observe(step, row['follower_diagnostic']['goal_reached']):
            raise ValueError('arrival/hold/censor exposure mismatch')
        if not _close(applied, [0.,0.,0.] if core['suppressed_after_exposure'] else command):
            raise ValueError('core command injection mismatch')
        if row['follower_diagnostic']['goal_reached']:
            goal_seen = True
            if not _close(command, [0.,0.,0.]):
                raise ValueError('Goal must hold zero command')
        elif goal_seen:
            raise ValueError('Goal state reverted without Path reset')
        actor_slice = core['actor_command_slice']
        critic_slice = core['critic_command_slice']
        if (len(actor_slice)!=2 or len(critic_slice)!=2 or
                actor_slice[1]-actor_slice[0]!=3 or critic_slice[1]-critic_slice[0]!=3):
            raise ValueError('command observation slice mismatch')
        alpha = float(core['command_observation_latency_alpha'])
        if not 0<=alpha<=1:
            raise ValueError('observation latency alpha invalid')
        expected = [applied[i] + alpha*(core['command_observation_previous_raw'][i]-applied[i])
                    for i in range(3)]
        if (not _close(core['command_observation'], expected) or
                not _close(row['actor_input'][actor_slice[0]:actor_slice[1]], expected) or
                not _close(row['critic_observation'][critic_slice[0]:critic_slice[1]], applied)):
            raise ValueError('Actor/Critic sees delayed or wrong-step command')
    if not exposure.finished_after(len(rows)-1):
        raise ValueError('incomplete arrival hold or right-censor')
    return {'rows': len(rows), 'env_id': env_id, 'first_stamp_ns': first_stamp,
            'last_stamp_ns': rows[-1]['logical_sim_time_ns'], 'goal_seen': goal_seen,
            'arrival_step': exposure.arrival_step, 'right_censored': exposure.censored}


def validate_output(output: Path, manifest: dict, assigned: list[dict]) -> dict:
    output = Path(output)
    marker = json.loads((output/'complete.json').read_text())
    summary = json.loads((output/'pilot-summary.json').read_text())
    steps = marker['control_steps']
    if (marker.get('protocol_version') != SCHEMA or marker.get('Path_activations') != 1 or
            marker.get('settling_control_steps') != 20 or not 41 <= steps <= 640 or
            summary['recorded_steps'] != [steps]*4):
        raise ValueError('ROS-free completion contract mismatch')
    final = summary['core_final']
    if (final['backend'] != 'ros_free_cpp_core' or final['instance_count'] != 4 or
            final['phase'] != 'physics_done' or final['control_step_count'] != steps):
        raise ValueError('core final-state/count mismatch')
    for key, name in (
            ('initial_state_sha256','initial-state.json'), ('terrain_sha256','terrain.json'),
            ('prepath_gate_sha256','prepath-gate.json'),
            ('process_provenance_sha256','process-provenance.json'),
            ('prepath_physical_gate_sha256','prepath-physical-state-gate.json'),
            ('prepath_shared_gate_sha256','prepath-shared-dem-gate.json'),
            ('prepath_shared_ledger_sha256','prepath-shared-dem-sha-ledger.json'),
            ('prepath_native_diagnostic_sha256','prepath-native-dem-diagnostic.json'),
            ('prepath_branch_input_diagnostics_sha256','prepath-branch-input-diagnostics.json'),
            ('prepath_branch_inputs_sha256','prepath-branch-inputs.npz'),
            ('prepath_dem_ledger_sha256','prepath-dem-sha-ledger.json'),
            ('direct_actor_gate_sha256','direct-actor-gate.json'),
            ('path_activation_sha256','path-activation.json'),
            ('pilot_summary_sha256','pilot-summary.json'),
            ('contact_body_names_sha256','contact-body-names.json'),
            ('live_material_sha256','live-material.json'),
            ('prepath_crossover_snapshot_sha256','prepath-crossover-snapshot.json'),
            ('prepath_approval_sha256','prepath-approval.json')):
        if digest(output/name) != marker[key]:
            raise ValueError('referenced output SHA mismatch: '+name)
    if digest(output/'path-contract.json') != marker['path_contract_sha256']:
        raise ValueError('frozen attempt path contract SHA mismatch')
    for name in ('prepath-dem-sha-ledger.json','prepath-shared-dem-sha-ledger.json'):
        ledger = json.loads((output/name).read_text())
        for relative, expected in ledger.items():
            if Path(relative).name != relative or digest(output/relative) != expected:
                raise ValueError('raw DEM SHA mismatch: '+relative)
    if not json.loads((output/'prepath-gate.json').read_text())['pass'] or \
            not json.loads((output/'cross-period-prepath-gate.json').read_text())['pass'] or \
            not json.loads((output/'live-material.json').read_text())['pass']:
        raise ValueError('pre-Path/material gate failure')
    verify_shared_artifacts(output, require_native=True)
    activation = json.loads((output/'path-activation.json').read_text())
    if (activation['count'] != 1 or activation['control_backend'] != 'ros_free_cpp_core' or
            activation['core_env_ids'] != list(range(4)) or
            activation['core_source_sha256'] != manifest['cpp_core_source_sha256'] or
            activation['pose_frame'] != 'tile_local'):
        raise ValueError('core activation provenance mismatch')
    validate_frozen_assignment(assigned, activation['trial_paths'], stage='post_path')
    traces = []
    for env in range(4):
        trace = output/f'trace-slot-{env}.jsonl'
        if digest(trace) != marker['trace_sha256'][env]:
            raise ValueError('trace SHA mismatch')
        rows = [json.loads(line) for line in trace.open()]
        if len(rows) != steps:
            raise ValueError('trace length mismatch')
        traces.append(validate_control_trace(rows, env))
    if len({r['first_stamp_ns'] for r in traces}) != 1 or \
            len({r['last_stamp_ns'] for r in traces}) != 1:
        raise ValueError('env logical stamp disagreement')
    if final['last_stamp_ns'] != traces[0]['last_stamp_ns']:
        raise ValueError('core final stamp mismatch')
    return {'pass': True, 'trace_rows': [r['rows'] for r in traces],
            'first_stamp_ns': traces[0]['first_stamp_ns'],
            'last_stamp_ns': traces[0]['last_stamp_ns'],
            'source_sha256':manifest['g3_simulator_source_sha256']}
