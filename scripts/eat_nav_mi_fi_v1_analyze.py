"""Read-only validation and descriptive analysis of sealed MI-FI v1 attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.mi_fi_v1_analysis import summarize_interactions
from eat_nav_mi_fi_v1_host import sha, verify_sim_seal
from eat_nav_mi_fi_v1_pilot import MANIFEST, PROTOCOL, validate_pilot_manifest


def _name(row):
    return f"{row['index']:02d}-{row['condition']}-" + (
        f"prepath-{row['repeat']}" if row['stage'] == 'prepath' else row['candidate'])


def verify_outer(attempt: Path):
    ledger = attempt/'outer-sha-ledger.json'
    seal = json.loads((attempt/'outer-seal.json').read_text())
    expected = json.loads(ledger.read_text())
    actual = {str(p.relative_to(attempt)): sha(p) for p in attempt.rglob('*')
              if p.is_file() and p.name not in ('outer-sha-ledger.json', 'outer-seal.json')}
    if sha(ledger) != seal['ledger_sha256'] or expected != actual or \
            seal['file_count'] != len(actual):
        raise ValueError(f'outer SHA/seal mismatch: {attempt.name}')
    return sha(attempt/'outer-seal.json')


def analyze(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads(MANIFEST.read_text())
    rows = validate_pilot_manifest(manifest)
    frozen = json.loads((root/'frozen-provenance.json').read_text())
    if frozen['manifest_sha256'] != sha(MANIFEST) or \
            frozen['protocol_sha256'] != sha(PROTOCOL) or frozen['assignment'] != rows:
        raise ValueError('frozen provenance mismatch')
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'pilot_complete' or status['completed'] != 16:
        raise ValueError('MI-FI pilot incomplete')
    labels = {condition: {} for condition in manifest['conditions']}
    prepath = []
    seals = []
    reset = {}
    post = {}
    trace_diagnostics = {}
    for row in rows:
        attempt = root/_name(row)
        if not json.loads((attempt/'decision.json').read_text())['pass']:
            raise ValueError(f'host-invalid attempt: {attempt.name}')
        outer = verify_outer(attempt)
        output = attempt/'sim-output'
        inner = verify_sim_seal(output)
        expected_mode = 'prepath_only' if row['stage'] == 'prepath' else 'candidate'
        if inner['mode'] != expected_mode:
            raise ValueError('mode mismatch')
        host_manifest = json.loads((attempt/'manifest.json').read_text())
        if any(host_manifest[key] != frozen[frozen_key] for key, frozen_key in
               (('source_sha256', 'sim_source_sha256'),
                ('host_source_sha256', 'host_source_sha256'),
                ('protocol_sha256', 'protocol_sha256'))):
            raise ValueError('source SHA mismatch')
        if host_manifest['condition'] != row['condition']:
            raise ValueError('condition mismatch')
        gate = json.loads((output/'prepath-pair-gate.json').read_text())
        if gate['pass'] is not True:
            raise ValueError('Gate P failed')
        prepath.append({'attempt': attempt.name, 'condition': row['condition'], 'gate': gate})
        seals.append({'attempt': attempt.name, 'outer_sha256': outer,
                      'inner_sha256': inner['simulator_seal_sha256']})
        if row['stage'] == 'prepath' and row['repeat'] == 0:
            reset[row['condition']] = json.loads((output/'flipper-reset.json').read_text())
            with np.load(output/'prepath.npz', allow_pickle=False) as data:
                post[row['condition']] = {
                    'flipper_joint_rad': data['flipper_prepath_joint_pos'].tolist(),
                    'flipper_joint_velocity_rad_s': data['flipper_prepath_joint_vel'].tolist(),
                    'root_state_w': data['root'].tolist(),
                    'actor_pre_noise_22d': data['actor_pre_noise'].tolist(),
                    'actor_input_22d': data['actor_input'].tolist(),
                    'dem_shape': list(data['dem'].shape),
                    'contact_force_shape': list(data['contact_force'].shape)}
        if row['stage'] != 'candidate':
            continue
        activation = json.loads((output/'path-activation.json').read_text())
        if activation['count_per_env'] != [1, 1, 1, 1]:
            raise ValueError('candidate Path count')
        labels[row['condition']][row['candidate']] = json.loads(
            (output/'motion-labels.json').read_text())['labels']
        streams = []
        for env in range(4):
            trace = [json.loads(line) for line in (output/f'trace-env-{env}.jsonl').read_text().splitlines()]
            if len(trace) != 40 or any(item['control_step'] != step for step, item in enumerate(trace)):
                raise ValueError('candidate trace step contract')
            streams.append({'env': env,
                'flipper_initial_rad': trace[0]['flipper_joint_position'],
                'flipper_final_rad': trace[-1]['flipper_joint_position'],
                'flipper_max_abs_torque': np.abs([item['flipper_applied_torque'] for item in trace]).max(axis=0).tolist(),
                'flipper_min_limit_margin_rad': np.asarray(
                    [item['flipper_joint_soft_limit_margin'] for item in trace]).min(axis=0).tolist(),
                'contact_peak_n': float(max(np.linalg.norm(item['contact_net_forces_w'], axis=-1).max()
                                            for item in trace)),
                'termination_count': sum(item['terminal'] is not None for item in trace)})
        trace_diagnostics[f"{row['condition']}:{row['candidate']}"] = streams
    outcome = summarize_interactions(labels)
    return {'schema': 'mi_fi_v1_analysis_v1', 'seals': seals, 'prepath': prepath,
            'initialization': {'reset': reset, 'post_settling': post},
            'motion': outcome, 'trace_diagnostics': trace_diagnostics}


def save(root: Path, result: dict):
    outputs = {
        'initialization_state_comparison.json': result['initialization'],
        'prepath_identity_audit.json': result['prepath'],
        'mi_fi_v1_motion_results.json': result['motion'],
        'flipper_trace_diagnostics.json': result['trace_diagnostics'],
        'mi_fi_v1_seals.json': result['seals'],
    }
    for name, value in outputs.items():
        with (root/name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    ledger = {name: sha(root/name) for name in outputs}
    ledger['analyzer_source_sha256'] = sha(__file__)
    with (root/'analysis-sha-ledger.json').open('x') as stream:
        json.dump(ledger, stream, sort_keys=True, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    save(args.root, result)
    print(json.dumps({'attempts': len(result['seals']), 'schema': result['schema']}))
