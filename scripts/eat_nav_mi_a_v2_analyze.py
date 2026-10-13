"""Read-only validation and descriptive analysis of nine sealed MI-A v2 pilot runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from visualnav_transformer.eat_nav.mi_a_v2_analysis import summarize_labels

try:
    from .eat_nav_mi_a_v2_host import sha, verify_sim_seal
    from .eat_nav_mi_a_v2_pilot import MANIFEST, PROTOCOL, validate_pilot_manifest
except ImportError:
    from eat_nav_mi_a_v2_host import sha, verify_sim_seal
    from eat_nav_mi_a_v2_pilot import MANIFEST, PROTOCOL, validate_pilot_manifest


def _attempt_name(row):
    return f"{row['index']:02d}-" + (f"prepath-{row['repeat']}" if row['stage'] == 'prepath'
                                      else f"{row['candidate']}-{row['stage']}")


def verify_outer(attempt: Path):
    ledger = attempt/'outer-sha-ledger.json'
    seal = json.loads((attempt/'outer-seal.json').read_text())
    expected = json.loads(ledger.read_text())
    actual = {str(p.relative_to(attempt)): sha(p) for p in attempt.rglob('*')
              if p.is_file() and p.name not in ('outer-sha-ledger.json', 'outer-seal.json')}
    if sha(ledger) != seal['ledger_sha256'] or expected != actual or \
            seal['file_count'] != len(actual):
        raise ValueError('outer SHA/seal mismatch')
    return sha(attempt/'outer-seal.json')


def analyze(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads(MANIFEST.read_text())
    rows = validate_pilot_manifest(manifest)
    frozen = json.loads((root/'frozen-provenance.json').read_text())
    if frozen['manifest_sha256'] != sha(MANIFEST) or \
            frozen['protocol_sha256'] != sha(PROTOCOL) or frozen['assignment'] != rows:
        raise ValueError('frozen pilot provenance mismatch')
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'pilot_complete' or status['completed'] != 9:
        raise ValueError('nine-process pilot incomplete')
    by_candidate, repeat = {}, None
    prepath = []
    mirrors = []
    seals = []
    for row in rows:
        attempt = root/_attempt_name(row)
        decision = json.loads((attempt/'decision.json').read_text())
        if not decision['pass']:
            raise ValueError(f'host-invalid attempt {attempt.name}')
        seal = verify_outer(attempt)
        verification = verify_sim_seal(attempt/'sim-output')
        expected_mode = 'prepath_only' if row['stage'] == 'prepath' else 'candidate'
        if verification['mode'] != expected_mode:
            raise ValueError('attempt mode mismatch')
        input_manifest = json.loads((attempt/'manifest.json').read_text())
        if input_manifest['protocol_sha256'] != frozen['protocol_sha256'] or \
                input_manifest['source_sha256'] != frozen['sim_source_sha256'] or \
                input_manifest['host_source_sha256'] != frozen['host_source_sha256']:
            raise ValueError('attempt source mismatch')
        pair_gate = json.loads((attempt/'sim-output/prepath-pair-gate.json').read_text())
        if not pair_gate['pass']:
            raise ValueError('Gate P failed in sealed attempt')
        prepath.append({'index': row['index'], 'name': attempt.name,
                        'gate_p': pair_gate, 'seal_sha256': seal})
        mirrors.append({'index': row['index'], 'diagnostic': json.loads(
            (attempt/'sim-output/mirror-diagnostic.json').read_text())})
        seals.append({'index': row['index'], 'outer_sha256': seal,
                      'inner_sha256': verification['simulator_seal_sha256']})
        if expected_mode == 'candidate':
            activation = json.loads((attempt/'sim-output/path-activation.json').read_text())
            if activation['count_per_env'] != [1, 1, 1, 1]:
                raise ValueError('Path publication count')
            controller = json.loads((attempt/'sim-output/controller-initial.json').read_text())
            if controller['fresh_instance_count'] != 4 or any(
                    state != [0., 0., 0., 1.] for state in controller['state_per_env']):
                raise ValueError('initial controller state')
            labels = json.loads((attempt/'sim-output/motion-labels.json').read_text())['labels']
            if row['stage'] == 'repeat':
                repeat = labels
            else:
                by_candidate[row['candidate']] = labels
    if repeat is None:
        raise ValueError('fresh-process repeat missing')
    outcome = summarize_labels(by_candidate, repeat, seed=manifest['terrain_seed'])
    return {'schema': 'mi_a_v2_pilot_analysis_v1', 'status': 'pilot_complete_descriptive',
            'attempt_seals': seals, 'prepath_identity': prepath,
            'mirror_diagnostics': mirrors, 'outcome': outcome,
            'formal_mi_a_gate': 'pending_independent_seed_schedule'}


def save(root: Path, analysis: dict):
    root = Path(root)
    outputs = {
        'prepath_identity_audit.json': analysis['prepath_identity'],
        'fresh_process_repeatability.json': analysis['outcome']['forward_direct_fresh_process_repeat_difference'],
        'mi_a_v2_pilot_results.json': analysis['outcome']['raw_motion'],
        'mi_a_v2_candidate_differences.json': analysis['outcome']['candidate_differences'],
        'mirror_diagnostics.json': analysis['mirror_diagnostics'],
        'mi_a_v2_analysis_provenance.json': {k: analysis[k] for k in
                                           ('schema', 'status', 'attempt_seals', 'formal_mi_a_gate')},
    }
    for name, value in outputs.items():
        with (root/name).open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    ledger = {name: sha(root/name) for name in outputs}
    ledger['analyzer_source_sha256'] = sha(__file__)
    with (root/'analysis-sha-ledger.json').open('x') as stream:
        json.dump(ledger, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    save(args.root, result)
    print(json.dumps({'status': result['status'], 'attempts': len(result['attempt_seals'])},
                     sort_keys=True))
