"""Read-only audit of the sealed crossover period-2 evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_g3_crossover_v2_continuation import sha, verify_prior_pair, write_new
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment
from visualnav_transformer.eat_nav.p1_g3_crossover_runtime import compare_prepath
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import validate_four_follower_readiness


SCHEMA = 'eatnav_p1_g3_crossover_period2_offline_audit_v1'
FAILURE = 'ValueError:crossover Path publication contract failed'


def classify_checks(checks: dict[str, str]) -> str:
    if any(value == 'fail' for value in checks.values()):
        return 'audit_fail'
    if any(value != 'pass' for value in checks.values()):
        return 'audit_inconclusive'
    return 'audit_pass'


def _require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def audit(root: Path, audit_root: Path, criteria_sha: str) -> dict:
    """All source roots are read-only; replay writes only to a disposable shadow."""
    from eat_nav_p1_g3_crossover_v3_host import validate_g3_output, validate_sample

    root = Path(root)
    audit_root = Path(audit_root)
    criteria = audit_root/'criteria.json'
    _require(sha(criteria) == criteria_sha, 'frozen audit criteria SHA drift')
    manifest = json.loads((root/'continuation-manifest.json').read_text())
    contract = json.loads((root/'manifest.json').read_text())
    batch = root/'batch-first-differential-0-period-2'
    output = batch/'output'
    checks = {}
    evidence = {'schema': SCHEMA, 'criteria_sha256': criteria_sha,
                'original_period2_stage_preserved': 'G3_crossover_batch_failed_or_inconclusive',
                'source_sha256': sha(Path(__file__)), 'new_host_source_sha256':
                sha(Path(__file__).with_name('eat_nav_p1_g3_crossover_v3_host.py'))}
    try:
        verify_prior_pair(root, manifest['prior_periods'])
        checks['period0_period1_valid_seals'] = 'pass'
        seal = verify_seal(batch)
        _require(seal['stage'] == evidence['original_period2_stage_preserved'],
                 'original period-2 stage unexpectedly changed')
        _require(json.loads((batch/'g3-host-status.json').read_text())['failure'] == FAILURE,
                 'original host failure differs from isolated validator defect')
        _require(sha(batch/'host-runner.py') == manifest['period2_host_source_sha256'] and
                 sha(batch/'source.py') == manifest['period2_simulator_source_sha256'] and
                 sha(batch/'input/g3-contract.json') == manifest['period2_manifest_sha256'] and
                 sha(root/'g3-assignment.json') == manifest['assignment_sha256'],
                 'frozen source or manifest SHA mismatch')
        for name, expected in json.loads((batch/'input-sha-ledger.json').read_text()).items():
            _require(sha(batch/name) == expected, 'period-2 staged input SHA mismatch:'+name)
        checks['period2_failed_seal_sha_valid'] = 'pass'
        checks['frozen_manifest_source_assignment_shas'] = 'pass'

        wanted = next(p for p in contract['processes'] if p['anchor_id']=='first-differential'
                      and p['batch_index']==0 and p['execution_index']==2)
        frozen = wanted['trials']
        staged = next(p for p in json.loads((batch/'input/g3-assignment.json').read_text())['processes']
                      if p['anchor_id']=='first-differential' and p['batch_index']==0
                      and p['execution_index']==2)['trials']
        validate_frozen_assignment(frozen, staged, stage='prelaunch')
        publication = json.loads((output/'path-publication.json').read_text())
        validate_frozen_assignment(frozen, publication['trial_paths'], stage='post_path')
        _require(publication['count'] == 1 and
                 json.loads((output/'complete.json').read_text())['Path_publications'] == 1,
                 'Path publication count mismatch')
        checks['path_role_sha_and_publication_count'] = 'pass'

        prior = root/'batch-first-differential-0-period-0/output/prepath-crossover-snapshot.json'
        current = output/'prepath-crossover-snapshot.json'
        equality = compare_prepath(json.loads(prior.read_text()), json.loads(current.read_text()))
        gate = json.loads((output/'cross-period-prepath-gate.json').read_text())
        approval = json.loads((output/'prepath-approval.json').read_text())
        _require(equality['pass'] and gate['pass'] and not gate['strict_failures'] and
                 gate['live_material_equal'] and approval['pass'] and
                 approval['snapshot_sha256'] == sha(current) and
                 gate['baseline_snapshot_sha256'] == sha(prior),
                 'cross-period pre-Path strict mismatch')
        checks['cross_period_strict_prepath'] = 'pass'
        material = json.loads((output/'live-material.json').read_text())
        effective = material['robot_effective_physx']
        _require(material['pass'] and not material['failures'] and
                 effective['coverage_pass'] and effective['startup_ranges_pass'] and
                 effective['collision_count'] == effective['mapped_collision_count'] and
                 material['terrain_collision_bindings'] and
                 all(row['binding_status']=='bound' and row['attribute_error'] is None
                     for row in material['terrain_collision_bindings']) and
                 approval['material_sha256'] == sha(output/'live-material.json'),
                 'live material audit not complete')
        checks['live_material'] = 'pass'
        validate_four_follower_readiness(publication['readiness'])
        checks['four_follower_readiness'] = 'pass'

        command = json.loads((batch/'g3-command.json').read_text())['argv']
        remote = command[command.index('--output')+1].rsplit('/output', 1)[0]
        with tempfile.TemporaryDirectory(prefix='eatnav-p1-g3-audit-shadow-') as temp:
            shadow = Path(temp)/batch.name
            shutil.copytree(batch, shadow)
            validation = validate_g3_output(shadow, remote, contract, frozen)
        evidence['post_path_validation'] = validation
        for name in ('exact_stamp_clock_tf_command_bridge_physics',
                     'same_episode_reset_termination',
                     'arrival_censor_hold_trace_completeness', 'host_container_sha'):
            checks[name] = 'pass'

        status = json.loads((batch/'g3-host-status.json').read_text())
        cleanup = json.loads((batch/'cleanup-evidence.json').read_text())
        _require(status['native_exit'] == cleanup['native_exit'] == 0 and
                 status['monitor_failure'] is None and
                 status['owned_gpu_bracket_count'] >= 1 and
                 status['elapsed_s'] <= 600 and
                 not status['owned_simulator_residual'] and
                 cleanup['owned_residual'] == cleanup['bridge_residual'] == [] and
                 all(cleanup['writer_state'].values()), 'native/monitor/cleanup failure')
        resource_path = batch/'resource-series.jsonl'
        samples = [json.loads(line) for line in resource_path.read_text().splitlines()]
        _require(bool(samples) and all(validate_sample(row) is None for row in samples),
                 'resource cap violation or attribution gap')
        evidence['resource'] = {'samples':len(samples),
            'gpu_device_total_peak_mib':max(row['gpu_device_total_mib'] for row in samples),
            'container_anon_shmem_peak_bytes':max(row['container_anon_plus_shmem_bytes'] for row in samples),
            'persistent_root_peak_bytes':max(row['persistent_root_bytes'] for row in samples),
            'host_elapsed_s':status['elapsed_s'], 'native_exit':status['native_exit'],
            'owned_residual':cleanup['owned_residual'], 'bridge_residual':cleanup['bridge_residual']}
        checks['resource_caps_native_exit_cleanup'] = 'pass'
        _require(verify_seal(batch)['stage'] == evidence['original_period2_stage_preserved'],
                 'original period-2 seal changed during audit')
        verify_prior_pair(root, manifest['prior_periods'])
    except (FileNotFoundError, KeyError, OSError) as exc:
        evidence['first_blocker'] = type(exc).__name__+':'+str(exc)
        checks['uncompleted_required_gate'] = 'missing'
    except Exception as exc:
        evidence['first_blocker'] = type(exc).__name__+':'+str(exc)
        checks['uncompleted_required_gate'] = 'fail'
    evidence['checks'] = checks
    evidence['decision'] = classify_checks(checks)
    evidence['period2_original_host_seal_sha256'] = sha(batch/'host-seal.json')
    evidence['period2_original_host_failure_unchanged'] = FAILURE
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--criteria-sha256', required=True)
    args = parser.parse_args()
    result = audit(args.source_root, args.audit_root, args.criteria_sha256)
    write_new(args.audit_root/'audit-result.json', result)
    print(result['decision'], result.get('first_blocker', 'all frozen gates verified'))


if __name__ == '__main__':
    main()
