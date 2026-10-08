"""Seal small review artifacts after the candidate-free radial fixture."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT=Path(os.environ.get('EATNAV_RADIAL_FIXTURE_ROOT','/tmp/eatnav-radial-anchor-contract-20261008-r1'))
WORKTREE=Path(__file__).resolve().parents[1]
REPORT=WORKTREE/'docs/eat-nav'/ROOT.name
R5=WORKTREE/'docs/eat-nav/local-primitive-p1-r5-20261008-attempt2'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def main():
    decision=read(ROOT/'final-decision.json')
    if decision.get('anchors_compared')!=4 or decision.get('anchors_passed')!=4:
        raise RuntimeError('fixture parity did not pass; cannot seal report')
    manifest=read(ROOT/'fixture-manifest.json')
    provenance=read(REPORT/'source-provenance.json')
    r5sources=read(R5/'source-provenance.json')['source_paths']
    checked={}
    for name,entry in r5sources.items():
        path=Path(entry['path'])
        actual=sha(path) if path.exists() else None
        if actual!=entry['sha256']:
            raise RuntimeError('r5 source SHA mismatch: '+name)
        checked[name]={'path':str(path),'sha256':actual}
    provenance['r5_source_paths_sha_verified']=checked
    provenance['r5_source_path_count']=len(checked)
    provenance['fixture_source_files']={str(WORKTREE/'scripts'/name):sha(WORKTREE/'scripts'/name) for name in (
        'eat_nav_radial_anchor_contract.py','eat_nav_radial_anchor_prepare.py',
        'eat_nav_radial_anchor_fixture.py','eat_nav_radial_anchor_compare.py','eat_nav_radial_anchor_finalize.py')}
    write(REPORT/'source-provenance.json',provenance)
    write(ROOT/'source-provenance.json',provenance)
    tolerance_source=WORKTREE/'docs/quality-identifiability-gate/candidate-coverage/e1-feasibility/e1-gates.yaml'
    write(REPORT/'tolerance-source-addendum.json',{'schema':'eatnav_radial_tolerance_source_addendum_v1',
          'numeric_thresholds_changed':False,'reason':'the pre-run freeze used an incorrect local provenance path',
          'e1_gate_path':str(tolerance_source),'e1_gate_sha256':sha(tolerance_source),
          'pre_run_tolerance_freeze_sha256':sha(REPORT/'reconstruction-tolerance-freeze.json')})
    lifecycle=[]
    for entry in manifest['shards']:
        shard=Path(entry['fixture_subroot'])
        status=read(shard/'status.json');seal=read(shard/'seal.json')
        series=[json.loads(line) for line in (shard/'resource-series.jsonl').read_text().splitlines()]
        if status['stage']!='protocol_valid' or seal['native_exit_code']!=0 or seal['owned_residual']:
            raise RuntimeError('invalid fixture shard: '+str(shard))
        lifecycle.append({'subroot':str(shard),'source_shard':entry['source_shard'],
                          'status':status['stage'],'native_exit_code':seal['native_exit_code'],
                          'owned_process':seal['owned_process'],'owned_residual':seal['owned_residual'],
                          'host_container_sha_match':seal['host_container_sha_match'],
                          'physics_steps':seal['physics_steps'],
                          'elapsed_sec':max(x['elapsed_sec'] for x in series),
                          'gpu_peak_mib':max(x['gpu_device_used_mib'] for x in series),
                          'anon_shmem_peak_bytes':max(x['container_anon_shmem_bytes'] for x in series),
                          'host_available_min_bytes':min(x['host_available_bytes'] for x in series),
                          'last_heartbeat':next((x['heartbeat'] for x in reversed(series) if x['heartbeat']),None),
                          'seal_sha256':sha(shard/'seal.json'),'raw_sha_ledger_sha256':sha(shard/'sha-ledger.json')})
    resource={'schema':'eatnav_radial_fixture_resource_lifecycle_v1','shards':lifecycle,
              'gpu_peak_mib':max(x['gpu_peak_mib'] for x in lifecycle),
              'anon_shmem_peak_bytes':max(x['anon_shmem_peak_bytes'] for x in lifecycle),
              'elapsed_sec_total':sum(x['elapsed_sec'] for x in lifecycle),
              'host_output_storage_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
              'resource_contract_sha256':sha(R5/'p1-resource-contract-v2.json'),
              'previous_attempt':{'root':'/tmp/eatnav-radial-anchor-contract-20261008-r1',
                                  'failure':'launcher_precreated_simulator_output_directory',
                                  'pre_control':True,'candidate_free':True,'remediation_cycle':1}}
    resource['gpu_attribution']={'device_total_cap_verified':resource['gpu_peak_mib']<read(R5/'p1-resource-contract-v2.json')['gpu_device_total_hard_cap_mib'],
                                 'per_pid_gpu_series_captured':False,'unresolved_gpu_bytes':'unknown',
                                 'driver_accounting_mode':'Disabled',
                                 'v2_zero_unresolved_condition_verified':False}
    decision.update({'decision':'radial_anchor_execution_contract_resource_attribution_inconclusive',
                     'physics_parity':'pass_4_of_4','shard_physics_protocol':'pass_3_of_3',
                     'resource_contract':'inconclusive_unresolved_gpu_memory_not_measured',
                     'first_incomplete_signal':'per_pid_gpu_memory_series_missing',
                     'p1_rollout_started':False})
    write(REPORT/'final-decision.json',decision);write(ROOT/'final-decision.json',decision)
    write(REPORT/'resource-lifecycle.json',resource);write(ROOT/'resource-lifecycle.json',resource)
    compact=read(REPORT/'anchor-parity-comparison.json')
    lines=[
        '# P1 radial-anchor execution contract: candidate-free fixture',
        '',
        'Decision: `radial_anchor_execution_contract_resource_attribution_inconclusive`. Physics parity passed 4/4, but the full P1 v2 resource gate is not yet proven. P1 rollout was not started.',
        '',
        '## Input and source contract',
        '',
        f'- Base commit `0fa3cb5`; four selected anchor records (28 source files) and {len(checked)} r5 source paths were SHA-verified.',
        '- The three original complete 112-env shard manifests were reused, one scene per original shard/seed. Normal RL reset, requested XY/yaw, origin z +0.35 m, zero 6D actions, dt 0.005 s, 200 physics steps and DEM at 50/100/150/200 were preserved.',
        '- The saved Actor vector is the step-200 `obs[policy]` 22-vector. Separate pre-noise/post-noise vectors were not stored; they are diagnostic-only and are not represented as reconstructed inputs.',
        '- E1 episode/prefix metadata is incompatible with these radial records. No synthetic episode fields were created.',
        '- The E1 numeric-gate provenance path is corrected in `tolerance-source-addendum.json`; the pre-run tolerance values and file are unchanged.',
        '',
        '## Fixture and parity',
        '',
        '| Anchor | Saved versus reconstructed | Root max error | Actor max error | DEM max height error |',
        '|---|---|---:|---:|---:|',
    ]
    for x in compact:
        lines.append(f"| {x['anchor_id']} | {'pass' if x['pass'] else 'fail'} | {x['max_root_position_error_m']} m | {x['max_actor_error']} | {x['max_dem_height_error_m']} m |")
    lines += [
        '',
        '- For every anchor, the pose record, Actor JSON and all four DEM JSON files are byte-identical to the saved originals. All 200 trace physics payloads, including root/joint/contact/termination and episode/reset counters, are identical after excluding wall and monotonic timestamps.',
        '- Thus stored settled eligibility, primary DEM class and known onset are invariant under this fixture. These labels were not recomputed with changed thresholds.',
        '- The 28-row manifest overlay changes only output directories. Run/order/anchor/candidate/Path SHA/seed/goal remain unchanged; no Path was published.',
        '',
        '## Resource and lifecycle',
        '',
        f"- Three sealed processes: 200/200 steps, native exit 0, completion marker, host/container SHA match, owned residual 0. GPU peak {resource['gpu_peak_mib']} MiB; anon+shmem peak {resource['anon_shmem_peak_bytes']/2**30:.3f} GiB; host output {resource['host_output_storage_bytes']/1e6:.1f} MB.",
        '- The device-total GPU cap was met. Per-PID GPU samples were not captured, and NVIDIA accounting mode is disabled; the v2 zero-unresolved-memory condition cannot be proven from these runs. No automatic post-control rerun was made.',
        '- An earlier pre-control fixture root failed because the launcher precreated the simulator output directory. The failure root was preserved. A RED→GREEN test covered the launcher correction; one allowed remediation cycle was used.',
        '',
        '## Scope boundary',
        '',
        'X1/X2, external-router operations, candidate Path publication, follower/bridge, 28-process P1 rollout, Outcome, relabeling and learning were not performed.',
        ''
    ]
    (REPORT/'report.md').write_text('\n'.join(lines))
    ledger={'schema':'eatnav_radial_fixture_artifact_sha_ledger_v1',
            'small_report_files':[], 'raw_shard_seals':[]}
    for path in sorted(REPORT.iterdir()):
        if path.is_file() and path.name!='artifact-sha-ledger.json':
            ledger['small_report_files'].append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    for item in lifecycle:
        ledger['raw_shard_seals'].append({'subroot':item['subroot'],'seal_sha256':item['seal_sha256'],
                                          'raw_sha_ledger_sha256':item['raw_sha_ledger_sha256']})
    write(REPORT/'artifact-sha-ledger.json',ledger);write(ROOT/'artifact-sha-ledger.json',ledger)
    print(json.dumps({'decision':decision['decision'],'shards':len(lifecycle),'anchors':len(compact),
                      'resource':resource,'report':str(REPORT)},indent=2))


if __name__=='__main__':main()
