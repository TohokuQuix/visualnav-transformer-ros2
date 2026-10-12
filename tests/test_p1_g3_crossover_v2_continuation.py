"""Cross-attempt provenance and period-2 staging tests; no simulator."""

import hashlib
import json
from pathlib import Path

import pytest

from eat_nav_p1_g1_lifecycle import seal_after_writers
from eat_nav_p1_g3_crossover_v2_continuation import (
    verify_prior_reference, verify_prior_pair, stage_continuation)
from eat_nav_p1_g3_crossover_v2_host import validate_period_assignment
from eat_nav_p1_g3_crossover_v2_host import approve_prepath
from eat_nav_p1_g3_crossover_v2_continue_outer import preflight


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def prior(tmp_path, period, roles):
    root=tmp_path/f'old-period-{period}'
    rows=[{'slot':i,'candidate_slot':f'crossover_{i}','condition':'A',
           'path_role':role,'path_sha256':role} for i,role in enumerate(roles)]
    put(root/'output/prepath-crossover-snapshot.json',{'envs':[{'actor_input':[0]}]})
    (root/'host-runner.py').write_text('host')
    (root/'source.py').write_text('sim')
    role_sets=(['direct','direct','left_20','right_20'],
               ['left_20','right_20','direct','direct'],
               ['right_20','left_20','right_20','left_20'])
    processes=[]
    for index,role_set in enumerate(role_sets):
        trial_rows=[{'slot':i,'candidate_slot':f'crossover_{i}','condition':'A',
                     'path_role':role,'path_sha256':role}
                    for i,role in enumerate(role_set)]
        processes.append({'anchor_id':'first-differential','batch_index':0,
                          'execution_index':index,'trials':trial_rows})
    put(root/'input/g3-assignment.json',{'processes':processes})
    put(root/'input/g3-contract.json',{'assignment_sha256':sha(root/'input/g3-assignment.json'),
                                      'g3_simulator_source_sha256':sha(root/'source.py'),
                                      'g3_host_source_sha256':sha(root/'host-runner.py'),
                                      'processes':processes})
    put(root/'g3-host-status.json',{'failure':None})
    seal_after_writers(root,writer_state={'simulator':True,'heartbeat':True,
        'resource':True,'log':True},stage='G3_crossover_batch_protocol_valid')
    spec={'period':period,'root':str(root.resolve()),'manifest_sha256':sha(root/'input/g3-contract.json'),
          'host_source_sha256':sha(root/'host-runner.py'),
          'simulator_source_sha256':sha(root/'source.py'),
          'seal_sha256':sha(root/'host-seal.json'),
          'snapshot_sha256':sha(root/'output/prepath-crossover-snapshot.json'),
          'path_roles':roles}
    return root,spec


def test_prior_reference_checks_all_sha_and_roles(tmp_path):
    root,spec=prior(tmp_path,0,['direct','direct','left_20','right_20'])
    verify_prior_reference(root,spec)
    with pytest.raises(ValueError,match='snapshot'):
        verify_prior_reference(root,{**spec,'snapshot_sha256':'wrong'})
    with pytest.raises(ValueError,match='host_source'):
        verify_prior_reference(root,{**spec,'host_source_sha256':'wrong'})
    with pytest.raises(ValueError,match='path_roles'):
        verify_prior_reference(root,{**spec,'path_roles':['left_20']*4})


def test_pair_aliases_require_exact_readonly_target_and_seal(tmp_path):
    first,a=prior(tmp_path,0,['direct','direct','left_20','right_20'])
    second,b=prior(tmp_path,1,['left_20','right_20','direct','direct'])
    aliases=tmp_path/'aliases';aliases.mkdir()
    (aliases/'batch-first-differential-0-period-0').symlink_to(first,target_is_directory=True)
    (aliases/'batch-first-differential-0-period-1').symlink_to(second,target_is_directory=True)
    verify_prior_pair(aliases,[a,b])
    (aliases/'batch-first-differential-0-period-1').unlink()
    (aliases/'batch-first-differential-0-period-1').symlink_to(first,target_is_directory=True)
    with pytest.raises(ValueError,match='target'):
        verify_prior_pair(aliases,[a,b])


def test_period_roles_are_frozen_per_period():
    roles=(['direct','direct','left_20','right_20'],
           ['left_20','right_20','direct','direct'],
           ['right_20','left_20','right_20','left_20'])
    for expected in roles:
        rows=[{'slot':i,'candidate_slot':f'crossover_{i}','condition':'A',
               'path_role':role,'path_sha256':role} for i,role in enumerate(expected)]
        validate_period_assignment(rows,rows)
        wrong=json.loads(json.dumps(rows));wrong[0]['path_sha256']='wrong'
        with pytest.raises(ValueError,match='assignment'):
            validate_period_assignment(wrong,rows)


def test_stage_requires_fresh_root(tmp_path):
    with pytest.raises(FileExistsError):
        stage_continuation(tmp_path,tmp_path,{}, {}, {}, tmp_path)


def test_cross_attempt_prepath_gate_uses_sealed_period_zero(tmp_path):
    old=Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v2-pilot-attempt-2')
    for i in (0,1):
        (tmp_path/f'batch-first-differential-0-period-{i}').symlink_to(
            old/f'batch-first-differential-0-period-{i}',target_is_directory=True)
    batch=tmp_path/'batch-first-differential-0-period-2';out=batch/'output';out.mkdir(parents=True)
    source=old/'batch-first-differential-0-period-0/output'
    for name in ('prepath-crossover-snapshot.json','live-material.json',
                 'prepath-gate.json','direct-actor-gate.json'):
        (out/name).write_bytes((source/name).read_bytes())
    put(out/'prepath-ready.json',{'Path_publications':0,'physics_steps_after_clone':0,
        'snapshot_sha256':sha(out/'prepath-crossover-snapshot.json'),
        'live_material_sha256':sha(out/'live-material.json')})
    assert approve_prepath(batch,'first-differential',0,2)['pass']
    assert not (out/'path-publication.json').exists()
    (out/'prepath-approval.json').unlink()
    (out/'cross-period-prepath-gate.json').unlink()
    snapshot=json.loads((out/'prepath-crossover-snapshot.json').read_text())
    snapshot['envs'][0]['root_state_w'][0]+=0.001
    put(out/'prepath-crossover-snapshot.json',snapshot)
    ready=json.loads((out/'prepath-ready.json').read_text())
    ready['snapshot_sha256']=sha(out/'prepath-crossover-snapshot.json')
    put(out/'prepath-ready.json',ready)
    with pytest.raises(ValueError,match='cross-period'):
        approve_prepath(batch,'first-differential',0,2)
    assert not (out/'prepath-approval.json').exists()


def test_fresh_staging_and_preflight_reject_assignment_drift(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    old=Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v2-pilot-attempt-2')
    manifest=json.loads((repo/'docs/eat-nav/p1-g3-crossover-v2-continuation-manifest.json').read_text())
    contract=json.loads((repo/'docs/eat-nav/p1-g3-crossover-v2-continuation-contract.json').read_text())
    root=tmp_path/'g3-crossover-v2-period2-continuation-attempt-0'
    stage_continuation(repo,old,manifest['prior_periods'],contract,manifest,root)
    host=repo/'scripts/eat_nav_p1_g3_crossover_v2_host.py'
    assert preflight(root,host)['Path_publications']==0
    assert not (root/'batch-first-differential-0-period-2/output').exists()
    staged=root/'batch-first-differential-0-period-2/input/g3-assignment.json'
    data=json.loads(staged.read_text())
    data['processes'][2]['trials'][0]['path_role']='direct'
    put(staged,data)
    with pytest.raises(ValueError,match='ledger|assignment'):
        preflight(root,host)
