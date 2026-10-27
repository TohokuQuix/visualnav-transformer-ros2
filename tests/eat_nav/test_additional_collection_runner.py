"""Ordered dispatch, fail-stop resume and blind logs across the simulator boundary."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import pytest
from test_additional_blind_verification import additional_fixture,source
from visualnav_transformer.eat_nav.fall_stall_audit import read


def api():
    name='visualnav_transformer.eat_nav.fall_stall_collection_runner'
    assert importlib.util.find_spec(name) is not None,'additional collection runner missing'
    return importlib.import_module(name)


@pytest.fixture
def runner_fixture(additional_fixture,tmp_path):
    from visualnav_transformer.eat_nav.fall_stall_collection_contract import load_freeze
    root,c=additional_fixture;frozen=load_freeze('docs/eat-nav/fall-stall-additional-collection-freeze-20261020/artifacts')
    frozen['sources']['pilot_critical_sha256']=c['sources']['pilot_critical_sha256']
    frozen['rows']=c['rows'][:2];frozen['manifest']['protocol_sha256']=c['manifest']['protocol_sha256']
    frozen['manifest']['output_root']=str(tmp_path/'collection');frozen['manifest']['counts']['total_processes']=2
    return root,frozen,Path(frozen['manifest']['output_root'])


def binding(c):return api().create_binding(c,{'container_image':'test-image','cuda':True,'test_runtime_fixture':True})


def test_dispatch_and_verified_resume_do_not_rerun_completed_rows(runner_fixture,capsys):
    fixture,c,output=runner_fixture;calls=[]
    def execute(target,**kwargs):
        calls.append((target,kwargs));print('PRIVATE_tip_over_89_progress_12.34')
        original=fixture/target.name;shutil.copytree(original,target)
    result=api().run_collection(c,output,binding(c),executor=execute)
    assert result['state']=='completed' and result['completed_processes']==2
    assert len(calls)==2 and calls[0][1]['prepath_only'] is True and calls[1][1]['prepath_only'] is False
    assert calls[1][1]['reference']==output/c['rows'][1]['reference_relative']
    assert all(k['control_steps']==200 and k['replan_every_steps']==40 for _,k in calls)
    assert 'PRIVATE' not in capsys.readouterr().out
    api().run_collection(c,output,read(output/'execution-binding.json'),executor=execute,resume=True)
    assert len(calls)==2


def test_failed_row_is_retained_and_never_automatically_retried(runner_fixture,capsys):
    fixture,c,output=runner_fixture;calls=[]
    def execute(target,**kwargs):
        calls.append(target);target.mkdir(parents=True)
        print('PRIVATE_tip_over_89_progress_12.34')
        raise RuntimeError('PRIVATE_tip_over_89_progress_12.34')
    with pytest.raises(api().BlindVerificationError) as error:
        api().run_collection(c,output,binding(c),executor=execute)
    assert error.value.code=='EXECUTION_FAILED' and len(calls)==1
    assert (output/c['rows'][0]['output_relative']).exists()
    assert read(output/'status.json')['state']=='stopped'
    assert 'PRIVATE' not in (output/'status.json').read_text()+capsys.readouterr().out
    with pytest.raises(api().BlindVerificationError):
        api().run_collection(c,output,read(output/'execution-binding.json'),executor=execute,resume=True)
    assert len(calls)==1


def test_complete_attempt_without_receipt_is_adopted_without_reexecution(runner_fixture):
    fixture,c,output=runner_fixture;calls=[]
    def execute(target,**kwargs):calls.append(target);shutil.copytree(fixture/target.name,target)
    bound=binding(c)
    api().initialize_collection(c,output,bound)
    first=output/c['rows'][0]['output_relative'];shutil.copytree(fixture/first.name,first)
    api().run_collection(c,output,bound,executor=execute,resume=True)
    assert len(calls)==1 and calls[0].name==c['rows'][1]['output_relative']


def test_changed_binding_and_unrequested_root_are_rejected(runner_fixture):
    fixture,c,output=runner_fixture;bound=binding(c)
    with pytest.raises(api().BlindVerificationError):api().run_collection(c,output.parent/'wrong',bound,executor=lambda *a,**k:None)
    api().initialize_collection(c,output,bound);changed=copy.deepcopy(bound);changed['runtime']['container_image']='changed'
    with pytest.raises(api().BlindVerificationError):api().run_collection(c,output,changed,executor=lambda *a,**k:None,resume=True)


def test_reserved_test_teacher_export_is_refused_before_any_trace_read(runner_fixture,tmp_path):
    name='visualnav_transformer.eat_nav.fall_stall_additional_targets'
    assert importlib.util.find_spec(name) is not None,'role-aware teacher adapter missing'
    adapter=importlib.import_module(name)
    _,c,output=runner_fixture
    with pytest.raises(api().BlindVerificationError) as error:adapter.build_development_dataset(output,c,tmp_path/'derived',roles=('reserved_test',))
    assert error.value.code=='RESERVED_TEST_DENIED' and not (tmp_path/'derived').exists()


def test_development_adapter_builds_v2_initial_inputs_without_reserved_data(additional_fixture,tmp_path):
    from visualnav_transformer.eat_nav.fall_stall_collection_contract import load_freeze
    from visualnav_transformer.eat_nav.bidirectional_bank import candidate
    from test_fall_stall_audit import put,reseal_attempt
    import numpy as np
    fixture,c=additional_fixture
    contract=load_freeze('docs/eat-nav/fall-stall-additional-collection-freeze-20261020/artifacts')
    contract['sources']['pilot_critical_sha256']=c['sources']['pilot_critical_sha256']
    contract['manifest']['protocol_sha256']=c['manifest']['protocol_sha256'];contract['rows']=c['rows']
    contract['manifest']['output_root']=str(tmp_path/'development-collection')
    for row in contract['rows']:
        row['role']='train'
        if row['stage']=='candidate':
            attempt=fixture/row['output_relative'];sim=attempt/'sim-output';path=candidate(row['direction'],row['angle']).waypoints.tolist()
            info=read(sim/'input-manifest.json');info['path_body_xy_m']=path;put(sim/'input-manifest.json',info)
            activation=read(sim/'path-activation.json');activation['paths_tile_xy_m']=[path]*4;put(sim/'path-activation.json',activation);reseal_attempt(attempt)
    output=Path(contract['manifest']['output_root'])
    def execute(target,**kwargs):shutil.copytree(fixture/target.name,target)
    api().run_collection(contract,output,binding(contract),executor=execute)
    adapter=importlib.import_module('visualnav_transformer.eat_nav.fall_stall_additional_targets')
    derived=tmp_path/'development-dataset'
    result=adapter.build_development_dataset(output,contract,derived,roles=('train',))
    assert result=={'state_groups':4,'candidate_outcomes':16,'reserved_test_included':False}
    with np.load(derived/'samples.npz',allow_pickle=False) as data:
        assert data['dem'].shape==(4,4,2,20,20)
        assert data['paths'].shape==(4,4,8,2)
        assert data['event_mask'][:,:,3,:].all()
        assert (data['event_targets'][:,:,3,2]==1).all()
    assert read(derived/'split.json')['roles']=={'train':[0,1,2,3]}


@pytest.mark.parametrize('mutation',['freeze_snapshot','execution_snapshot','saved_assignment','receipt_symlink'])
def test_resume_rejects_changed_run_metadata_before_dispatch(runner_fixture,tmp_path,mutation):
    from visualnav_transformer.eat_nav.fall_stall_audit import seal_derived
    fixture,c,output=runner_fixture;bound=binding(c);api().initialize_collection(c,output,bound);calls=[]
    if mutation=='freeze_snapshot':
        f=output/'freeze';(f/'assignment.json').write_text('{}\n')
        (f/'sha-ledger.json').unlink();(f/'derived-seal.json').unlink();seal_derived(f)
    elif mutation=='execution_snapshot':(output/'sources/scripts/eat_nav_additional_collection.py').write_text('changed\n')
    elif mutation=='saved_assignment':(output/'assignment.json').write_text('{"rows": []}\n')
    else:
        (output/'receipts').rmdir();outside=tmp_path/'outside';outside.mkdir();(output/'receipts').symlink_to(outside,target_is_directory=True)
    def execute(target,**kwargs):calls.append(target);shutil.copytree(fixture/target.name,target)
    with pytest.raises(api().BlindVerificationError):api().run_collection(c,output,bound,executor=execute,resume=True)
    assert calls==[]


def test_resume_rejects_future_attempt_before_launching_missing_prefix(runner_fixture):
    fixture,c,output=runner_fixture;bound=binding(c);api().initialize_collection(c,output,bound);calls=[]
    future=output/c['rows'][1]['output_relative'];shutil.copytree(fixture/future.name,future)
    def execute(target,**kwargs):calls.append(target);shutil.copytree(fixture/target.name,target)
    with pytest.raises(api().BlindVerificationError):api().run_collection(c,output,bound,executor=execute,resume=True)
    assert calls==[]
