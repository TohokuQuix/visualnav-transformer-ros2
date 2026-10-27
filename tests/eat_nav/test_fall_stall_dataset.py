"""End-to-end dataset export using real sealed source and teacher fixtures."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_fall_stall_audit import source, put, reseal_attempt
from visualnav_transformer.eat_nav.fall_stall_audit import audit_pilot, seal_derived


def api():
    name='visualnav_transformer.eat_nav.fall_stall_dataset'
    assert importlib.util.find_spec(name) is not None,'dataset source/teacher join is missing'
    return importlib.import_module(name)


@pytest.fixture
def paired_cohort(source,tmp_path):
    root=tmp_path/'paired-source';root.mkdir()
    original=json.loads((source/'pilot-manifest.json').read_text())
    original_rows=json.loads((source/'assignment.json').read_text())['rows']
    groups=[];rows=[]
    for replicate in range(3):
        group={**original['state_groups'][0],'replicate':replicate,'terrain_seed':123+replicate,
               'state_group_id':f'k_rail_step_field-level1-rep{replicate}'}
        groups.append(group)
        for i,old in enumerate(original_rows):
            row={**old,**group};rows.append(row)
            old_attempt=next(source.glob(f'{i:03d}-*'))
            attempt=root/f'{len(rows)-1:03d}-{group["state_group_id"]}-{row["candidate_id"] or "prepath"}'
            shutil.copytree(old_attempt,attempt)
            for relative in ('manifest.json','sim-output/input-manifest.json','sim-output/prepath-identity.json'):
                path=attempt/relative;value=json.loads(path.read_text());value['seed']=group['terrain_seed'];put(path,value)
            reseal_attempt(attempt)
    put(root/'pilot-manifest.json',{**original,'state_groups':groups,'process_count':15,
                                  'candidate_process_count':12,'candidate_trajectory_count':48})
    put(root/'assignment.json',{'rows':rows})
    put(root/'status.json',{'state':'completed','completed_processes':15,'total_processes':15})
    teachers=tmp_path/'teachers';audit_pilot(root,teachers)
    return root,teachers


def reseal_teacher(root):
    (root/'sha-ledger.json').unlink();(root/'derived-seal.json').unlink();seal_derived(root)


def test_full_export_and_portable_reload_keep_pairs_masks_and_rotations(paired_cohort,tmp_path):
    source,teachers=paired_cohort;output=tmp_path/'dataset'
    summary=api().build_dataset(source,teachers,output)
    arrays,index,split=api().load_dataset(output)
    assert summary['sample_count']==12 and summary['trajectory_count']==48
    assert arrays['dem'].shape==(12,4,2,20,20)
    assert arrays['paths'].shape==(12,4,8,2)
    assert arrays['event_targets'].shape==(12,4,4,4)
    assert arrays['candidate_mask'].all()
    assert len(index)==12 and all(len(row['candidates'])==4 for row in index)
    assert split['folds'][0]['roles']['train']['base_seeds']==[125]
    assert split['folds'][1]['roles']['train']['base_seeds']==[123]
    assert split['folds'][2]['roles']['train']['base_seeds']==[124]
    # Loading the sealed dataset needs neither the simulator tree nor the teacher tree.
    source.rename(tmp_path/'source-moved');teachers.rename(tmp_path/'teachers-moved')
    reloaded,_,_=api().load_dataset(output)
    for name in arrays:np.testing.assert_array_equal(reloaded[name],arrays[name])


def test_teacher_line_order_does_not_change_pairing(paired_cohort,tmp_path):
    source,teachers=paired_cohort
    first=tmp_path/'first';api().build_dataset(source,teachers,first)
    path=teachers/'targets.jsonl';lines=path.read_text().splitlines();path.write_text('\n'.join(reversed(lines))+'\n')
    reseal_teacher(teachers)
    second=tmp_path/'second';api().build_dataset(source,teachers,second)
    left,_,_=api().load_dataset(first);right,_,_=api().load_dataset(second)
    for name in left:np.testing.assert_array_equal(left[name],right[name])


@pytest.mark.parametrize('mutation',['duplicate','missing','wrong_identity','changed_target'])
def test_resealed_teacher_inconsistency_is_rejected_before_publication(paired_cohort,tmp_path,mutation):
    source,teachers=paired_cohort;path=teachers/'targets.jsonl'
    lines=path.read_text().splitlines()
    if mutation=='duplicate':lines.append(lines[0])
    elif mutation=='missing':lines=lines[1:]
    else:
        record=json.loads(lines[0])
        if mutation=='wrong_identity':record['seed']=999
        else:record['targets']['horizons']['10s']['events']['tip_over']['label']=1
        lines[0]=json.dumps(record)
    path.write_text('\n'.join(lines)+'\n');reseal_teacher(teachers)
    with pytest.raises(ValueError,match='teacher|identity|target|duplicate|coverage'):
        api().build_dataset(source,teachers,tmp_path/'dataset')
    assert not (tmp_path/'dataset').exists()


def test_source_mutation_after_teacher_freeze_is_rejected(paired_cohort,tmp_path):
    source,teachers=paired_cohort
    trace=next(source.glob('001-*/sim-output/trace-env-0.jsonl'))
    trace.write_text(trace.read_text()+'\n')
    with pytest.raises(ValueError,match='source|SHA|seal|ledger'):
        api().build_dataset(source,teachers,tmp_path/'dataset')


def test_output_is_fresh_and_outside_both_input_trees(paired_cohort,tmp_path):
    source,teachers=paired_cohort
    for output in (source/'derived',teachers/'derived'):
        with pytest.raises(ValueError,match='outside|source|teacher'):
            api().build_dataset(source,teachers,output)
    with pytest.raises(FileExistsError):api().build_dataset(source,teachers,tmp_path)


def test_portable_loader_rejects_mutation_even_if_seal_is_rebuilt(paired_cohort,tmp_path):
    source,teachers=paired_cohort;output=tmp_path/'dataset';api().build_dataset(source,teachers,output)
    path=output/'split.json';split=json.loads(path.read_text())
    split['folds'][0]['roles']['train']['sample_indices'][0]=0
    put(path,split);reseal_teacher(output)
    with pytest.raises(ValueError,match='split|coverage|seed'):
        api().load_dataset(output)


def test_dataset_npz_and_metadata_are_reproducible(paired_cohort,tmp_path):
    source,teachers=paired_cohort;first=tmp_path/'first';second=tmp_path/'second'
    api().build_dataset(source,teachers,first);api().build_dataset(source,teachers,second)
    for file in first.iterdir():
        if file.is_file():assert file.read_bytes()==(second/file.name).read_bytes(),file.name
