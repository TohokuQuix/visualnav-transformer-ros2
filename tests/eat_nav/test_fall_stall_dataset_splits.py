import copy
import importlib
import importlib.util

import numpy as np
import pytest


def api():
    name='visualnav_transformer.eat_nav.fall_stall_dataset_splits'
    assert importlib.util.find_spec(name) is not None,'seed split freezing is missing'
    return importlib.import_module(name)


def cohort():
    index=[]
    for replicate,seed in enumerate((101,102,103)):
        for env in range(4):
            index.append({'sample_index':len(index),'sample_id':f'terrain-level1-rep{replicate}/env{env}',
                          'seed':seed,'env':env,'terrain':'terrain','level':1,'replicate':replicate})
    targets=np.zeros((12,4,4,4),dtype=np.int8)
    targets[4,0,3,0]=1
    masks=np.ones_like(targets,dtype=bool)
    targets[5,0,3,0]=-1;masks[5,0,3,0]=False
    return index,{'event_targets':targets,'event_mask':masks,'candidate_mask':np.ones((12,4),dtype=bool)}


def test_rotation_keeps_seed_families_and_flags_positive_free_heads():
    index,arrays=cohort();split=api().freeze_splits(index,arrays)
    assert split['formal_unseen_test'] is False
    assert len(split['folds'])==3
    fold=split['folds'][0]
    assert fold['roles']['exploratory_holdout']['base_seeds']==[101]
    assert fold['roles']['validation']['base_seeds']==[102]
    assert fold['roles']['train']['base_seeds']==[103]
    support=fold['roles']['validation']['event_support_10s']['tip_over']
    assert {k:support[k] for k in ('positive','negative','unknown')}=={'positive':1,'negative':14,'unknown':1}
    assert fold['roles']['train']['event_support_10s']['tip_over']['status']=='unsupported'
    assert fold['roles']['validation']['event_support_10s']['tip_over']['status']=='supported'
    assert fold['roles']['validation']['event_support_10s']['tip_over']['positive_base_seed_count']==1
    api().validate_split(index,arrays,split)


def test_changed_outcomes_do_not_change_seed_assignments():
    index,arrays=cohort();first=api().freeze_splits(index,arrays)
    arrays['event_targets'][:]=1;arrays['event_mask'][:]=True
    second=api().freeze_splits(index,arrays)
    for a,b in zip(first['folds'],second['folds']):
        for role in ('train','validation','exploratory_holdout'):
            assert a['roles'][role]['sample_indices']==b['roles'][role]['sample_indices']


def test_split_rejects_seed_leakage_and_stale_support():
    index,arrays=cohort();split=api().freeze_splits(index,arrays)
    broken=copy.deepcopy(split)
    broken['folds'][0]['roles']['train']['sample_indices'][0]=4
    with pytest.raises(ValueError):api().validate_split(index,arrays,broken)
    broken=copy.deepcopy(split)
    broken['folds'][0]['roles']['validation']['event_support_10s']['tip_over']['positive']=99
    with pytest.raises(ValueError):api().validate_split(index,arrays,broken)


@pytest.mark.parametrize('mutation',['duplicate_sample','seed_wrong_replicate','missing_env','missing_replicate'])
def test_incomplete_or_ambiguous_seed_families_are_rejected(mutation):
    index,arrays=cohort()
    if mutation=='duplicate_sample':index[1]['sample_id']=index[0]['sample_id']
    elif mutation=='seed_wrong_replicate':index[4]['seed']=101
    elif mutation=='missing_env':index[1]['env']=0
    else:
        index=index[:-4]
        arrays={key:value[:-4] for key,value in arrays.items()}
    with pytest.raises(ValueError):api().freeze_splits(index,arrays)
