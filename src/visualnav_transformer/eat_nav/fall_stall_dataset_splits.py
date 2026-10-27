"""Freeze outcome-independent replicate rotations with complete seed families."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .fall_stall_dataset_tensors import check
from .fall_stall_targets import EVENT_NAMES


SCHEMA='eatnav_fall_stall_split_v1'
ROLES=('train','validation','exploratory_holdout')


def validate_index(index):
    check(bool(index),'empty sample index')
    check([row['sample_index'] for row in index]==list(range(len(index))),'sample index is not contiguous')
    check(len({row['sample_id'] for row in index})==len(index),'duplicate sample ID')
    families=defaultdict(list);cells=defaultdict(dict)
    for row in index:
        check(row['replicate'] in (0,1,2),'replicate outside frozen rotation')
        families[row['seed']].append(row)
    for seed,rows in families.items():
        check(len(rows)==4 and {r['env'] for r in rows}=={0,1,2,3},'incomplete or duplicate env in seed family')
        cell={(r['terrain'],r['level'],r['replicate']) for r in rows}
        check(len(cell)==1,'base seed crosses terrain/level/replicate')
        terrain,level,replicate=next(iter(cell))
        check(replicate not in cells[(terrain,level)],'duplicate cell replicate')
        cells[(terrain,level)][replicate]=seed
    check(all(set(replicates)=={0,1,2} for replicates in cells.values()),'missing replicate in terrain/level cell')


def freeze_splits(index,arrays):
    validate_index(index)
    count=len(index)
    check(arrays['event_targets'].shape==(count,4,4,4) and arrays['event_mask'].shape==(count,4,4,4)
          and arrays['candidate_mask'].shape==(count,4),'split tensor dimensions mismatch')
    folds=[]
    for fold in range(3):
        roles={}
        for role,replicate in (('train',(fold+2)%3),('validation',(fold+1)%3),('exploratory_holdout',fold)):
            rows=[r for r in index if r['replicate']==replicate]
            indices=[r['sample_index'] for r in rows]
            support={}
            for head,name in enumerate(EVENT_NAMES):
                labels=arrays['event_targets'][indices,:,3,head]
                known=arrays['event_mask'][indices,:,3,head]
                active=arrays['candidate_mask'][indices]
                positive=(labels==1)&known&active;negative=(labels==0)&known&active
                unknown=~known&active
                pos=int(positive.sum());neg=int(negative.sum())
                support[name]={'positive':pos,'negative':neg,'unknown':int(unknown.sum()),
                               'positive_state_group_count':int(positive.any(axis=1).sum()),
                               'negative_state_group_count':int(negative.any(axis=1).sum()),
                               'positive_base_seed_count':len({r['seed'] for r,yes in zip(rows,positive.any(axis=1)) if yes}),
                               'negative_base_seed_count':len({r['seed'] for r,yes in zip(rows,negative.any(axis=1)) if yes}),
                               'status':'supported' if pos and neg else 'unsupported',
                               'reason':'both_classes_present' if pos and neg else 'no_positive' if not pos else 'no_negative'}
            roles[role]={'base_seeds':sorted({r['seed'] for r in rows}),'sample_ids':[r['sample_id'] for r in rows],
                         'sample_indices':indices,'sample_count':len(rows),
                         'candidate_slot_count':len(rows)*4,
                         'trajectory_count':int(arrays['candidate_mask'][indices].sum()),
                         'event_support_10s':support}
        folds.append({'fold':fold,'roles':roles})
    return {'schema':SCHEMA,'rule':'replicate_rotation_without_outcome_reassignment',
            'group_key':'base_seed_all_envs_candidates_windows','formal_unseen_test':False,
            'status':'frozen_exploratory_split','folds':folds}


def validate_split(index,arrays,split):
    expected=freeze_splits(index,arrays)
    check(split==expected,'split rule/coverage/seed leakage/support mismatch')
