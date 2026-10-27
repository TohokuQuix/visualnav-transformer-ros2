import numpy as np
import pytest

def test_padding_does_not_create_new_labels_or_inputs():
    from eat_nav_r4_star_learning import pad_four
    old={'dem':np.zeros((1,4,2,20,20),np.float32),'paths':np.ones((1,4,8,2),np.float32),
         'directions':np.ones((1,4,1),np.float32),'candidate_mask':np.ones((1,4),bool),
         'event_targets':np.ones((1,4,4,4),np.int8),'event_mask':np.ones((1,4,4,4),bool),
         'progress_targets':np.ones((1,4,4),np.float32),'progress_mask':np.ones((1,4,4),bool),
         'event_free_10s':np.ones((1,4),np.int8),'event_free_mask':np.ones((1,4),bool)}
    six=pad_four(old)
    for k in old:
        if k=='dem':np.testing.assert_array_equal(six[k],old[k]);continue
        np.testing.assert_array_equal(six[k][:,:4],old[k])
        assert six[k].shape[1]==6
        assert (six[k][:,4:]==(-1 if k in ('event_targets','event_free_10s') else 0)).all()

def test_six_tensor_pack_accepts_reverse_curved_teacher():
    from eat_nav_r4_star_tensors import pack_groups,validate_arrays
    from eat_nav_r4_star import candidate_paths
    from visualnav_transformer.eat_nav.fall_stall_targets import build_targets
    dem=np.zeros((4,2,20,20),np.float32);dem[:,1]=1
    root=[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    group=dict(initial=dict(dem=dem,actor_pre_noise=np.zeros(22),root=root),candidates={
        'reverse_-30':dict(path=candidate_paths()[4],direction=-1,targets=build_targets(__import__('test_fall_stall_targets').trace(count=1),candidate_paths()[4].tolist()))})
    arrays=pack_groups([group]);validate_arrays(arrays)
    assert arrays['paths'].shape==(1,6,8,2)
    assert arrays['candidate_mask'].tolist()==[[False,False,False,False,True,False]]
    assert not arrays['event_mask'].any()
    assert arrays['directions'][0,4,0]==-1

def test_new_manifest_only_collects_two_new_candidates_on_same_split():
    from eat_nav_r4_star_learning_run import jobs
    from eat_nav_r4_run import BASE
    from visualnav_transformer.eat_nav.fall_stall_audit import read
    collection,evaluation=jobs();old=read(BASE/'manifest.json')
    assert len(collection)==224 and len(evaluation)==36
    for c in collection:
        assert c['candidate'] in (4,5)
        assert c['role'] in ('train','validation')
        assert any(all(r[k]==c[k] for k in ('field','seed','role','anchor')) for r in old['collect'])

def test_collection_adapter_only_extends_candidates():
    from pathlib import Path
    old=Path('scripts/eat_nav_all_fields_sim.py').read_text()
    expected=old.replace('choices=range(4)','choices=range(6)').replace('from eat_nav_all_fields import *','from eat_nav_all_fields import *\n# Keep legacy geometry_choice for the identical four-candidate prefix.\nfrom eat_nav_r4_star import candidate_paths,pack_inputs,CANDIDATE_IDS,ANGLES,extend_path')
    assert Path('scripts/eat_nav_r4_star_collect_sim.py').read_text()==expected

def test_merge_slot_uses_its_own_trace_targets_without_changing_original_four():
    from eat_nav_r4_star_learning import pad_four,merge_slot,CANDIDATE_ARRAY_KEYS
    from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups as pack_old
    from test_fall_stall_dataset_tensors import group
    from test_fall_stall_targets import trace
    from visualnav_transformer.eat_nav.fall_stall_targets import build_targets
    from eat_nav_r4_star import candidate_paths
    g=group();old=pack_old([g]);new=pad_four(old)
    rows=trace(vx=-.3,wz=0,body_wz=0)
    targets=build_targets(rows,candidate_paths()[4].tolist())
    added=dict(initial=g['initial'],candidates={'reverse_-30':dict(path=candidate_paths()[4],direction=-1,targets=targets)})
    merge_slot(new,0,4,added)
    for k in old:np.testing.assert_array_equal(new[k][:,:4] if k in CANDIDATE_ARRAY_KEYS else new[k],old[k])
    assert new['candidate_mask'][0,4] and not new['candidate_mask'][0,5]
    assert new['event_mask'][0,4,3,2] and new['event_targets'][0,4,3,2]==1
    assert (new['event_targets'][0,5]==-1).all()

def test_training_smoke_six_candidates_uses_same_config_and_restores(tmp_path):
    from eat_nav_r4_star_learning import fit_field,pad_four,merge_slot
    from eat_nav_current_learning import load_current_dataset
    from test_fall_stall_dataset_tensors import group,teacher
    from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups
    from eat_nav_r4_star import candidate_paths
    from visualnav_transformer.eat_nav.fall_stall_audit import read,verify_derived
    from eat_nav_command_free_learning import load_checkpoint
    g=group();arrays=pad_four(pack_groups([g,g]))
    for i in range(2):
        for slot,name in [(4,'reverse_-30'),(5,'reverse_+30')]:
            add=dict(initial=g['initial'],candidates={name:dict(path=candidate_paths()[slot],direction=-1,targets=teacher(i))})
            merge_slot(arrays,i,slot,add)
    path=fit_field(arrays,[dict(seed=42,role='train'),dict(seed=43,role='validation')],{'train':[0],'validation':[1]},123,tmp_path/'model',epochs=1)
    verify_derived(path.parent);model,payload=load_checkpoint(path)
    assert payload['config']['learning_rate']==.001
    assert payload['config']['source_weights']==[2/3,1/3]
    assert read(path.parent/'training.json')['history'][0]['epoch']==1

@pytest.mark.parametrize('failed_ledger',[False,True])
def test_dataset_keeps_old_data_on_failed_branch_and_excludes_mismatched_anchor(tmp_path,failed_ledger):
    from eat_nav_r4_star_learning import field_dataset,pad_four,npz
    from test_fall_stall_dataset_tensors import group
    from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups
    from visualnav_transformer.eat_nav.fall_stall_audit import write,seal_derived,read
    from eat_nav_r4_star import pack_inputs
    from test_fall_stall_targets import trace
    import json
    baseline=tmp_path/'baseline';dataset=baseline/'datasets'/'k_rail_field';dataset.mkdir(parents=True)
    g=group()
    from eat_nav_r4_star import candidate_paths,CANDIDATE_IDS
    for j,name in enumerate(CANDIDATE_IDS[:4]):g['candidates'][name]['path']=candidate_paths()[j]
    old=pack_groups([g]);source=tmp_path/'original';source.mkdir()
    inputs={k:old[k] for k in ('dem','actor_state','body_state','paths','directions','candidate_mask')}
    np.savez_compressed(source/'initial-input.npz',**inputs);np.savez_compressed(source/'anchor-state.npz',root=np.ones(13))
    write(source/'anchor.json',{'root':g['initial']['root']})
    np.savez_compressed(dataset/'samples.npz',**old)
    write(dataset/'sample-index.json',[dict(seed=42,anchor=0,role='train',sources=[str(source)])]);write(dataset/'split.json',{'train':[0],'validation':[]});seal_derived(dataset)
    root=tmp_path/'new';root.mkdir()
    write(root/'plan.json',{'collect':[dict(field='k_rail_field',seed=42,anchor=0,candidate=k) for k in (4,5)]})
    write(root/'collection-progress.json',{'results':[{'index':0,'state':'failed' if failed_ledger else 'completed','result':{'stop_reason':'horizon_complete'}},{'index':1,'state':'failed'}]})
    out=root/'collection-runtime/collect-000/sim-output';out.mkdir(parents=True)
    write(out/'complete.json',{'anchor_available':True,'candidate':4,'seed':42})
    six=pack_inputs(old['dem'][0],old['actor_state'][0],old['body_state'][0]);np.savez_compressed(out/'initial-input.npz',**six)
    np.savez_compressed(out/'anchor-state.npz',root=np.ones(13)*(1 if failed_ledger else 2))
    (out/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace(vx=-.3)));seal_derived(out)
    assert field_dataset(root,'k_rail_field',baseline)==(None,None,None)
    actual=npz(root/'datasets/k_rail_field/samples.npz')
    for k,v in pad_four(old).items():np.testing.assert_array_equal(actual[k],v)
    statuses=[r['status'] for r in read(root/'datasets/k_rail_field/anchor-audit.json')]
    assert statuses==(['failed_run','failed_run'] if failed_ledger else ['anchor_mismatch','failed_run'])
