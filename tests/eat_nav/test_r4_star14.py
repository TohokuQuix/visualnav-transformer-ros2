import numpy as np

def test_fourteen_paths_follow_existing_bank_and_six_is_exact_subset():
    from eat_nav_r4_star14 import candidate_paths,CANDIDATE_IDS,OLD_SLOTS,EVAL_SLOTS
    from visualnav_transformer.eat_nav.bidirectional_bank import bank
    from visualnav_transformer.eat_nav.motion_14_stage_a import candidate_role
    from eat_nav_r4_star import candidate_paths as six
    b=bank();assert len(b)==14
    assert CANDIDATE_IDS==tuple(candidate_role(c.direction,c.arc_angle_deg) for c in b)
    np.testing.assert_array_equal(candidate_paths(),np.stack([c.waypoints for c in b]).astype(np.float32))
    np.testing.assert_array_equal(candidate_paths()[list(EVAL_SLOTS)],six())
    assert OLD_SLOTS==(0,1,2,7)

def test_old_four_padding_maps_reverse_to_seven_and_never_invents_targets():
    from test_fall_stall_dataset_tensors import group
    from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups
    from eat_nav_r4_star14_learning import pad_four,CANDIDATE_ARRAY_KEYS
    from eat_nav_r4_star14 import OLD_SLOTS,ADDED_SLOTS
    old=pack_groups([group()]);new=pad_four(old)
    for k in old:
        if k in CANDIDATE_ARRAY_KEYS:
            np.testing.assert_array_equal(new[k][:,list(OLD_SLOTS)],old[k])
            assert (new[k][:,list(ADDED_SLOTS)]==(-1 if k in ('event_targets','event_free_10s') else 0)).all()
        else:np.testing.assert_array_equal(new[k],old[k])

def test_legacy_prefix_reverse_maps_to_slot_seven():
    from eat_nav_r4_star14 import geometry_choice,candidate_paths
    from eat_nav_all_fields import geometry_choice as old
    mapping=[0,1,2,7]
    for goal in ([10,0],[-10,0],[1,-4],[1,4]):assert geometry_choice(goal)==mapping[old(goal)]
    assert geometry_choice([-10,0])==7

def test_added_jobs_cover_missing_ten_on_original_splits():
    from eat_nav_r4_star14_run import jobs
    from eat_nav_r4_star14 import ADDED_SLOTS
    c,e=jobs();assert len(c)==1120 and len(e)==36
    assert {j['candidate'] for j in c}==set(ADDED_SLOTS)
    assert {j['role'] for j in c}=={'train','validation'}

def test_curved_target_merge_and_training_restore_with_fourteen(tmp_path):
    from eat_nav_r4_star14_learning import pad_four,merge_slot,fit_field
    from test_fall_stall_dataset_tensors import group,teacher
    from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups
    from eat_nav_r4_star14 import ADDED_SLOTS,CANDIDATE_IDS,candidate_paths,DIRECTIONS
    from visualnav_transformer.eat_nav.fall_stall_audit import verify_derived
    from eat_nav_command_free_learning import load_checkpoint
    g=group();arrays=pad_four(pack_groups([g,g]))
    for i in (0,1):
        for k in ADDED_SLOTS:
            merge_slot(arrays,i,k,dict(initial=g['initial'],candidates={CANDIDATE_IDS[k]:dict(path=candidate_paths()[k],direction=int(DIRECTIONS[k]),targets=teacher(i))}))
    assert arrays['paths'].shape==(2,14,8,2) and arrays['candidate_mask'].all()
    path=fit_field(arrays,[dict(seed=42,role='train'),dict(seed=43,role='validation')],{'train':[0],'validation':[1]},123,tmp_path/'model',epochs=1)
    verify_derived(path.parent);model,payload=load_checkpoint(path)
    assert payload['config']['learning_rate']==.001

def test_circle_geometry_and_reverse_extension_follow_original_convention():
    from visualnav_transformer.eat_nav.bidirectional_bank import bank
    from eat_nav_r4_star14 import candidate_paths,extend_path,ANGLES
    from eat_nav_all_fields import rotation
    for k,c in enumerate(bank()):
        if c.arc_angle_deg:
            r=.75/np.deg2rad(c.arc_angle_deg);travel=c.waypoints*c.direction
            np.testing.assert_allclose(travel[:,0]**2+(travel[:,1]-r)**2,r*r,rtol=0,atol=1e-14)
        points=candidate_paths()[k].astype(float);q,h=extend_path(points,ANGLES[k],k)
        np.testing.assert_allclose(q[8:],(points[-1]+points@rotation(ANGLES[k]).T)[1:])
        assert h==2*ANGLES[k]

def test_collection_adapter_only_changes_bank_and_preserves_prefix_mapping():
    from pathlib import Path
    old=Path('scripts/eat_nav_all_fields_sim.py').read_text()
    expected=old.replace('choices=range(4)','choices=range(14)').replace('from eat_nav_all_fields import *','from eat_nav_all_fields import *\n# Preserve the original prefix: old reverse slot3 maps to canonical slot7.\nfrom eat_nav_r4_star14 import candidate_paths,pack_inputs,CANDIDATE_IDS,ANGLES,extend_path,geometry_choice')
    assert Path('scripts/eat_nav_r4_star14_collect_sim.py').read_text()==expected

def test_pairing_uses_old_slot_mapping_and_failed_branch_stays_unknown(tmp_path):
    from eat_nav_r4_star14 import candidate_paths,CANDIDATE_IDS,OLD_SLOTS,ADDED_SLOTS,pack_inputs
    from eat_nav_r4_star14_learning import field_dataset,npz,pad_four,CANDIDATE_ARRAY_KEYS
    from test_fall_stall_dataset_tensors import group
    from test_fall_stall_targets import trace
    from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups
    from visualnav_transformer.eat_nav.fall_stall_audit import write,seal_derived,read
    import json
    baseline=tmp_path/'baseline';dest=baseline/'datasets/k_rail_field';dest.mkdir(parents=True)
    g=group()
    for j,k in enumerate(OLD_SLOTS):g['candidates'][list(g['candidates'])[j]]['path']=candidate_paths()[k]
    old=pack_groups([g]);source=tmp_path/'old-branch';source.mkdir()
    initial={k:old[k] for k in ('dem','actor_state','body_state','paths','directions','candidate_mask')}
    np.savez_compressed(source/'initial-input.npz',**initial);np.savez_compressed(source/'anchor-state.npz',root=np.ones(13))
    write(source/'anchor.json',{'root':g['initial']['root']})
    np.savez_compressed(dest/'samples.npz',**old);write(dest/'sample-index.json',[dict(seed=42,anchor=0,role='train',sources=[str(source)])]);write(dest/'split.json',{'train':[0],'validation':[]});seal_derived(dest)
    root=tmp_path/'new';root.mkdir();write(root/'plan.json',{'collect':[dict(field='k_rail_field',seed=42,anchor=0,candidate=k) for k in ADDED_SLOTS]})
    entries=[dict(index=j,state='completed' if j==0 else 'failed',result={'stop_reason':'horizon_complete'}) for j in range(10)]
    write(root/'collection-progress.json',{'results':entries})
    for j,k in enumerate(ADDED_SLOTS[:2]):
        out=root/f'collection-runtime/collect-{j:03d}/sim-output';out.mkdir(parents=True)
        write(out/'complete.json',{'candidate':k,'seed':42,'anchor_available':True})
        np.savez_compressed(out/'anchor-state.npz',root=np.ones(13));np.savez_compressed(out/'initial-input.npz',**pack_inputs(old['dem'][0],old['actor_state'][0],old['body_state'][0]))
        (out/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace()));seal_derived(out)
    assert field_dataset(root,'k_rail_field',baseline)==(None,None,None)
    arrays=npz(root/'datasets/k_rail_field/samples.npz')
    for key in old:np.testing.assert_array_equal(arrays[key][:,list(OLD_SLOTS)] if key in CANDIDATE_ARRAY_KEYS else arrays[key],old[key])
    assert arrays['candidate_mask'][0,ADDED_SLOTS[0]]
    assert not arrays['candidate_mask'][0,ADDED_SLOTS[1]]
    statuses=read(root/'datasets/k_rail_field/anchor-audit.json')
    assert statuses[0]['status']=='included' and statuses[1]['status']=='failed_run'

def test_report_does_not_claim_na_models_learned_all_candidates(tmp_path):
    from eat_nav_r4_star14_run import correct_report
    p=tmp_path/'comparison-report.md'
    p.write_text('追加後退曲線は現行4候補datasetで直接学習していない。同じweightsによる候補拡張であり追加学習の効果ではない。')
    correct_report(tmp_path)
    text=p.read_text()
    assert '学習成功モデル' in text and 'N/A' in text

def test_runner_updates_existing_status_without_stopping(tmp_path,monkeypatch):
    import eat_nav_r4_star14_run as runner
    from visualnav_transformer.eat_nav.fall_stall_audit import write,read
    import sys
    monkeypatch.setattr(runner,'QUEUE',tmp_path)
    monkeypatch.setattr(runner,'verify_sources',lambda plan:None)
    monkeypatch.setattr(runner,'verify_predecessor',lambda:None)
    monkeypatch.setattr(runner,'run',lambda root,plan:None)
    monkeypatch.setattr(sys,'argv',['runner'])
    write(tmp_path/'plan.json',{})
    write(tmp_path/'status.json',{'state':'waiting'})
    runner.main()
    assert read(tmp_path/'status.json')['state']=='completed'

def test_comparison_report_can_update_after_each_case(tmp_path):
    from eat_nav_r4_star_audit import report
    from visualnav_transformer.eat_nav.fall_stall_audit import write
    baseline=tmp_path/'old';baseline.mkdir()
    write(baseline/'suite-results.json',[])
    write(tmp_path/'manifest.json',{'fields':[],'runs':[{},{}]})
    report(tmp_path,[],baseline)
    report(tmp_path,[],baseline)
