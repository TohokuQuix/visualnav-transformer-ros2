"""Adapter tests use real small sealed artifacts, not mocked verification."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.long_horizon_labels import label_trace
from test_fall_stall_targets import trace


def module():
    name='visualnav_transformer.eat_nav.fall_stall_audit'
    assert importlib.util.find_spec(name) is not None, 'v2 differential audit implementation is missing'
    return importlib.import_module(name)


def put(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root,outer=False):
    ledger='outer-sha-ledger.json' if outer else 'sha-ledger.json'
    seal_name='outer-seal.json' if outer else 'host-seal.json'
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*')
           if p.is_file() and p.name not in (ledger,seal_name)}
    put(root/ledger,files)
    put(root/seal_name,{'ledger_sha256':sha(root/ledger),'file_count':len(files)})


@pytest.fixture
def source(tmp_path):
    import numpy as np
    root=tmp_path/'source';root.mkdir()
    repo=Path(__file__).resolve().parents[2]
    frozen={key:sha(repo/path) for key,path in {
        'labels':'src/visualnav_transformer/eat_nav/long_horizon_labels.py',
        'host_runner':'scripts/eat_nav_motion_14_collection_host.py',
        'sim_runner':'scripts/eat_nav_motion_14_collection_sim.py',
        'follower':'src/visualnav_transformer/eat_nav/p1_follower_core.cpp',
        'config':'config/eat_nav.yaml'}.items()}
    frozen.update(actor='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d',
                  actor_env='14fc2441d928a92b1ba45c06ccd8603eb0cf71aebb5d216efc4005f09696545c',
                  protocol='1df876cc86e749dbff56b9b81ab1fd10e2ad69790f576bbda08db0c82844d9a7')
    bank=[{'candidate_id':name,'direction':direction,'angle':angle} for name,direction,angle in
          [('forward_direct',1,0),('forward_-30',1,-30),('forward_+30',1,30),('reverse_direct',-1,0)]]
    group={'state_group_id':'k_rail_step_field-level1-rep0','terrain_type':'k_rail_step_field',
           'terrain_level':1,'replicate':0,'terrain_seed':123,'env_tile_count':4}
    assignment=[{**group,'stage':'prepath','candidate_id':None,'direction':1,'angle':0,'failure_induction':False}]
    assignment += [{**group,'stage':'candidate',**c,'failure_induction':False} for c in bank]
    put(root/'pilot-manifest.json',{'schema':'eatnav_passive_paired_long_horizon_pilot_v0.2',
        'source_sha256':frozen,'state_groups':[group],'candidates':bank,'condition':'front_folded','repeat':0,
        'control_hz':20,'control_steps':200,'replan_every_steps':40,'num_envs_per_process':4,
        'process_count':5,'candidate_process_count':4,'candidate_trajectory_count':16,'failure_induction':False})
    put(root/'assignment.json',{'rows':assignment})
    put(root/'status.json',{'state':'completed','completed_processes':5,'total_processes':5})
    baseline=None
    for i,a in enumerate(assignment):
        attempt=root/f'{i:03d}-{a["state_group_id"]}-{a["candidate_id"] or "prepath"}'
        sim=attempt/'sim-output';sim.mkdir(parents=True)
        roots=np.array([trace(1)[0]['root_state']]*4)
        dem=np.zeros((4,4,2,20,20),dtype=np.float32);dem[:,:,1]=1
        np.savez(sim/'prepath.npz',root=roots,root_tile_local=roots,dem=dem,
                 actor_pre_noise=np.zeros((4,22)),actor_input=np.zeros((4,22)),actor_post_noise=np.zeros((4,22)))
        grid=np.stack(np.meshgrid(np.arange(20)*.1-.95,np.arange(20)*.1-.95),axis=-1).reshape(400,2)
        starts=np.zeros((4,4,400,3));starts[:,:,:,:2]=grid;starts[:,:,:,2]=1.
        hits=starts.copy();hits[:,:,:,2]=0.
        stamps=[250_000_000,500_000_000,750_000_000,1_000_000_000]
        np.savez(sim/'raw-dem.npz',ray_starts_w=starts,ray_hits_w=hits,ray_valid=np.ones((4,4,400),dtype=bool),
                 root_state_w=np.array([roots]*4),env_origins_w=np.zeros((4,4,3)),logical_stamp_ns=np.array(stamps))
        put(sim/'prepath-identity.json',{'seed':123,'dem_frame':'robot_base','dem_resolution_m':.1,'dem_stamps_ns':stamps})
        put(sim/'raw-dem-metadata.json',{'logical_stamp_ns':stamps,'source':'height_scanner_eat native world-frame rays'})
        path=[[.75*a['direction']*point/7,0.] for point in range(8)]
        mode='prepath_only' if a['stage']=='prepath' else 'candidate'
        manifest={'seed':123,'terrain_type':group['terrain_type'],'terrain_level':1,'condition':'front_folded',
                  'repeat':0,'direction':a['direction'],'angle_deg':a['angle'],'mode':mode,'control_steps':200,
                  'replan_every_steps':40,'actor_env_sha256':frozen['actor_env'],'rl_checkpoint_sha256':frozen['actor'],
                  'protocol_sha256':frozen['protocol'],'source_sha256':frozen['sim_runner'],
                  'host_source_sha256':frozen['host_runner'],'eat_config_sha256':frozen['config'],
                  'reference_prepath_sha256':baseline}
        put(attempt/'manifest.json',manifest)
        put(sim/'input-manifest.json',{**manifest,'arc_angle_deg':a['angle'],
                                     'checkpoint_sha256':frozen['actor'],'core_source_sha256':frozen['follower'],
                                     'path_body_xy_m':path})
        put(sim/'prepath-pair-gate.json',{'pass':True})
        complete={'status':'complete','mode':mode,'env_count':4,'control_steps':0,
                  'trace_sha256':[],'prepath_sha256':sha(sim/'prepath.npz'),'raw_dem_sha256':sha(sim/'raw-dem.npz')}
        if mode=='candidate':
            rows=trace(200)
            for row in rows:
                row['follower_diagnostic']={'commanded_distance':.6,'actual_distance':0.,
                                            'remaining_path_distance':.75}
            for env in range(4):
                (sim/f'trace-env-{env}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            put(sim/'long-horizon-event-labels.json',{'trials':[label_trace(rows)]*4})
            put(sim/'path-activation.json',{'activation_stamp_ns':1_000_000_000,'paths_tile_xy_m':[path]*4,
                                          'direction':a['direction'],'arc_angle_deg':a['angle']})
            complete.update(control_steps=200,requested_control_steps=200,replan_every_steps=40,
                            trace_lengths=[200]*4,terminated_envs=[],censored_envs=[],
                            trace_sha256=[sha(sim/f'trace-env-{env}.jsonl') for env in range(4)])
        else:baseline=sha(sim/'prepath.npz')
        put(sim/'complete.json',complete)
        seal(sim)
        put(attempt/'decision.json',{'pass':True,'failure':None})
        put(attempt/'verification.json',{'simulator_seal_sha256':sha(sim/'host-seal.json')})
        seal(attempt,outer=True)
    return root


def test_rejects_source_containment_and_existing_output(source,tmp_path):
    with pytest.raises(ValueError,match='source|outside'):
        module().audit_pilot(source,source/'derived')
    with pytest.raises(FileExistsError):
        module().audit_pilot(source,tmp_path)


def test_sealed_fixture_produces_v2_masks_diffs_and_derived_seal(source,tmp_path):
    output=tmp_path/'derived'
    result=module().audit_pilot(source,output)
    assert result['trajectory_count']==16
    assert result['processes_verified']==5
    assert result['event_counts_v2']['linear_stall']==80
    assert result['horizon_support']['1s']['tip_over']=={'positive':0,'negative':16,'unknown':0}
    assert (output/'targets.jsonl').is_file()
    assert (output/'label-diff.json').is_file()
    module().verify_derived(output)


def test_mutated_trace_is_rejected_by_source_seal(source,tmp_path):
    path=next(source.glob('001-*/sim-output/trace-env-0.jsonl'))
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='seal|SHA|ledger'):
        module().audit_pilot(source,tmp_path/'derived')
    assert not (tmp_path/'derived').exists()


def test_resealed_wrong_seed_is_rejected_by_assignment(source,tmp_path):
    attempt=next(source.glob('001-*'))
    manifest=json.loads((attempt/'manifest.json').read_text());manifest['seed']=999
    put(attempt/'manifest.json',manifest);seal(attempt,outer=True)
    with pytest.raises(ValueError,match='identity|seed|assignment'):
        module().audit_pilot(source,tmp_path/'derived')


def test_derived_seal_detects_modified_teacher(source,tmp_path):
    output=tmp_path/'derived';module().audit_pilot(source,output)
    with (output/'targets.jsonl').open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='seal|SHA|ledger'):
        module().verify_derived(output)


def test_diff_reports_confirmation_shift_without_calling_it_new_event():
    rows=trace()
    for i in range(17,22):
        from test_fall_stall_targets import state
        rows[i]['post_root_state']=state(tilt=65.)
    for row in rows:row['follower_diagnostic']={}
    legacy=label_trace(rows)
    v2=importlib.import_module('visualnav_transformer.eat_nav.fall_stall_targets').build_targets(rows,[[0.,0.],[.75,0.]])
    diff=module().compare_labels(legacy,v2)
    shift=next(d for d in diff['event_changes'] if d['event']=='near_fall')
    assert shift['kind']=='confirmation_time_shift'
    assert shift['v1_steps']==[17] and shift['v2_steps']==[21]
    changed=next(d for d in diff['horizon_changes'] if d['event']=='near_fall' and d['horizon_sec']==1)
    assert changed['v1_label']==1 and changed['v2_label']==0


def test_cycle_audit_explains_coarse_best_false_positive():
    rows=trace(40)
    rows[-1]['post_root_state'][0]=.06
    for row in rows:
        row['follower_diagnostic']={'commanded_distance':.6,'actual_distance':.06,
                                    'remaining_path_distance':.75,'best_remaining_path_distance':.75}
    rows[-1]['follower_diagnostic']['remaining_path_distance']=.69
    v2=importlib.import_module('visualnav_transformer.eat_nav.fall_stall_targets').build_targets(rows,[[0.,0.],[.75,0.]])
    evidence=module().compare_cycle_measurements(rows,v2)[0]
    assert evidence['diagnostic_best_rule_label']==1
    assert evidence['v2']['linear_stall']['label']==0
    assert evidence['v2']['best_progress_m']==pytest.approx(.06)


def test_figures_are_covered_by_derived_seal(source,tmp_path):
    pytest.importorskip('matplotlib')
    output=tmp_path/'derived'
    module().audit_pilot(source,output,make_figures=True)
    files=module().verify_derived(output)
    assert any(name.startswith('figures/') and name.endswith('.png') for name in files)


def reseal_attempt(attempt):
    sim=attempt/'sim-output'
    complete=json.loads((sim/'complete.json').read_text())
    complete['prepath_sha256']=sha(sim/'prepath.npz');complete['raw_dem_sha256']=sha(sim/'raw-dem.npz')
    put(sim/'complete.json',complete);seal(sim)
    put(attempt/'verification.json',{'simulator_seal_sha256':sha(sim/'host-seal.json')})
    seal(attempt,outer=True)


@pytest.mark.parametrize('field',['dem','actor_pre_noise'])
def test_resealed_missing_initial_tensor_is_rejected(source,tmp_path,field):
    import numpy as np
    attempt=next(source.glob('000-*'));path=attempt/'sim-output/prepath.npz'
    with np.load(path) as arrays:values={k:arrays[k] for k in arrays.files if k!=field}
    np.savez(path,**values);reseal_attempt(attempt)
    with pytest.raises(ValueError,match='missing required prepath tensor'):
        module().audit_pilot(source,tmp_path/'derived')


def test_future_dem_timestamp_is_rejected_even_if_resealed(source,tmp_path):
    attempt=next(source.glob('000-*'));path=attempt/'sim-output/prepath-identity.json'
    value=json.loads(path.read_text());value['dem_stamps_ns'][-1]=2_000_000_000
    put(path,value);reseal_attempt(attempt)
    with pytest.raises(ValueError,match='stamp|DEM|initial'):
        module().audit_pilot(source,tmp_path/'derived')


def test_frozen_spec_hashes_are_anchored_not_only_self_consistent(source,tmp_path):
    path=source/'pilot-manifest.json';pilot=json.loads(path.read_text())
    pilot['source_sha256']['actor']='different-actor';put(path,pilot)
    for attempt in sorted(source.glob('[0-9][0-9][0-9]-*')):
        for relative,key in [('manifest.json','rl_checkpoint_sha256'),('sim-output/input-manifest.json','checkpoint_sha256')]:
            path=attempt/relative;value=json.loads(path.read_text());value[key]='different-actor';put(path,value)
        reseal_attempt(attempt)
    with pytest.raises(ValueError,match='frozen|actor'):
        module().audit_pilot(source,tmp_path/'derived')


def test_translated_activation_path_is_rejected_even_when_resealed(source,tmp_path):
    attempt=next(source.glob('001-*'));path=attempt/'sim-output/path-activation.json'
    value=json.loads(path.read_text())
    value['paths_tile_xy_m']=[[[x+100,y] for x,y in points] for points in value['paths_tile_xy_m']]
    put(path,value);reseal_attempt(attempt)
    with pytest.raises(ValueError,match='Path|activation|origin'):
        module().audit_pilot(source,tmp_path/'derived')


def test_symlink_attempt_cannot_hide_source_inside_output_parent(source,tmp_path):
    attempt=next(source.glob('001-*'));external=tmp_path/'external-attempt'
    attempt.rename(external);attempt.symlink_to(external,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink|source|containment'):
        module().audit_pilot(source,external/'derived')
    assert not (external/'derived').exists()


@pytest.mark.parametrize('mutation',['nonbinary_mask','nonfinite_actor','raw_height','raw_axes'])
def test_initial_tensor_and_frame_content_is_recomputed_after_resealing(source,tmp_path,mutation):
    import numpy as np
    attempt=next(source.glob('000-*'))
    filename='raw-dem.npz' if mutation.startswith('raw_') else 'prepath.npz'
    path=attempt/'sim-output'/filename
    with np.load(path) as arrays:values={k:arrays[k] for k in arrays.files}
    if mutation=='nonbinary_mask':values['dem'][0,0,1,0,0]=.5
    elif mutation=='nonfinite_actor':values['actor_pre_noise'][0,0]=float('nan')
    elif mutation=='raw_height':values['ray_hits_w'][0,0,210,2]=.1
    else:values['ray_starts_w'][0,0,:,0]+=1.
    np.savez(path,**values);reseal_attempt(attempt)
    with pytest.raises(ValueError,match='DEM|initial|raw'):
        module().audit_pilot(source,tmp_path/'derived')


def test_resealed_short_path_is_rejected_by_frozen_input_shape(source,tmp_path):
    attempt=next(source.glob('001-*'));sim=attempt/'sim-output'
    path=sim/'input-manifest.json';value=json.loads(path.read_text())
    value['path_body_xy_m']=[value['path_body_xy_m'][0],value['path_body_xy_m'][-1]];put(path,value)
    path=sim/'path-activation.json';value=json.loads(path.read_text())
    value['paths_tile_xy_m']=[[points[0],points[-1]] for points in value['paths_tile_xy_m']];put(path,value)
    reseal_attempt(attempt)
    with pytest.raises(ValueError,match='Path|shape'):
        module().audit_pilot(source,tmp_path/'derived')
