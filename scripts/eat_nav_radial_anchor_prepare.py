"""Read-only provenance gate and frozen inputs for the candidate-free P1 fixture."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R5 = ROOT / 'docs/eat-nav/local-primitive-p1-r5-20261008-attempt2'
OUTPUT = Path(os.environ.get('EATNAV_RADIAL_FIXTURE_ROOT', '/tmp/eatnav-radial-anchor-contract-20261008-r1'))
REPORT = ROOT / 'docs/eat-nav' / OUTPUT.name
_spec = importlib.util.spec_from_file_location('contract', HERE / 'eat_nav_radial_anchor_contract.py')
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    if REPORT.exists() or OUTPUT.exists():
        raise RuntimeError('fresh output root required')
    selected_path = R5 / 'selected-four-anchors-r3.json'
    manifest_path = R5 / 'p1-run-manifest-r3.json'
    paths_path = R5 / 'frozen-primitive-paths-r3.json'
    resource_path = R5 / 'p1-resource-contract-v2.json'
    selected = read(selected_path)['selected']
    manifest = read(manifest_path)
    resources = read(resource_path)
    assert len(selected) == 4 and len(manifest['rows']) == 28
    assert [a['anchor_id'] for a in selected] == [
        'first-differential', 'second-differential', 'first-graded', 'second-graded']
    provenance = {'schema':'eatnav_radial_anchor_source_provenance_v1', 'base_commit':'0fa3cb5b7828c78ad74221a5f7ecf057333a449d',
                  'r5_files':{str(p):contract.sha256(p) for p in (selected_path,manifest_path,paths_path,resource_path,
                                                                     R5/'x1-gate.json',R5/'x2-readiness-gate.json',R5/'x2-process-result.json')},
                  'anchors':[], 'source_shards':{}, 'source_records_sha_verified':True}
    contracts = []
    for anchor in selected:
        shard = Path(anchor['source_shard'])
        pose_id = anchor['pose_id']
        freeze = read(shard/'input-freeze.json')
        pose_manifest = read(shard/'pose-manifest.json')
        pose = next((x for x in pose_manifest['poses'] if x['pose_id']==pose_id), None)
        if pose is None or sum(x['pose_id']==pose_id for x in pose_manifest['poses']) != 1:
            raise RuntimeError('pose missing or duplicated: '+pose_id)
        record = read(shard/'run/poses'/f'{pose_id}.json')
        actor = read(shard/'run/actor'/pose_id/'settled_step_200.json')
        frames = [read(shard/'run/dem'/pose_id/f'frame-{i:02d}.json') for i in range(4)]
        actor_hash = contract.field_hash(actor['field_order'])
        contract.verify_radial_source(anchor,freeze,pose,actor,frames,record,actor_hash)
        for filename, expected in anchor['source_path_shas'].items():
            if contract.sha256(filename) != expected:
                raise RuntimeError('source record SHA mismatch: '+filename)
        for name in ('input-freeze.json','pose-manifest.json','prelaunch.json','seal.json','sha-ledger.json','run/sim-finished.json','run/terrain-mesh-identity.json'):
            if not (shard/name).exists():
                raise RuntimeError('critical source missing: '+str(shard/name))
        if contract.sha256(shard/'pose-manifest.json') != freeze['pose_manifest_sha256']:
            raise RuntimeError('source pose manifest SHA mismatch')
        provenance['source_shards'][str(shard)] = {'input_freeze_sha256':contract.sha256(shard/'input-freeze.json'),
                                                   'pose_manifest_sha256':contract.sha256(shard/'pose-manifest.json'),
                                                   'prelaunch_sha256':contract.sha256(shard/'prelaunch.json'),
                                                   'sim_source_sha256':freeze['sim_source_sha256'],
                                                   'terrain_mesh_sha256':freeze['terrain_mesh_sha256'],
                                                   'batch_seed':pose['batch_seed'],
                                                   'pose_order':[x['pose_id'] for x in pose_manifest['poses']]}
        provenance['anchors'].append({'anchor_id':anchor['anchor_id'],'pose_id':pose_id,'source_shard':str(shard),
                                      'record_sha256':anchor['source_path_shas'],'terrain':pose['terrain'],
                                      'terrain_seed':pose['terrain_seed'],'source_family':pose['source_family'],
                                      'env_index':pose['env_index'],'actor_field_order_hash':actor_hash})
        contracts.append({'anchor_id':anchor['anchor_id'],'pose_id':pose_id,'source_shard':str(shard),
                          'batch_seed':pose['batch_seed'],'terrain_seed':pose['terrain_seed'],
                          'terrain':pose['terrain'],'difficulty':'frozen_generator_0.5',
                          'source_family':pose['source_family'],'environment_index':pose['env_index'],
                          'terrain_row':pose['row'],'terrain_column':pose['column'],
                          'requested_xy_yaw':pose['requested_xy_yaw'],'root_z_rule':'scene.env_origins.z + 0.35 m',
                          'actor_field_order_hash':actor_hash,'actor_config_identity':actor['config_identity'],
                          'source_record_sha256':anchor['source_path_shas'],
                          'settled_phase':'post_step_200_pre_Path'})
    over,diff=contract.output_overlay(manifest,str(OUTPUT/'future-p1-attempt'))
    if any(set(x['changed']) != {'output_directory'} for x in diff):
        raise RuntimeError('overlay changed logical manifest')
    # These values are frozen before any new physics run.
    tol={'schema':'eatnav_radial_reconstruction_tolerance_v1',
         'source':'anchor_placement_parity.py strict DEM geometry and prior E1 replay root/Actor bounds',
         'dem_source_sha256':contract.sha256(ROOT/'src/visualnav_transformer/eat_nav/anchor_placement_parity.py'),
         'e1_gate_sha256':contract.sha256(ROOT/'docs/eat-nav/e1-feasibility/e1-gates.yaml') if (ROOT/'docs/eat-nav/e1-feasibility/e1-gates.yaml').exists() else None,
         'dem_height_rms_m':1e-5,'dem_height_max_abs_m':1e-4,'dem_mask_iou_min':1.0,
         'dem_origin_max_abs_m':1e-5,'dem_yaw_abs_rad':1e-5,
         'root_position_max_abs_m':0.02,'root_orientation_max_abs_rad':0.03,
         'root_velocity_max_abs_mps_or_radps':1e-4,'joint_position_max_abs_rad':1e-4,
         'joint_velocity_max_abs_radps':1e-4,'actor_max_abs':1e-4,
         'exact':['pose_id','environment_index','terrain_seed','terrain_type','source_family','DEM_frame_order',
                  'DEM_shape','DEM_frame','DEM_resolution','DEM_capture_step','joint_names','actor_field_order',
                  'termination_count','reset_counter','contact_bounds','eligibility']}
    # Original three shard results bound this forecast; 20% is applied to every physical resource.
    history={'first_shard_A':{'gpu_peak_mib':5853,'anon_shmem_peak_gib':3.649,'elapsed_sec':34.74,'storage_mb':7.6},
             'first_shard_B':{'gpu_peak_mib':5857,'anon_shmem_peak_gib':3.649,'elapsed_sec':34.88,'storage_mb':22.1},
             'second_shard_A':{'gpu_peak_mib':5861,'anon_shmem_peak_gib':3.658,'elapsed_sec':37.06,'storage_mb':57.0}}
    forecast={'schema':'eatnav_radial_fixture_resource_forecast_v1','source':'three original candidate-free shards',
              'historical':history,'margin_factor':1.2,
              'gpu_peak_mib':max(x['gpu_peak_mib'] for x in history.values())*1.2,
              'anon_shmem_peak_bytes':int(max(x['anon_shmem_peak_gib'] for x in history.values())*2**30*1.2),
              'runtime_sec':sum(x['elapsed_sec'] for x in history.values())*1.2,
              'storage_bytes':int(sum(x['storage_mb'] for x in history.values())*1e6*1.2),
              'caps':{k:resources[k] for k in ('gpu_device_total_hard_cap_mib','container_anon_plus_shmem_cap_bytes',
                                              'per_process_timeout_sec','heartbeat_stale_sec','storage_cap_bytes',
                                              'minimum_host_available_bytes')},
              'pass':True}
    forecast['pass']=(forecast['gpu_peak_mib']<resources['gpu_device_total_hard_cap_mib'] and
                      forecast['anon_shmem_peak_bytes']<resources['container_anon_plus_shmem_cap_bytes'] and
                      forecast['storage_bytes']<resources['storage_cap_bytes'] and
                      all(1.2*x['elapsed_sec']<resources['per_process_timeout_sec'] for x in history.values()))
    if not forecast['pass']:
        raise RuntimeError('resource forecast exceeds cap')
    table=[
        ('terrain batch seed / terrain seed / difficulty','saved source','pose manifest, input freeze, terrain generator'),
        ('terrain type / tile index / scene origin','saved source','pose manifest and run terrain mesh identity'),
        ('requested XY/yaw','saved source','pose manifest and prephysics pose'),
        ('root z rule','saved source','input freeze: env origin + 0.35 m'),
        ('root pose and velocity','saved source','reset and prephysics state; per-step trace'),
        ('joint position and velocity','saved source','reset and prephysics state; per-step trace'),
        ('flipper reset and fold mode','reconstructable','original RL reset event and fixed source; recorded mode'),
        ('reset event ordering','reconstructable','original frozen settling simulator source'),
        ('zero 6D action','saved source','pose record and input freeze'),
        ('dt / decimation / 200 steps','reconstructable','frozen RL config and simulator source'),
        ('termination / reset handling','saved source','termination-aware per-step trace'),
        ('DEM four frame order / unit / mask','saved source','frame records at physics 50,100,150,200'),
        ('Actor 22 field order and post-step vector','saved source','settled_step_200.json'),
        ('Actor pre-noise / post-noise separate vectors','diagnostic only','not separated in source record'),
        ('TF map/base transform','reconstructable','root pose and recorded map DEM; no TF published'),
        ('pre-Path anchor timestamp','saved source','step 200 logical stamp; no Path publisher')]
    input_contract={'schema':'eatnav_episode_vs_radial_input_contract_v1','critical_missing':[],
                    'e1':'g0e-train episode_key/step/local_step and 100-env action prefix required',
                    'radial':'normal RL reset plus deterministic candidate-free 200-step settling; no episode prefix',
                    'fields':[{'field':a,'status':b,'evidence':c} for a,b,c in table]}
    execution={'schema':contract.CONTRACT_VERSION,'name':'P1 radial-anchor execution','anchors':contracts,
               'scene_strategy':'three complete original 112-env shard manifests, separate original batch-seed scenes',
               'reset_protocol':'normal RL reset; requested XY/yaw only; root z origin + 0.35 m; original joint/flipper reset; zero velocities',
               'settling_protocol':'zero 6D action; dt 0.005 s; decimation 10; 20 control/200 physics steps',
               'dem_capture_physics_steps':[50,100,150,200],
               'path_publication_boundary':'after step-200 anchor; forbidden in fixture',
               'forbidden_synthetic_fields':sorted(contract.FORBIDDEN),
               'incompatible_with':'g0e-train episode-prefix replay contract',
               'termination':'any physical termination, reset contamination, buffer invalid/full, nonfinite or missing marker fails',
               'manifest_sha256':contract.sha256(manifest_path),
               'manifest_overlay_scope':'output_directory only; logical 28 rows and all Path SHA invariant'}
    fixture={'schema':'eatnav_radial_fixture_manifest_v1','output_root':str(OUTPUT),
             'candidate_free':True,'shards':[{'source_shard':s,'batch_seed':v['batch_seed'],
                                             'pose_order':v['pose_order'],
                                             'fixture_subroot':str(OUTPUT/f'shard-{i:02d}')}
                                            for i,(s,v) in enumerate(provenance['source_shards'].items())],
             'anchor_ids':[a['anchor_id'] for a in selected],
             'forbidden':['Path publish','follower','bridge','Zenoh control','candidate command','Outcome']}
    REPORT.mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    outputs={'source-provenance.json':provenance,'episode-vs-radial-input-contract.json':input_contract,
             'radial-anchor-execution-contract.json':execution,'manifest-output-overlay.json':over,
             'manifest-logical-identity-diff.json':diff,'reconstruction-tolerance-freeze.json':tol,
             'fixture-manifest.json':fixture,'resource-forecast.json':forecast}
    for name,value in outputs.items():
        write(REPORT/name,value)
        write(OUTPUT/name,value)
    print(json.dumps({'report':str(REPORT),'output':str(OUTPUT),'anchors':len(contracts),
                      'shards':len(fixture['shards']),'forecast':forecast,'critical_missing':[]},indent=2))


if __name__=='__main__':
    main()
