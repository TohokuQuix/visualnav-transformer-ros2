import json
from pathlib import Path

from visualnav_transformer.eat_nav.p1_batch_plan import build_plan
from visualnav_transformer.eat_nav.p1_batch_prepath import compare_prepath


BACKUP = Path('/home/kojima/eatnav-backups/2026-10-09-p1-radial-28-analysis')


def test_exact_reference_passes_and_one_actor_corruption_stops_all_paths():
    manifest = json.loads((BACKUP / 'integrated-manifest.json').read_text())
    contract = json.loads(Path('config/eat_nav_p1_batch_source_contract.json').read_text())
    tolerance = json.loads(Path('config/eat_nav_p1_batch_parity.json').read_text())['pre_path_tolerance']
    plan = build_plan(manifest, contract)
    records = []
    for item in plan['targets']:
        run = BACKUP / 'runs' / item['physical_run_id'] / 'run'
        pose = item['source_pose_id']
        state = json.loads((run / 'trace' / f'{pose}.jsonl').read_text().splitlines()[-1])['state']
        actor = json.loads((run / 'actor' / pose / 'settled_step_200.json').read_text())
        frames = [json.loads((run / 'dem' / pose / f'frame-{i:02d}.json').read_text()) for i in range(4)]
        records.append({'physical_run_id': item['physical_run_id'],
                        'target_env_index': item['target_env_index'],
                        'terrain_metadata': {'seed': item['terrain_seed'], 'terrain_type': item['terrain_type']},
                        'target_tile_origin_xy': item['source_tile_origin_xy'],
                        'root_state_w': state['root_state_w'],
                        'joint_position': state['joint_position'],
                        'joint_velocity': state['joint_velocity'],
                        'actor_input': actor['values'],
                        'dem_frames': [{**f, 'finite': True} for f in frames],
                        'terminated_during_settling': False})
    assert compare_prepath(records, plan, BACKUP, tolerance)['pass']
    records[0]['actor_input'] = list(records[0]['actor_input'])
    records[0]['actor_input'][0] += 0.1
    failed = compare_prepath(records, plan, BACKUP, tolerance)
    assert failed['pass'] is False
    assert failed['Path_publications'] == 0
    assert failed['rows'][0]['first_failing_signal'] == 'actor_input'
