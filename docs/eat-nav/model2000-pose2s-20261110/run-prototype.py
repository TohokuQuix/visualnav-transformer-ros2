import json,sys,subprocess
from pathlib import Path
from eat_nav_single_pose2s_pipeline import stage_source,run_batch,write
from visualnav_transformer.eat_nav.single_pose2s_contract import make_manifest
root=Path('/home/ubuntu/eatnav-model2000-pose2s-data-20261110/prototype-r2')
root.mkdir(parents=True,exist_ok=True)
m=make_manifest(123);m['parents']=[p for p in m['parents'] if p['role']=='train' and p['field']=='pyramid_stairs' and p['level']==4][:2];m['evaluation_terrain_policy']='known_terrain_independent_runs'
write(root/'manifest.json',m)
stage=stage_source(root)
run_batch(root,stage,m['parents'],'prototype-n2',record_lidar=True)
subprocess.run([sys.executable,'scripts/eat_nav_single_pose2s_fastlio_batch.py','--raw',str(root/'raw'),'--workers','2'],check=True)
from visualnav_transformer.eat_nav.single_pose2s_dataset import build_dataset
build_dataset(root/'raw',m,root/'dataset')
print((root/'dataset/audit.json').read_text())
