"""Freeze only redundant logging metadata from approved goal/Path identities."""
import hashlib,json,pathlib
ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
DOC=pathlib.Path('/tmp/eatnav-vnt-goal-ood-simulator/docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,d):
 p=pathlib.Path(p);v=json.dumps(d,sort_keys=True,indent=2)+'\n'
 if p.exists():
  if p.read_text()!=v:raise FileExistsError(p)
 else:
  t=p.with_name(p.name+'.tmp');t.write_text(v);t.replace(p)
rows=read(DOC/'run-identities.json')['runs'];states=read(ROOT/'inputs/selected-states.json')['train'];out=[]
for row in rows:
 state=states[row['state_index']];folder=ROOT/'inputs'/f"state-{row['state_index']:03d}"
 path=folder/f"{row['candidate_id']}-base-path.json"
 if sha(path)!=row['base_path_sha256']:raise ValueError('Path drift '+row['run_id'])
 data={'run_id':row['run_id'],'state_id':row['state_id'],'goal_id':row['goal_id'],
       'goal_class':row['goal_class'],'goal_xy':row['goal_xy'],
       'mission_logged_goal':state['logged_goal'],'base_path':read(path),
       'base_path_sha256':sha(path),'diffusion_seed':row['seed']}
 target=folder/f"{row['run_id']}-goal-metadata.json";atomic(target,data)
 out.append({'run_id':row['run_id'],'metadata_sha256':sha(target)})
atomic(DOC/'goal-metadata-sha.json',{'run_identities_sha256':sha(DOC/'run-identities.json'),'runs':out})
print('GOAL_METADATA_FROZEN',len(out),sha(DOC/'goal-metadata-sha.json'))
