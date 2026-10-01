"""Freeze the approved four-state simulator pilot from existing offline paths."""
from __future__ import annotations
import gzip,hashlib,json,pathlib
import numpy as np,yaml

ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
OFF=pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood')
MANIFEST=pathlib.Path('/tmp/eatnav-extended/output/manifest.json')
DOC=pathlib.Path('/tmp/eatnav-vnt-goal-ood-simulator/docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator')
OFFDOC=pathlib.Path('/tmp/eatnav-vnt-goal-ood-simulator/docs/quality-identifiability-gate/candidate-coverage/goal-ood')
SEED=(1001,1002)
GOALS=('logged','id_forward_long','near_left_lateral','far_diag_right')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def atomic(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 content=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
 if path.exists():
  if path.read_text()!=content:raise FileExistsError('freeze changed '+str(path))
  return
 t=path.with_name(path.name+'.tmp');t.write_text(content);t.replace(path)
def npz_once(path,**arrays):
 path=pathlib.Path(path)
 if path.exists():
  with np.load(path) as prior:
   if set(prior.files)!=set(arrays) or any(not np.array_equal(prior[k],v) for k,v in arrays.items()):raise FileExistsError(path)
  return
 t=path.with_name(path.name+'.tmp');
 with t.open('wb') as f:np.savez_compressed(f,**arrays)
 t.replace(path)

def main():
 ROOT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
 proposal=read(OFFDOC/'simulator-pilot-proposal.json');selection=read(OFFDOC/'state-selection.json');sweep=read(OFFDOC/'goal-sweep-inputs.json')
 provenance=read(OFFDOC/'source-provenance.json');manifest=read(MANIFEST)
 if sha(MANIFEST)!=provenance['input_sha256']['manifest'] or sha(OFFDOC/'simulator-pilot-proposal.json')!='3f0c628e96fa073a7fb2cf43f29a57b48162a9b5c5c8770b08429d2a883bb463':
  raise ValueError('source manifest/proposal SHA mismatch')
 if sha(OFF/'selected-states.npz')!=sweep['selected_states_npz_sha256']:raise ValueError('selected state array drift')
 if len(proposal['states'])!=4 or proposal['total_runs']!=44 or proposal['paired_seeds']!=list(SEED):raise ValueError('proposal scope')
 selected=proposal['states'];maxstep=max(s['step'] for s in selected)+120
 actions={};commands=[{} for _ in selected];anchors={}
 for chunk in manifest['chunks']:
  path=MANIFEST.parent/chunk['file']
  if sha(path)!=chunk['sha256']:raise ValueError('manifest chunk SHA mismatch')
  with gzip.open(path,'rt') as f:
   for line in f:
    row=json.loads(line);step=int(row['step'])
    if step>maxstep:continue
    env=int(row['env_id']);actions.setdefault(step,{})[env]=row['action']
    for j,state in enumerate(selected):
     if env!=state['episode_key'][1] or step>state['step']:continue
     commands[j][step]=row['command']
     if step==state['step']:
      if [row['session_id'],env,row['episode_id']]!=state['episode_key'] or row['pre_reset']:
       raise ValueError('anchor episode identity/pre-reset mismatch '+state['state_id'])
      anchors[j]=row
  if len(anchors)==4 and all(len(actions.get(step,{}))==100 for step in range(maxstep+1)):break
 if len(anchors)!=4 or not all(len(actions.get(step,{}))==100 for step in range(maxstep+1)):
  raise ValueError('incomplete 100-env full-prefix actions')
 all_actions=np.asarray([[actions[t][e] for e in range(100)] for t in range(maxstep+1)],dtype=np.float32)
 states_out=[];identities=[];input_audit=[]
 for j,state in enumerate(selected):
  if state['state_id'] not in {s['state_id'] for s in selection['states']} or state['local_step']>40:raise ValueError('state selection drift')
  sid=state['state_id'];folder=ROOT/'inputs'/f'state-{j:03d}';folder.mkdir(parents=True,exist_ok=True)
  step=int(state['step']);command=np.asarray([commands[j][t] for t in range(step+1)],dtype=np.float32)
  npz_once(folder/'prefix-actions.npz',actions=all_actions[:step+1],selected_env_commands=command)
  npz_once(folder/'suffix-actions-120.npz',actions=all_actions[step+1:step+121])
  atomic(folder/'logged-anchor.json',anchors[j])
  state_cfg={'schema_version':3,'control_steps':120,'anchor_limits':{'position_m':.02,'orientation_rad':.03,'dem_rms_m':.03,'mask_iou_min':.95},
             'source_state':{'split':'train','state_index':j,'env_index':state['episode_key'][1],'episode_id':state['episode_key'][2],'anchor_step':step}}
  config_content=yaml.safe_dump(state_cfg,sort_keys=True);config_path=folder/'config-120.yaml'
  if config_path.exists() and config_path.read_text()!=config_content:raise FileExistsError(config_path)
  if not config_path.exists():config_path.write_text(config_content)
  with np.load(OFF/'inference'/f'{sid}.npz') as data:
   if data['paths'].shape!=(13,8,8,2) or data['seeds'].tolist()!=sweep['seeds']:raise ValueError('offline Path shape/seed')
   paths=np.asarray(data['paths'],dtype=np.float32);goals=np.asarray(data['goals'],dtype=np.float32)
  marker=read(OFF/'inference'/f'{sid}.done.json')
  if sha(OFF/'inference'/f'{sid}.npz')!=marker['output_sha256']:raise ValueError('offline output SHA drift')
  state_rows=[r for r in sweep['inputs'] if r['state_id']==sid]
  for goal_id in GOALS:
   row=next(r for r in state_rows if r['goal_id']==goal_id)
   if not row['geometry_screen_valid']:raise ValueError('selected goal geometry invalid')
   order=row['goal_order'];assert np.allclose(goals[order],row['goal_xy'],atol=1e-7)
   for k,seed in enumerate(SEED):
    candidate=f'{goal_id}-seed-{seed}';points=[[0.,0.]]+paths[order,k].astype(float).tolist()
    path=folder/f'{candidate}-base-path.json';atomic(path,points)
    identities.append({'order':len(identities),'kind':'candidate','state_index':j,'state_id':sid,'terrain':state['terrain'],
       'episode_key':state['episode_key'],'anchor_step':step,'prefix_steps':step+1,'goal_id':goal_id,
       'goal_class':row['declared_class'],'goal_xy':row['goal_xy'],'geometry_screen_valid':row['geometry_screen_valid'],
       'seed':seed,'candidate_id':candidate,'base_path_sha256':sha(path),'offline_output_sha256':marker['output_sha256'],
       'offline_goal_order':order,'offline_seed_index':k})
  nominal='logged-seed-1001'
  states_out.append({**state,'terrain_type':state['terrain'],'state_index':j})
  input_audit.append({'state_id':sid,'terrain':state['terrain'],'episode_key':state['episode_key'],'local_step':state['local_step'],
    'prefix_steps':step+1,'prefix_sha256':sha(folder/'prefix-actions.npz'),'suffix_sha256':sha(folder/'suffix-actions-120.npz'),
    'logged_anchor_sha256':sha(folder/'logged-anchor.json'),'config_sha256':sha(config_path),
    'offline_output_sha256':marker['output_sha256'],'nominal_candidate_id':nominal,
    'nominal_base_path_sha256':sha(folder/f'{nominal}-base-path.json')})
 if len(identities)!=32:raise ValueError('candidate count')
 # The run order is fixed by proposal state order, then three nominal controls,
 # then the proposal's four goal types with seeds 1001/1002.
 runs=[]
 for j,state in enumerate(selected):
  nominal=next(x for x in identities if x['state_index']==j and x['candidate_id']=='logged-seed-1001')
  for repeat in range(3):runs.append({**nominal,'kind':'duplicate','repeat':repeat,'run_id':f'{state["state_id"]}-nominal-{repeat:02d}','order':len(runs)})
  for candidate in [x for x in identities if x['state_index']==j]:
   runs.append({**candidate,'run_id':f'{state["state_id"]}-{candidate["candidate_id"]}','order':len(runs)})
 if len(runs)!=44:raise ValueError('44 run count')
 atomic(ROOT/'inputs/selected-states.json',{'train':states_out,'heldout_used':False})
 atomic(DOC/'run-identities.json',{'runs':runs,'control_hz':20,'control_steps':120,'path_points':8,'path_interval_sec':.25,'candidate_count':32,'duplicate_count':12})
 atomic(DOC/'pilot-input-freeze.json',{'proposal_sha256':sha(OFFDOC/'simulator-pilot-proposal.json'),
    'offline_goal_sweep_inputs_sha256':sha(OFFDOC/'goal-sweep-inputs.json'),
    'offline_state_selection_sha256':sha(OFFDOC/'state-selection.json'),
    'offline_ledger_sha256':sha(OFFDOC/'artifact-sha-ledger.json'),
    'manifest_sha256':sha(MANIFEST),'split_hash':provenance['split_hash'],
    'reference_checkpoint_sha256':provenance['input_sha256']['checkpoint'],
    'selected_state_sha256':sha(ROOT/'inputs/selected-states.json'),
    'states':input_audit,'run_identities_sha256':sha(DOC/'run-identities.json'),
    'outcomes_used':False,'paths_reinferred':False})
 atomic(ROOT/'s0.done.json',{'stage':'S0_INPUTS_FROZEN','freeze_sha256':sha(DOC/'pilot-input-freeze.json'),
    'run_identities_sha256':sha(DOC/'run-identities.json')})
 atomic(ROOT/'status.json',{'stage':'S0_INPUTS_FROZEN','completed_runs':0,'candidate_runs':0})
 print('S0_FROZEN',len(runs),sha(DOC/'pilot-input-freeze.json'),flush=True)
if __name__=='__main__':main()
