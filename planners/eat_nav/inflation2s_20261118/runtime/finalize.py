"""Freeze ten Scene2 runs/video for final visual QA; do not resume predecessor early."""
from pathlib import Path
import json,hashlib,csv,subprocess
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 index=read(R/'indices/original.json');assert len(index)==10;assert sha(R/'checkpoint.pt')==read(R/'protocol.json')['nn_sha256']
 for p,h in read(R/'baseline-lock.json').items():assert sha(p)==h,p
 source=read(R/'source.json')
 for p,h in source['files'].items():assert sha(R/p)==h
 import sys;sys.path.insert(0,str(R.parent/'audit'));from field_contract import verify_L_mesh
 mesh=np.load(R/'converted/scene2_mesh.npz');verify_L_mesh(mesh,read(R/'converted/geometry.json'))
 ref=np.load(R/'reference-physics.npz');windows=[];total=0;maxdiff=0.
 for batch in {Path(p).parent for p in index.values()}:
  identity=read(batch/'identity.json');assert identity['checkpoint_sha256']==sha(R/'checkpoint.pt');assert identity['mesh_sha256']==sha(R/'converted/scene2_mesh.npz')
  phys=np.load(batch/'physics.npz')
  for k in ['masses','inertias','materials']:assert np.array_equal(phys[k],np.repeat(ref[k][None],len(phys[k]),axis=0)),k
  dec={(d['trial_id'],d['step']):d for d in read(batch/'decisions.json')}
  for w in read(batch/'windows.json'):
   if w['trial_id'] not in index or not w['valid']:continue
   i=round(w['t0_s']/.05);j=i+40;d=dec[(w['trial_id'],i)];cid=w['candidate_id'];assert cid==d['candidate_id'];assert d['dem_stamp_ns']<=d['t0_ns'] and d['dem_received_ns']<=d['t0_ns']
   with np.load(Path(index[w['trial_id']])/'trace.npz') as z:root=z['root'];rw=z['root_world'];assert j<len(root);assert np.allclose(root[j,:3]-root[i,:3],rw[j,:3]-rw[i,:3],atol=2e-6)
   Q=Rotation.from_quat(root[i,[4,5,6,3]]).as_matrix();T=Rotation.from_quat(root[j,[4,5,6,3]]).as_matrix();P=np.array(d['predicted_rotations'][cid]);p=np.array(d['predicted_positions'][cid]);delta=root[j,:3]-root[i,:3];e=Q@p-delta;xyz=float(np.linalg.norm(e)*100);rp=Rotation.from_matrix(Q@P).as_euler('xyz');rt=Rotation.from_matrix(T).as_euler('xyz');ang=np.abs(np.arctan2(np.sin(rp-rt),np.cos(rp-rt)))*180/np.pi
   maxdiff=max(maxdiff,abs(xyz-w['position3d_cm']),max(abs(ang[k]-w[n]) for k,n in enumerate(['roll_abs_deg','pitch_abs_deg','yaw_abs_deg'])));assert maxdiff<.002
   windows.append(dict(trial_id=w['trial_id'],t0_s=w['t0_s'],t1_s=w['t1_s'],candidate_id=cid,forward_abs_cm=w['forward_abs_cm'],left_abs_cm=w['left_abs_cm'],height_abs_cm=w['height_abs_cm'],position3d_cm=xyz,horizontal_error_cm=float(np.linalg.norm(e[:2])*100),roll_abs_deg=float(ang[0]),pitch_abs_deg=float(ang[1]),yaw_abs_deg=float(ang[2])));total+=1
 out=R/'results/original';audit=read(out/'audit.json');assert audit['state']=='passed' and len(audit['trials'])==10
 with (out/'windows.csv').open('w') as f:
  keys=list(windows[0]) if windows else ['trial_id'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(windows)
 video=read(R/'video-index.json');prov=read(R/'video/original/replay-provenance.json');assert prov['physics_integration_steps_during_frames']==0 and prov['trace_sha256']==video['trace_sha256'] and prov['mesh_sha256']==sha(R/'converted/scene2_mesh.npz');assert sha(video['video'])==video['sha256']
 probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-show_streams','-show_format','-of','json',video['video']]));assert int(probe['streams'][0]['nb_frames'])==prov['frame_count']
 qa=R/'video/QA';qa.mkdir(exist_ok=True);duration=float(probe['format']['duration'])
 for name,t in [('begin',min(1.,duration*.1)),('middle',duration*.5),('end',max(0.,duration-.3))]:
  p=qa/(name+'.png')
  if not p.exists():subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',video['video'],'-frames:v','1',str(p)],check=True)
 result=dict(state='passed_pending_visual_QA',trials=10,complete_executed_windows=total,max_error_recalculation_difference=maxdiff,model_sha256=sha(R/'checkpoint.pt'),scene_source_commit=source['commit'],field_source_kind=source['kind'],mesh_sha256=sha(R/'converted/scene2_mesh.npz'),physics_all_arrays_match_reference=True,source_and_scene1_baseline_unchanged=True,video_sha256=video['sha256'],video_frame_count=prov['frame_count'],video_duration_s=duration,video_physics_steps=0)
 write(R/'final-audit.json',result);write(R/'status.json',dict(phase='compute_completed_pending_visual_QA',completed_trials=10,goal_count=read(out/'summary.json')['goal_count'],QA_images=str(qa)))
 with (R/'LEDGER.md').open('a') as f:f.write('Ten formal runs, analysis, recorded-state video and numerical/source/physics/frame audit finished. Pending visualQA threeframes in video/QA. Do not rerun simulation. After visualQA markcompleted/report, No predecessor resume; old experiments remain unchanged.\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
