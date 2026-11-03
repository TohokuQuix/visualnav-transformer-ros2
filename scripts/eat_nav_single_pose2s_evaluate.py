"""Freeze and evaluate executed candidates; no path-selection experiment here."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from visualnav_transformer.eat_nav.single_pose2s_model import SinglePose2s,prepare
from visualnav_transformer.eat_nav.single_pose2s_contract import MODEL_SCHEMA,PROFILE_SHA
from visualnav_transformer.eat_nav.single_pose2s_geometry import path_errors,orientation_errors,arc_position
from visualnav_transformer.eat_nav.single_pose2s_metrics import evaluate_records,save_evaluation
from visualnav_transformer.eat_nav.moving_pose2s_geometry import sixd_to_matrix,quat_to_matrix
from visualnav_transformer.eat_nav.bidirectional_bank import bank


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(state,q0,cid,kind):
    if kind=='no_motion':return np.zeros(3),np.eye(3)
    if kind=='constant_twist':
        v=np.asarray(state[18:21],float);w=np.asarray(state[3:6],float)
        W=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
        se=np.zeros((4,4));se[:3,:3]=W;se[:3,3]=v
        motion=expm(se*2.);return motion[:3,3],motion[:3,:3]
    c=bank()[cid];theta=np.deg2rad(c.arc_angle_deg);xy=arc_position(.6,theta,c.direction)
    R0=quat_to_matrix(q0);yaw=np.arctan2(R0[1,0],R0[0,0]);cy,sy=np.cos(yaw),np.sin(yaw)
    Ryaw=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    dyaw=theta*.6/.75;c,s=np.cos(dyaw),np.sin(dyaw);Rz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R0.T@Ryaw@np.r_[xy,0.],R0.T@Rz@R0


def make_rows(a,records,position,rotation,pred_valid,kind='model'):
    rows=[]
    for i,r in enumerate(records):
        good=bool(a['pose_mask'][i]&a['input_valid'][i]);valid=good and bool(pred_valid[i]);q0=a['root0'][i,3:7]
        row=dict(parent_id=r['parent_id'],field=r['field'],level=r['level'],candidate_id=r['candidate_id'],stage=r['stage'],start_motion=r.get('start_motion','unknown'),ground_truth_valid=good,prediction_valid=valid,euler_singular=False,reason=r.get('reason'))
        row.update({k:None for k in ['longitudinal_cm','lateral_cm','height_cm','roll_deg','pitch_deg','yaw_deg','rotation_deg','distance_3d_cm']})
        if valid:
            gtR,_=sixd_to_matrix(a['rotation'][i]);e=path_errors(position[i],a['position'][i],q0,r['candidate_id']);o=orientation_errors(rotation[i],gtR,q0)
            row.update(longitudinal_cm=e['longitudinal_m']*100,lateral_cm=e['lateral_m']*100,height_cm=e['height_m']*100,distance_3d_cm=e['distance_3d_m']*100,projection_clamped=e['projection_clamped'],roll_deg=float(o['rpy_abs_deg'][0]),pitch_deg=float(o['rpy_abs_deg'][1]),yaw_deg=float(o['rpy_abs_deg'][2]),euler_singular=o['euler_singular'],rotation_deg=o['rotation_deg'])
            if kind=='planned_arc':row.update(height_cm=None,roll_deg=None,pitch_deg=None,rotation_deg=None,distance_3d_cm=None)
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--role',choices=['validation','final_test'],default='final_test');parser.add_argument('--device',default='cpu');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!=MODEL_SCHEMA or checkpoint.get('profile_sha256')!=PROFILE_SHA:raise ValueError('checkpoint schema/profile mismatch')
    freeze=dict(schema='eatnav_single_dem_pose2s_accuracy_freeze_v1',role=args.role,checkpoint_sha256=sha(args.checkpoint),dataset_manifest_sha256=sha(args.dataset/'manifest.json'),role_samples_sha256=sha(args.dataset/(args.role+'-samples.npz')),evaluation_code_sha256=sha(__file__),metrics_sha256=sha(Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/single_pose2s_metrics.py'),opened_unix=time.time(),pass_rule='maximum per490 groups, >=5 samples and parents',profile_sha256=PROFILE_SHA)
    (args.output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    with np.load(args.dataset/(args.role+'-samples.npz'),allow_pickle=False) as f:a={k:f[k] for k in f.files}
    records=json.loads((args.dataset/(args.role+'-records.json')).read_text());n=len(records)
    model=SinglePose2s().to(args.device);model.load_state_dict(checkpoint['state_dict']);model.eval();inputs=prepare(a,checkpoint['normalization'],args.device)
    outputs=[]
    with torch.inference_mode():
        for i in range(0,n,128):outputs.append({k:v[:,0].cpu().numpy() for k,v in model(*(x[i:i+128] for x in inputs)).items()})
    pos=np.concatenate([o['position_z'] for o in outputs])*checkpoint['normalization']['position_std']+checkpoint['normalization']['position_mean'];R,valid=sixd_to_matrix(np.concatenate([o['rotation_6d'] for o in outputs]));valid&=np.isfinite(pos).all(1)
    rows=make_rows(a,records,pos,R,valid);report=evaluate_records(rows);report['freeze']=freeze
    save_evaluation(args.output,rows,report)
    for kind in ['no_motion','constant_twist','planned_arc']:
        positions=np.zeros((n,3));rotations=np.tile(np.eye(3),(n,1,1));available=a['pose_mask']&a['input_valid']
        for i in np.flatnonzero(available):positions[i],rotations[i]=baseline(a['state'][i],a['root0'][i,3:7],int(a['candidate_id'][i]),kind)
        base_rows=make_rows(a,records,positions,rotations,available,kind);save_evaluation(args.output/'baselines'/kind,base_rows,evaluate_records(base_rows))
    # A precision failure is an honest scientific result; selector remains gated.
    (args.output/'accuracy-attestation.json').write_text(json.dumps(dict(accuracy_achieved=report['accuracy_achieved'],checkpoint_sha256=freeze['checkpoint_sha256'],passed_groups=report['passed_groups'],required_groups=490,role=args.role,selector_allowed=bool(args.role=='final_test' and report['accuracy_achieved'])),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed_groups','missing_groups','insufficient_groups','accuracy_achieved']}),flush=True)

if __name__=='__main__':main()
