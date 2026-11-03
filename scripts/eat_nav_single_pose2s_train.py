"""Train on train parents, select on validation; never read final-test outcomes."""
from __future__ import annotations
import argparse,json,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.single_pose2s_model import SinglePose2s,fit_normalization,prepare,pose_loss
from visualnav_transformer.eat_nav.single_pose2s_contract import INPUT_SCHEMA,MODEL_SCHEMA,PROFILE_SHA
from visualnav_transformer.eat_nav.single_pose2s_geometry import path_errors
from visualnav_transformer.eat_nav.moving_pose2s_geometry import sixd_to_matrix,quat_to_matrix,matrix_to_rpy


def save_json(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def validation_basis(arrays,records,ids):
    basis=[]
    for i in ids:
        gt=arrays['position'][i];q=arrays['root0'][i,3:7];cid=records[i]['candidate_id'];axes=[]
        for axis in np.eye(3):
            e=path_errors(gt+axis,gt,q,cid);axes.append([e['longitudinal_m'],e['lateral_m']])
        basis.append(np.asarray(axes).T)
    return np.asarray(basis)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=200);p.add_argument('--seed',type=int,default=123);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--device',default='cuda');p.add_argument('--diagnostic',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    parts=[];records=[];sizes=[]
    for role in ['train','validation']:
        with np.load(a.dataset/(role+'-samples.npz'),allow_pickle=False) as f:parts.append({k:f[k] for k in f.files})
        records.extend(json.loads((a.dataset/(role+'-records.json')).read_text()));sizes.append(len(parts[-1]['dem']))
    arrays={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
    splits=dict(train=list(range(sizes[0])),validation=list(range(sizes[0],sum(sizes))))
    audit=json.loads((a.dataset/'audit.json').read_text())
    if not audit['complete_parent_coverage'] and not a.diagnostic:raise ValueError('production dataset incomplete')
    good=arrays['pose_mask']&arrays['input_valid'];train=np.array([i for i in splits['train'] if good[i]],int);val=np.array([i for i in splits['validation'] if good[i]],int)
    if not len(train) or not len(val):raise ValueError('train and independent validation required')
    norm=fit_normalization(arrays,train);save_json(a.output/'normalization.json',norm)
    model=SinglePose2s().to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    # Inputs are current-only. No final labels are used by optimization/selection.
    inputs=prepare(arrays,norm,a.device);position=torch.as_tensor(arrays['position'],device=a.device);rotation=torch.as_tensor(arrays['rotation'],device=a.device)
    R0=quat_to_matrix(arrays['root0'][val,3:7]);Rgt,_=sixd_to_matrix(arrays['rotation'][val]);gt_rpy,gt_lock=matrix_to_rpy(R0@Rgt);basis=validation_basis(arrays,records,val)
    best=float('inf');best_epoch=0;history=[];start=time.time()
    config=dict(epochs=a.epochs,seed=a.seed,batch_size=a.batch_size,lr=.001,weight_decay=.0001,min_epochs=30,patience=25,diagnostic=a.diagnostic,selection='validation normalized path longitudinal/lateral and future wrapped RPY MAE',profile_sha256=PROFILE_SHA)
    config.update(training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_source_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/single_pose2s_model.py').read_bytes()).hexdigest(),torch_version=str(torch.__version__))
    save_json(a.output/'config.json',config)
    for epoch in range(1,a.epochs+1):
        model.train();losses=[]
        order=np.random.permutation(train)
        for begin in range(0,len(order),a.batch_size):
            ids=order[begin:begin+a.batch_size];optimizer.zero_grad(set_to_none=True)
            out=model(*(x[ids] for x in inputs));loss=pose_loss(out,position[ids],rotation[ids],norm)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach().cpu()))
        model.eval();parts=[]
        with torch.inference_mode():
            for begin in range(0,len(val),a.batch_size):
                ids=val[begin:begin+a.batch_size];out=model(*(x[ids] for x in inputs));parts.append({k:v[:,0].cpu().numpy() for k,v in out.items()})
        pos=np.concatenate([r['position_z'] for r in parts])*norm['position_std']+norm['position_mean'];Rpred,valid=sixd_to_matrix(np.concatenate([r['rotation_6d'] for r in parts]));prpy,plock=matrix_to_rpy(R0@Rpred)
        longitudinal_lateral=np.abs(np.einsum('nij,nj->ni',basis,pos-arrays['position'][val]));angles=np.rad2deg(np.abs(np.arctan2(np.sin(prpy-gt_rpy),np.cos(prpy-gt_rpy))))
        score=float(np.concatenate([longitudinal_lateral/np.array([.3,.1]),angles/10],1).mean()) if valid.all() and not (plock|gt_lock).any() else 1e9
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_score=score,elapsed_seconds=time.time()-start);history.append(row);print(json.dumps(row),flush=True)
        if score<best:
            best=score;best_epoch=epoch
            payload=dict(schema=MODEL_SCHEMA,input_schema=INPUT_SCHEMA,profile_sha256=PROFILE_SHA,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},normalization=norm,config=config,best_epoch=best_epoch,best_validation_score=best,dataset_manifest_sha256=hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest(),dataset_audit_sha256=hashlib.sha256((a.dataset/'audit.json').read_bytes()).hexdigest(),accuracy_achieved=False,final_test_opened=False)
            tmp=a.output/'checkpoint.pt.tmp';torch.save(payload,tmp);tmp.replace(a.output/'checkpoint.pt')
        save_json(a.output/'history.json',history)
        if epoch>=30 and epoch-best_epoch>=25:break
    if not (a.output/'checkpoint.pt').exists():raise ValueError('no finite validation checkpoint')
    sha=hashlib.sha256((a.output/'checkpoint.pt').read_bytes()).hexdigest();save_json(a.output/'training-complete.json',dict(state='completed',checkpoint_sha256=sha,best_epoch=best_epoch,epochs=len(history),train_windows=len(train),validation_windows=len(val),final_test_opened=False,diagnostic=a.diagnostic,elapsed_seconds=time.time()-start))

if __name__=='__main__':main()
