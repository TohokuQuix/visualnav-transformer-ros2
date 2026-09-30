"""Frozen goal sweep using unchanged reference EMA sampler; no simulator imports."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,time
import numpy as np,torch
from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config

ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood')
SRC=pathlib.Path('/tmp/eatnav-train')
CHECKPOINT=pathlib.Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,data):
    content=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n';p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_text()!=content:raise ValueError('frozen marker drift '+str(p))
        return
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(content);tmp.replace(p)
def run(index):
    started=time.monotonic();source=read(ROOT/'goal-sweep-inputs.json')
    if sha(CHECKPOINT)!='191d6a03266bca2ce8d43afac44b04828ae0d2571632be4f47a5f0dc3564b4e0':raise ValueError('checkpoint drift')
    if sha(ROOT/'selected-states.npz')!=source['selected_states_npz_sha256']:raise ValueError('state arrays drift')
    if index<0 or index>=64:raise ValueError('state index')
    selected=np.load(ROOT/'selected-states.npz');rows=source['inputs'][index*13:(index+1)*13]
    if len(rows)!=13 or len({r['state_id'] for r in rows})!=1:raise ValueError('input grouping drift')
    sid=rows[0]['state_id'];dem=selected['dem'][index]
    if hashlib.sha256(dem.tobytes()).hexdigest()!=rows[0]['dem_sha256']:raise ValueError('DEM drift')
    out=ROOT/'inference'/f'{sid}.npz';marker=ROOT/'inference'/f'{sid}.done.json'
    if marker.exists():
        if not out.exists() or read(marker)['output_sha256']!=sha(out) or read(marker)['input_sha256']!=sha(ROOT/'goal-sweep-inputs.json'):
            raise ValueError('resume marker drift')
        print('RESUME_SKIP',sid,flush=True);return
    cfg=load_eat_config(SRC/'docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
    model,stats,_=load_checkpoint(CHECKPOINT,cfg);model=model.to('cuda').eval()
    if not np.allclose(stats.mean,[.028116857632994652,-4.290757715352811e-05],atol=1e-8):raise ValueError('normalization drift')
    torch.cuda.reset_peak_memory_stats();d=torch.as_tensor(dem[None],device='cuda');paths=[]
    with torch.inference_mode():
        for row in rows:
            goal=torch.as_tensor(np.asarray(row['goal_xy'],dtype=np.float32)[None],device='cuda')
            samples=[]
            for seed in source['seeds']:
                path=model.sample(d,goal,None,seed=seed)[0].detach().cpu().numpy()
                if path.shape!=(8,2) or not np.isfinite(path).all():raise ValueError('invalid sampled Path')
                samples.append(path)
            paths.append(samples)
    paths=np.asarray(paths,dtype=np.float32)
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,paths=paths,goals=np.asarray([r['goal_xy'] for r in rows],dtype=np.float32),
        seeds=np.asarray(source['seeds'],dtype=np.int64),state_index=index)
    tmp.replace(out)
    elapsed=time.monotonic()-started
    write(marker,{'state_id':sid,'state_index':index,'path_count':104,'input_sha256':sha(ROOT/'goal-sweep-inputs.json'),
        'source_model_sha256':sha(SRC/'src/visualnav_transformer/eat_nav/model.py'),
        'checkpoint_sha256':sha(CHECKPOINT),'output_sha256':sha(out),'elapsed_sec':elapsed,
        'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'gpu_reserved_peak_bytes':torch.cuda.max_memory_reserved()})
    print('STATE_DONE',sid,'elapsed_sec',round(elapsed,2),'gpu_peak_GB',round(torch.cuda.max_memory_allocated()/1e9,3),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state-index',type=int,required=True);run(p.parse_args().state_index)
