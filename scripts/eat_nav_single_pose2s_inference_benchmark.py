"""Measure actual checkpoint inference; no robot publishers or planner changes."""
from __future__ import annotations
import argparse,json,time,platform,hashlib
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.single_pose2s_model import SinglePose2s,predict,prepare,latency_summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--repeats',type=int,default=100);a=p.parse_args()
    if a.repeats<10:raise ValueError('at least10 repeats')
    torch.set_num_threads(4)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    with np.load(a.input,allow_pickle=False) as f:sample={k:f[k] for k in f.files}
    if sample['paths'].shape[:2]!=(1,14):raise ValueError('one observation, all14 candidates required')
    def sync():
        if str(a.device).startswith('cuda'):torch.cuda.synchronize()
    def clock(function):
        sync();start=time.perf_counter();out=function();sync();return time.perf_counter()-start,out
    first,result=clock(lambda:predict(payload,sample,a.device))
    if not np.asarray(result['rotation_valid']).all():raise ValueError('invalid public inference')
    setup_start=time.perf_counter();model=SinglePose2s().to(a.device);model.load_state_dict(payload['state_dict']);model.eval();sync();setup=time.perf_counter()-setup_start
    inputs=prepare(sample,payload['normalization'],a.device)
    with torch.inference_mode():
        for _ in range(20):model(*inputs)
        sync()
        model_times=[clock(lambda:model(*inputs))[0] for _ in range(a.repeats)]
    preprocessing=[clock(lambda:prepare(sample,payload['normalization'],a.device))[0] for _ in range(a.repeats)]
    total=[clock(lambda:predict(payload,sample,a.device))[0] for _ in range(a.repeats)]
    report=dict(schema='eatnav_single_dem_pose2s_inference_benchmark_v1',checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),device=a.device,torch_version=torch.__version__,python=platform.python_version(),cpu_threads=torch.get_num_threads(),batch=1,candidates=14,warmup_iterations=20,model_only=latency_summary(model_times),preprocessing_only=latency_summary(preprocessing),public_end_to_end=latency_summary(total),first_public_call_in_existing_process_ms=first*1000,persistent_model_setup_ms=setup*1000,public_api_includes='input validation, preprocessing, model construction/load, model forward, coordinate/output conversion; no file IO',model_only_excludes='preprocessing, construction, postprocessing, file IO',horizon_seconds=2)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
