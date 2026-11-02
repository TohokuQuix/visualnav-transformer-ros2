#!/usr/bin/env python3
"""Evaluate a frozen 2-second model on validation or authorized audit parents.

This command opens the requested role; it does not select models or tune a
threshold. Audit use must follow the separately recorded recipe/weight freeze.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from visualnav_transformer.eat_nav.moving_pose2s_metrics import evaluate_predictions, parent_bootstrap, pose_errors, quat_to_matrix, sixd_to_matrix, pose_metrics, event_metrics, state_groups, EVENTS
from visualnav_transformer.eat_nav.moving_pose2s_model import load_checkpoint, predict, prepare_inputs
from visualnav_transformer.eat_nav.bidirectional_bank import bank

INPUT_KEYS=('dem','actor_state','body_state','paths','directions','candidate_mask','current_quat_wxyz')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def predict_batched(model,normalization,inputs,ids,support,batch_size=64):
    """Only allow current input arrays across the inference boundary."""
    clean={k:inputs[k] for k in INPUT_KEYS if k in inputs};batches=[]
    for offset in range(0,len(ids),batch_size):batches.append(predict(model,normalization,clean,ids[offset:offset+batch_size],support))
    if not batches:raise ValueError('empty evaluation role')
    result=dict(batches[0])
    for key,value in result.items():
        if isinstance(value,np.ndarray):result[key]=np.concatenate([b[key] for b in batches],axis=0)
    result['validity']={key:np.concatenate([b['validity'][key] for b in batches],axis=0) if isinstance(value,np.ndarray) else value for key,value in batches[0]['validity'].items()}
    return result


def benchmark_cpu(model,normalization,inputs,index,support,*,warmup=20,repetitions=200):
    if next(model.parameters()).device.type!='cpu':raise ValueError('CPU benchmark requires CPU model')
    example={k:np.asarray(inputs[k])[index:index+1].copy() for k in INPUT_KEYS if k in inputs}
    candidates=bank();example['paths']=np.stack([c.waypoints for c in candidates])[None].astype(np.float32);example['directions']=np.array([c.direction for c in candidates],np.float32)[None,:,None];example['candidate_mask']=np.ones((1,14),bool)
    transformed=prepare_inputs(example,[0],normalization,'cpu');model.eval()
    def elapsed(run):
        times=[]
        with torch.inference_mode():
            for _ in range(warmup):run()
            for _ in range(repetitions):
                start=time.perf_counter_ns();run();times.append((time.perf_counter_ns()-start)/1e6)
        return dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),count=repetitions)
    nn=elapsed(lambda:model(**transformed));combined=elapsed(lambda:model(**prepare_inputs(example,[0],normalization,'cpu')))
    original=predict(model,normalization,example,[0],support)
    permutation=np.arange(14)[::-1];swapped=dict(example)
    for key in ('paths','directions','candidate_mask'):swapped[key]=example[key][:,permutation]
    reverse=predict(model,normalization,swapped,[0],support)
    single=dict(example)
    for key in ('paths','directions','candidate_mask'):single[key]=example[key][:,:1]
    one=predict(model,normalization,single,[0],support)
    for key in ('delta_position_body_m','delta_quaternion_wxyz','event_probability'):
        np.testing.assert_allclose(original[key][:,permutation],reverse[key],atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(original[key][:,:1],one[key],atol=1e-5,rtol=1e-5)
    return dict(device='cpu',batch_size=1,candidates=14,warmup=warmup,repetitions=repetitions,torch_threads=torch.get_num_threads(),nn_only=nn,input_transform_plus_nn=combined,scope='host CPU; excludes output geometry/serialization; not measured on robot PC',candidate_permutation_verified=True,single_candidate_consistency_verified=True,current_input_example_index=int(index))


def paired_model_comparison(arrays,index,ids,current,reference,resamples=200):
    """Compare both models on identical windows, including moving windows alone.

    Moving-only means ``not index.is_initial`` and is independent of measured
    speed: a later stuck window is still a moving-state decision point.
    Negative paired errors favor the current model over the reference model.
    """
    a={k:np.asarray(v)[ids] for k,v in arrays.items()}
    rows=[index[i] for i in ids];shape=a['pose_mask'].shape
    mask=np.asarray(a['pose_mask'],bool)&np.asarray(a['candidate_mask'],bool)
    gt,valid=sixd_to_matrix(a['rotation_targets']);mask&=valid
    window_parents=np.array([r.get('parent_id',str((r.get('field'),r.get('seed')))) for r in rows])
    parents=np.broadcast_to(window_parents[:,None],shape)
    errors=[];brier=[]
    event_mask=np.asarray(a['event_mask'],bool)&np.asarray(a['candidate_mask'],bool)[...,None]
    for prediction in (current,reference):
        errors.append(pose_errors(prediction['delta_position_body_m'],a['position_targets'],quat_to_matrix(prediction['delta_quaternion_wxyz']),gt,np.broadcast_to(a['body_state'][:,None,:3],shape+(3,)),mask,prediction['validity']['rotation_valid']))
        brier.append(np.where(event_mask&np.isfinite(a['event_targets']),(prediction['event_probability']-a['event_targets'])**2,np.nan))

    def subset_report(take,include_ci=True):
        result=dict(window_count=int(take.sum()),candidate_count=int(a['candidate_mask'][take].sum()),independent_parents=len(set(window_parents[take].tolist())))
        for name,prediction,error in zip(('current','reference'),(current,reference),errors):
            result[name]=dict(pose=pose_metrics({key:value[take] for key,value in error.items()}),events={head:event_metrics(a['event_targets'][take,...,h],prediction['event_probability'][take,...,h],event_mask[take,...,h],parents[take]) for h,head in enumerate(EVENTS)})
        if include_ci:
            ci={key:parent_bootstrap(errors[0][key][take],parents[take],reference=errors[1][key][take],resamples=resamples) for key in ('position_3d_m','so3_rad')}
            for h,name in enumerate(EVENTS):ci[name+'_brier']=parent_bootstrap(brier[0][take,...,h],parents[take],reference=brier[1][take,...,h],resamples=resamples)
            result['paired_difference_95ci']=ci
        return result

    all_windows=subset_report(np.ones(len(ids),bool))
    moving=np.array([not r.get('is_initial',False) for r in rows],bool)
    # Preserve the original whole-role CI keys for existing readers.
    result=dict(all_windows['paired_difference_95ci'])
    result.update(all_windows=all_windows,moving_only=subset_report(moving),moving_definition='not sample_index.is_initial; includes subsequent low-speed/stalled windows',difference_direction='current_minus_reference; negative error difference favors current')
    result['state_stratified']={group:subset_report(take,include_ci=False) for group,take in state_groups(a,rows).items() if take.any()}
    return result


def verify_roles(index,split):
    seen={};members=set()
    for role,ids in split['roles'].items():
        if role not in ('train','validation','audit_eval'):raise ValueError('reserved or unknown split role')
        for i in ids:
            if i in members or i<0 or i>=len(index):raise ValueError('invalid/duplicate split member')
            members.add(i);r=index[i];parent=r.get('parent_id',str((r.get('field'),r.get('seed'))))
            if parent in seen and seen[parent]!=role:raise ValueError('parent leakage across split roles')
            seen[parent]=role


def split_diagnostics(index,split):
    """Geometry overlap is a reported scope limitation, not parent leakage."""
    hashes={role:{index[i].get('geometry_sha256') for i in ids if index[i].get('geometry_sha256')} for role,ids in split['roles'].items()}
    overlap={}
    roles=list(hashes)
    for j,first in enumerate(roles):
        for second in roles[j+1:]:
            common=sorted(hashes[first]&hashes[second]);overlap[first+'__'+second]=dict(count=len(common),sha256=common)
    return dict(geometry_hash_overlap=overlap,geometry_hash_count_by_role={role:len(h) for role,h in hashes.items()},geometry_hash_missing_by_role={role:sum(not index[i].get('geometry_sha256') for i in ids) for role,ids in split['roles'].items()},geometry_scope='shared terrain geometry can occur across independent parent runs; missing hashes are not evidence of disjoint geometry')


def write_reports(output,result):
    with (output/'state-stratified.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['stratum','model','windows','independent_parents','pose_count','position_3d_mean_m','position_3d_median_m','position_3d_p95_m','so3_count','so3_mean_rad','so3_p95_rad'])
        for name,row in result['state_stratified'].items():
            for model,metrics in row['models'].items():
                p=metrics['pose']['position_3d_m'];r=metrics['pose']['rotation_so3_rad'];writer.writerow([name,model,row['window_count'],row['independent_parents'],p['count'],p['mean'],p['median'],p['p95'],r['count'],r['mean'],r['p95']])
        for binding,comparison in result.get('paired_model_minus_reference_95ci',{}).items():
            for name,row in comparison['metrics']['state_stratified'].items():
                p=row['reference']['pose']['position_3d_m'];r=row['reference']['pose']['rotation_so3_rad']
                writer.writerow(['reference:'+name,'reference:'+binding,row['window_count'],row['independent_parents'],p['count'],p['mean'],p['median'],p['p95'],r['count'],r['mean'],r['p95']])
    lines=['# Moving-state 2s endpoint evaluation','',f"Role: {result['metadata']['role']}. Checkpoint SHA-256: `{result['metadata']['checkpoint_sha256']}`.",'',f"{result['window_count']} executed-candidate windows from {result['independent_parents']} independent parents. Pose supervision: complete 2s only. Missing pose: {result['pose_missing_count']}; invalid predicted rotations: {result['invalid_rotation_count']}.",'','| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |','|---|---:|---:|---:|---:|']
    def fmt(value):return 'N/A' if value is None else f'{value:.6g}'
    for name in ('model','no_motion','constant_body_twist'):
        p=result[name]['pose']['position_3d_m'];r=result[name]['pose']['rotation_so3_rad'];lines.append(f"| {name} | {p['count']} | {fmt(p['mean'])} | {fmt(p['p95'])} | {fmt(r['mean'])} |")
    lines+=['','| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for head,row in result['model']['events'].items():lines.append(f"| {head} | {row['positive']} | {row['negative']} | {row['unknown']} | {row['independent_positive_parents']} | {fmt(row['ap'])} | {fmt(row['auroc'])} | {fmt(row['brier'])} | {fmt(row['bce'])} |")
    lines+=['','AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.','', 'Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.','',f"CPU batch1 K14 NN median/P95: {fmt(result['inference_benchmark']['nn_only']['median_ms'])}/{fmt(result['inference_benchmark']['nn_only']['p95_ms'])} ms. Input transform + NN: {fmt(result['inference_benchmark']['input_transform_plus_nn']['median_ms'])}/{fmt(result['inference_benchmark']['input_transform_plus_nn']['p95_ms'])} ms. Host measurement, not robot-PC performance."]
    for binding,comparison in result.get('paired_model_minus_reference_95ci',{}).items():
        row=comparison['metrics']['moving_only'];ci=row['paired_difference_95ci']
        lines+=['','Moving-only same-holdout model comparison','',f"Reference checkpoint: `{comparison['checkpoint']}` (SHA-256 `{binding}`). Moving means a subsequent window (`is_initial=false`), including low-speed/stalled states.",'',f"{row['window_count']} windows from {row['independent_parents']} independent parents. Negative paired error differences favor the current model.",'','| Model | Pose count | Mean 3D error (m) | Mean SO(3) (rad) |','|---|---:|---:|---:|']
        for name in ('current','reference'):
            p=row[name]['pose']['position_3d_m'];r=row[name]['pose']['rotation_so3_rad'];lines.append(f"| {name} | {p['count']} | {fmt(p['mean'])} | {fmt(r['mean'])} |")
        lines+=['','| Paired current minus reference | Difference | Parent bootstrap 95% CI |','|---|---:|---|']
        for key in ('position_3d_m','so3_rad'):
            value=ci[key];lines.append(f"| {key} | {fmt(value['estimate'])} | [{fmt(value['low'])}, {fmt(value['high'])}] |")
        lines+=['','Per-event moving-only Brier differences and both-model state strata are recorded in metrics.json; reference pose strata are also included in state-stratified.csv.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def evaluate(checkpoint,dataset,output,*,role='validation',device='cpu',threads=8,bootstrap_resamples=200,benchmark_warmup=20,benchmark_repetitions=200,compare_checkpoints=()):
    if role not in ('validation','audit_eval'):raise ValueError('evaluation role must be validation or audit_eval')
    if device!='cpu':raise ValueError('this evaluator currently requires --device cpu')
    checkpoint=Path(checkpoint);dataset=Path(dataset);output=Path(output)
    if output.exists():raise FileExistsError('fresh evaluation output required')
    split=json.loads((dataset/'split.json').read_text());index=json.loads((dataset/'sample-index.json').read_text());manifest=json.loads((dataset/'manifest.json').read_text());verify_roles(index,split)
    if manifest.get('schema')!='eatnav_moving_pose2s_dataset_v1' or manifest.get('horizon_sec')!=2.:raise ValueError('dataset schema/horizon mismatch')
    for name,expected in manifest.get('files',{}).items():
        if sha(dataset/name)!=expected:raise ValueError('dataset hash mismatch: '+name)
    ids=np.asarray(split['roles'][role],int)
    if not len(ids):raise ValueError('empty evaluation role')
    torch.set_num_threads(threads);model,payload=load_checkpoint(checkpoint,device=device)
    metadata=dict(schema='eatnav_moving_pose2s_evaluation_v1',role=role,audit_evaluation_opened=role=='audit_eval',reserved_test_opened=False,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),dataset=str(dataset.resolve()),dataset_file_sha256={name:sha(dataset/name) for name in ('samples.npz','sample-index.json','split.json','manifest.json')},split_sha256=sha(dataset/'split.json'),checkpoint_metadata=payload.get('metadata',{}),split_audit=dict(split_diagnostics(index,split),declared={k:v for k,v in split.items() if k!='roles'}),dataset_manifest=manifest,support=payload['support'],input_keys=list(INPUT_KEYS),future_ground_truth_used_as_input=False)
    output.mkdir(parents=True);write_json(output/'evaluation-open-record.json',metadata)
    with np.load(dataset/'samples.npz',allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    inputs={key:arrays[key] for key in INPUT_KEYS if key in arrays};prediction=predict_batched(model,payload['normalization'],inputs,ids,payload['support'])
    result=evaluate_predictions(arrays,index,ids,prediction,train_indices=split['roles']['train'],bootstrap_resamples=bootstrap_resamples);result['metadata']=metadata
    result['inference_benchmark']=benchmark_cpu(model,payload['normalization'],inputs,int(ids[0]),payload['support'],warmup=benchmark_warmup,repetitions=benchmark_repetitions)
    result['paired_model_minus_reference_95ci']={}
    for path in compare_checkpoints:
        path=Path(path);reference,ref_payload=load_checkpoint(path,device=device);ref_prediction=predict_batched(reference,ref_payload['normalization'],inputs,ids,ref_payload['support']);binding=sha(path);result['paired_model_minus_reference_95ci'][binding]=dict(checkpoint=str(path.resolve()),checkpoint_sha256=binding,metrics=paired_model_comparison(arrays,index,ids,prediction,ref_prediction,bootstrap_resamples))
    np.savez_compressed(output/'predictions.npz',indices=ids,**{key:value for key,value in prediction.items() if isinstance(value,np.ndarray)},rotation_valid=prediction['validity']['rotation_valid'])
    write_json(output/'metrics.json',result);write_reports(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--role',choices=('validation','audit_eval'),default='validation');parser.add_argument('--device',choices=('cpu',),default='cpu');parser.add_argument('--threads',type=int,default=8);parser.add_argument('--bootstrap-resamples',type=int,default=200);parser.add_argument('--compare-checkpoint',type=Path,action='append',default=[])
    a=parser.parse_args();result=evaluate(a.checkpoint,a.dataset,a.output,role=a.role,device=a.device,threads=a.threads,bootstrap_resamples=a.bootstrap_resamples,compare_checkpoints=a.compare_checkpoint);print(json.dumps(dict(output=str(a.output),checkpoint_sha256=result['metadata']['checkpoint_sha256'],role=a.role,window_count=result['window_count'])))
if __name__=='__main__':main()
