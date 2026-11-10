"""Assemble verified local results for review; never uploads raw data or pushes Git."""
from pathlib import Path
import argparse,csv,hashlib,json,os,shutil,subprocess
import numpy as np
from visualnav_transformer.eat_nav.single_pose2s_contract import INPUT_SCHEMA,PROFILE_SHA
from visualnav_transformer.eat_nav.bidirectional_bank import bank

REPO=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();prod=root/'production';out=REPO/'docs/eat-nav/model2000-pose2s-20261110'
    if not (root/'compute-complete.json').exists():raise ValueError('computation is not completed')
    audit=json.loads((prod/'dataset/audit.json').read_text());training=json.loads((prod/'model-seed123/training-complete.json').read_text());att=json.loads((prod/'final-evaluation/accuracy-attestation.json').read_text());b=json.loads((root/'selector-b/report/report.json').read_text())
    if not audit['complete_parent_coverage'] or audit['observed_parents']!=1330 or training['state']!='completed' or b['completed_runs']!=350:raise ValueError('incomplete computation evidence')
    if sha(prod/'model-seed123/checkpoint.pt')!=training['checkpoint_sha256'] or att['checkpoint_sha256']!=training['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
    for src,dst in [(prod/'model-seed123',out/'models/seed123'),(prod/'final-evaluation',out/'prediction-evaluation'),(root/'selector-b/report',out/'selector-b-evaluation')]:
        shutil.copytree(src,dst,dirs_exist_ok=True)
    for src,name in [(prod/'dataset/audit.json','dataset-audit.json'),(prod/'dataset/split.json','split.json'),(prod/'dataset/source-index.json','dataset-source-index.json'),(prod/'manifest.json','manifest.json'),(prod/'collection-runtime-identity.json','collection-runtime-identity.json')]:shutil.copyfile(src,out/name)
    with np.load(prod/'dataset/final_test-samples.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    ids=np.flatnonzero(arrays['pose_mask']&arrays['input_valid'])
    if not len(ids):raise ValueError('no valid inference example')
    i=int(ids[0]);sample={k:arrays[k][i:i+1] for k in ['dem','state','state_mask','state_age','dem_context','t0_ns','dem_stamp_ns','dem_received_ns']}
    sample.update(paths=np.array([[c.waypoints for c in bank()]]),directions=np.array([[[c.direction] for c in bank()]]),input_schema=np.array(INPUT_SCHEMA),profile_sha256=np.array(PROFILE_SHA),q0_wxyz=arrays['root0'][i:i+1,3:7])
    example=out/'example';example.mkdir(exist_ok=True);np.savez_compressed(example/'input-fourteen.npz',**sample)
    def run(name,*args):subprocess.run([os.sys.executable,str(REPO/'scripts'/name),*map(str,args)],cwd=REPO,check=True)
    checkpoint=out/'models/seed123/checkpoint.pt'
    run('eat_nav_single_pose2s_predict.py','--checkpoint',checkpoint,'--input',example/'input-fourteen.npz','--output',example/'predictions.json','--all-fourteen')
    for device in ['cpu','cuda']:
        run('eat_nav_single_pose2s_inference_benchmark.py','--checkpoint',checkpoint,'--input',example/'input-fourteen.npz','--output',out/f'inference-{device}.json','--device',device)
    current=json.loads((out/'prediction-evaluation/report.json').read_text())
    oldroot=Path('/home/ubuntu/eatnav-moving-state-pose2s-20261103/docs/eat-nav/single-dem-pose2s-20261103')
    old=json.loads((oldroot/'prediction-evaluation/report.json').read_text());oldb=json.loads((oldroot/'selector-b-evaluation/report.json').read_text())
    comparison=dict(old_prediction=dict(overall=old['overall'],passed_groups=old['passed_groups']),new_prediction=dict(overall=current['overall'],passed_groups=current['passed_groups']),old_B=oldb['summary'],new_B=b['summary'],policy_settings_included_in_change=True,old_dataset_used_in_training=False,old_weights_used_in_training=False)
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    def rows(p):
        with p.open() as f:return list(csv.DictReader(f))
    oldrows=rows(oldroot/'prediction-evaluation/per_window.csv');newrows=rows(out/'prediction-evaluation/per_window.csv')
    def index(rs):return {(r['parent_id'],r['stage']):r for r in rs}
    x,y=index(oldrows),index(newrows);common=set(x)&set(y)
    valid=lambda r:r['ground_truth_valid']=='True'
    shared=[k for k in common if valid(x[k]) and valid(y[k])]
    paired=dict(common_parent_segments=len(common),both_valid=len(shared),old_only_valid=sum(valid(x[k]) and not valid(y[k]) for k in common),new_only_valid=sum(valid(y[k]) and not valid(x[k]) for k in common),metrics={})
    for metric in ['longitudinal_cm','lateral_cm','height_cm','roll_deg','pitch_deg','yaw_deg','distance_3d_cm']:
        pairs=[(abs(float(x[k][metric])),abs(float(y[k][metric]))) for k in shared if x[k].get(metric) and y[k].get(metric)]
        paired['metrics'][metric]=dict(count=len(pairs),old_mae=float(np.mean([v[0] for v in pairs])) if pairs else None,new_mae=float(np.mean([v[1] for v in pairs])) if pairs else None)
    (out/'matched-segments-comparison.json').write_text(json.dumps(paired,indent=2)+'\n')
    index_data=dict(schema='eatnav_model2000_pose2s_release_v1',checkpoint='models/seed123/checkpoint.pt',checkpoint_sha256=sha(checkpoint),actor_sha256='56c16ecca41b315ef57aea57beae17a05d94c5c0f54ede88066e9688b7f8a770',profile_sha256=PROFILE_SHA,input_schema=INPUT_SCHEMA,horizon_seconds=2,strict_accuracy_achieved=att['accuracy_achieved'],strict_passed_groups=att['passed_groups'],raw_dataset_local_only=str(prod),dataset_git_upload=False,real_robot_compatibility_verified=False,automatic_real_robot_commands=False,example_is_all14_predictions_not_all14_ground_truth=True)
    (out/'model-index.json').write_text(json.dumps(index_data,indent=2)+'\n')
    (out/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'},indent=2)+'\n')
    print(json.dumps(index_data,indent=2))

if __name__=='__main__':main()
