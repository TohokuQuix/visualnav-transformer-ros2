"""Local orchestration. Requires explicit compatibility or simulation-only authorization."""
from __future__ import annotations
import argparse,json,subprocess,os,time,fcntl
from pathlib import Path
from visualnav_transformer.eat_nav.single_pose2s_dataset import build_dataset

REPO=Path(__file__).resolve().parents[1]


def save(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output.resolve()
    authorization=root/'execution-authorization.json'
    if not authorization.exists():raise ValueError('real-PC compatibility or user-approved simulation-only execution authorization is required')
    allowed=json.loads(authorization.read_text())
    if not allowed.get('user_explicitly_confirmed') or allowed.get('scope') not in ['real_pc_compatibility_confirmed','simulation_only_collection_training_evaluation']:raise ValueError('execution gate is unresolved')
    lock=(root/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(schema='eatnav_single_dem_pose2s_workflow_v1',pid=os.getpid(),started_unix=time.time(),scope=allowed['scope'],accuracy_achieved=False,path_selection_started=False,github_product_upload_done=False)
    status_path=root/'workflow-status.json'
    def phase(name):status.update(phase=name,updated_unix=time.time());save(status_path,status);print('PHASE '+name,flush=True)
    def run(script,argv):
        subprocess.run([os.sys.executable,str(REPO/'scripts'/script),*map(str,argv)],cwd=REPO,env=dict(os.environ,PYTHONPATH=str(REPO/'src')+':'+str(REPO/'scripts')),check=True)
    try:
        phase('production_collection')
        if not (root/'collection-complete.json').exists():run('eat_nav_single_pose2s_pipeline.py',['--output',root,'--mode','collect','--num-envs','30','--record-lidar'])
        phase('actual_fastlio_processing')
        run('eat_nav_single_pose2s_fastlio_batch.py',['--raw',root/'raw','--workers','4'])
        phase('dataset_audit')
        manifest=json.loads((root/'manifest.json').read_text());dataset=root/'dataset'
        if not (dataset/'audit.json').exists():build_dataset(root/'raw',manifest,dataset)
        audit=json.loads((dataset/'audit.json').read_text())
        if not audit['complete_parent_coverage']:raise ValueError('missing expected parents; do not train as completed collection')
        save(root/'dataset-completed-notification.json',dict(state='dataset_completed_and_audited',audit_path=str(dataset/'audit.json'),valid_pose_windows=audit['valid_pose_windows'],observed_parents=audit['observed_parents'],notification_requested_by_user=True,dataset_not_for_git=True,completed_unix=time.time()))
        print('DATASET_COMPLETED '+json.dumps(dict(valid_windows=audit['valid_pose_windows'],parents=audit['observed_parents'],role_counts=audit['role_counts'],audit=str(dataset/'audit.json'))),flush=True)
        phase('training')
        models=root/'model-seed123'
        if not (models/'training-complete.json').exists():
            if models.exists():raise ValueError('partial training retained; inspect before creating a new attempt')
            import torch
            device='cuda' if torch.cuda.is_available() else 'cpu'
            run('eat_nav_single_pose2s_train.py',['--dataset',dataset,'--output',models,'--seed','123','--device',device])
        phase('independent_accuracy_evaluation')
        evaluation=root/'final-evaluation'
        if not (evaluation/'accuracy-attestation.json').exists():
            if evaluation.exists():raise ValueError('partial evaluation retained; inspect freeze before resuming')
            run('eat_nav_single_pose2s_evaluate.py',['--dataset',dataset,'--checkpoint',models/'checkpoint.pt','--output',evaluation,'--role','final_test','--device','cpu'])
        attestation=json.loads((evaluation/'accuracy-attestation.json').read_text());status.update(accuracy_achieved=attestation['accuracy_achieved'],passed_groups=attestation['passed_groups'],dataset_valid_windows=audit['valid_pose_windows'],dataset_observed_parents=audit['observed_parents'])
        phase('accuracy_passed_release_review_pending' if attestation['accuracy_achieved'] else 'prediction_accuracy_not_met')
        status.update(selector_allowed=bool(attestation['accuracy_achieved']),ended_unix=time.time());save(status_path,status)
    except BaseException as exc:
        status.update(phase='failed_preserved',error=repr(exc),updated_unix=time.time());save(status_path,status);raise

if __name__=='__main__':main()
