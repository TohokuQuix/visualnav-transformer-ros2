"""Approved model2000 workflow through B rollouts; never implicitly pushes Git."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();prod=root/'production'
    def run(name,*args):
        subprocess.run([os.sys.executable,str(REPO/'scripts'/name),*map(str,args)],cwd=REPO,check=True)
    run('eat_nav_single_pose2s_workflow.py','--output',prod)
    notice=prod/'dataset-completed-notification.json'
    if not notice.exists():raise RuntimeError('dataset completion evidence missing')
    checkpoint=prod/'model-seed123/checkpoint.pt'
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selector=root/'selector-b';selector.mkdir(exist_ok=True)
    auth=dict(user_explicitly_authorized=True,checkpoint_sha256=sha,scope='exploratory_selector_and_offline_bag',strict_accuracy_achieved=False,defer_github_push=True,reason='2026-11-10 approved model2000 spec section8 permits exploratory B even if strict accuracy unmet',real_robot_command_authorized=False,conditions=['B'],planned_runs=350,github_after='B evaluation and artifact verification')
    target=selector/'execution-authorization.json'
    if target.exists() and json.loads(target.read_text())!=auth:raise ValueError('selector authorization changed')
    target.write_text(json.dumps(auth,indent=2)+'\n')
    run('eat_nav_single_pose2s_selector_run.py','--output',selector,'--checkpoint',checkpoint,'--conditions','B')
    run('eat_nav_single_pose2s_selector_report.py','--root',selector,'--output',selector/'report')
    (root/'compute-complete.json').write_text(json.dumps(dict(state='computation_complete_release_review_pending',checkpoint_sha256=sha,finished_unix=time.time(),github_push_completed=False),indent=2)+'\n')

if __name__=='__main__':main()
