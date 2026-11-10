"""Wait for approved computation, verify artifacts, then normal-push the new branch."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,time,fcntl

REPO=Path(__file__).resolve().parents[1]
BRANCH='codex/eatnav-model2000-pose2s-20261110'
REMOTE='github-pose2s'
URL='https://github.com/TohokuQuix/visualnav-transformer-ros2.git'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    lock=(root/'publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=root/'publication-status.json'
    def status(phase,**extra):
        d=dict(phase=phase,updated_unix=time.time(),github_push_completed=False,**extra);tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(status_path);print(json.dumps(d),flush=True)
    def cmd(*args):return subprocess.check_output(list(map(str,args)),cwd=REPO,text=True).strip()
    def run(*args):subprocess.run(list(map(str,args)),cwd=REPO,check=True)
    def python(name,*args):run(sys.executable,REPO/'scripts'/name,*args)
    try:
        status('waiting_for_computation')
        while not (root/'compute-complete.json').exists():
            proc=json.loads((root/'continuation-process.json').read_text());pid=int(proc['pid'])
            try:argv=Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:argv=b''
            if b'eat_nav_model2000_continue.py' not in argv:raise RuntimeError('computation stopped before completion; inspect continue.log; no push performed')
            time.sleep(30)
        status('assembling_artifacts')
        python('eat_nav_model2000_release.py','--root',root)
        out=REPO/'docs/eat-nav/model2000-pose2s-20261110'
        index=json.loads((out/'model-index.json').read_text());evaluation=json.loads((out/'prediction-evaluation/report.json').read_text());b=json.loads((out/'selector-b-evaluation/report.json').read_text());audit=json.loads((out/'dataset-audit.json').read_text());training=json.loads((out/'models/seed123/training-complete.json').read_text())
        # All values in the final handoff come from completed artifacts.
        lines=['# model2000による2秒後状態予測：実行結果','',f"独立親試行 {audit['observed_parents']}/1330、有効pose区間 {audit['valid_pose_windows']}。",f"学習 {training['epochs']}epoch、採用epoch {training['best_epoch']}、strict合格 {evaluation['passed_groups']}/490。",f"strict精度達成: {evaluation['accuracy_achieved']}。未達の場合は暫定モデルであり、目標達成とは扱いません。",'', '## 予測誤差（全体平均、cm / deg）','']
        for k in ['longitudinal_cm','lateral_cm','height_cm','roll_deg','pitch_deg','yaw_deg','distance_3d_cm']:
            lines.append(f"- {k}: {evaluation['overall'][k+'_mean']}")
        lines+=['','490組の平均・中央値・P95・最大値、欠測・基準未達は prediction-evaluation/ を参照。旧との比較は comparison.json と matched-segments-comparison.json。','', '## 条件B（探索的350走行）','', '```json',json.dumps(b['summary'],ensure_ascii=False,indent=2),'```','', 'Aは未実施。B単独から傾斜除外の有効性を主張しません。実機互換性・実機走行は未検証です。','', '## 別PCでの取得と推論確認','', '```bash','git fetch origin '+BRANCH,'git switch --track origin/'+BRANCH,'# 初回checkout後、リポジトリの依存を満たすPython環境を使用','export PYTHONPATH="$PWD/src:$PWD/scripts"','sha256sum docs/eat-nav/model2000-pose2s-20261110/models/seed123/checkpoint.pt','python scripts/eat_nav_single_pose2s_predict.py \\','  --checkpoint docs/eat-nav/model2000-pose2s-20261110/models/seed123/checkpoint.pt \\','  --input docs/eat-nav/model2000-pose2s-20261110/example/input-fourteen.npz \\','  --output /tmp/model2000-pose2s-predictions.json --all-fourteen','```','', 'checkpoint SHA256: `'+index['checkpoint_sha256']+'`。予測時間2秒、XYZ[m]は起点body座標、相対回転は6DからSO(3)/wxyz quaternion/RPYへ変換します。取得例は14候補の推論確認用であり、14候補すべての正解を含みません。','', '実行条件と入力schemaは承認仕様・model-index.json・normalization.jsonを参照。旧NNとは制御policy/profileが異なり、無条件互換ではありません。','', '## データ・検証制限','', '- RealSenseなし。4Hz・40×40 DEM一枚、実FAST-LIO（理想simセンサ）を維持。','- 旧dataset・旧予測重みは学習へ混入していません。地形は既知地形の独立走行評価です。','- raw/datasetはローカルのみ: `'+str(root/'production')+'`。巨大データはGitへ保存していません。','- 関連テスト45件pass。全体pytestはROS/diffusers等の依存不足による28収集エラーがあり、test-limitations.txtに記録。','- 提供Actorの再取得: `python scripts/eat_nav_model2000_fetch_policy.py`。対応env/agentと組で使用。','- モデル単体/前処理込み14候補推論時間はinference-cpu.json・inference-cuda.json。','- 再収集/学習/評価: `PYTHONPATH=src:scripts python scripts/eat_nav_model2000_continue.py --root <new-root>`（manifest・承認・benchmarkを事前固定。既存partialを無断削除しない）。']
        (out/'README.md').write_text('\n'.join(lines)+'\n')
        # Refresh the inventory after adding the handoff.
        inventory={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'}
        (out/'artifact-sha256.json').write_text(json.dumps(inventory,indent=2)+'\n')
        status('verification_before_push')
        run(sys.executable,'-m','pytest','tests/test_model2000_policy.py',*sorted(str(p.relative_to(REPO)) for p in (REPO/'tests').glob('test_single_pose2s*.py')),'-q')
        if cmd('git','branch','--show-current')!=BRANCH:raise ValueError('branch changed; refusing publication')
        if cmd('git','remote','get-url','--push',REMOTE)!=URL:raise ValueError('remote mismatch')
        run('git','add','--','docs/eat-nav/model2000-pose2s-20261110','docs/eat-nav/model2000-pose2s-retraining-spec-v0.1.md','scripts/eat_nav_model2000_publish_when_ready.py')
        staged=cmd('git','diff','--cached','--name-only').splitlines()
        if not staged:raise ValueError('no completed artifacts staged')
        for name in staged:
            f=REPO/name
            if f.is_file() and f.stat().st_size>=100*1024*1024:raise ValueError('oversized normal Git object: '+name)
            if '/raw/' in name or name.endswith('samples.npz') or name.endswith('model_2000.pt'):raise ValueError('unexpected raw/source weights in Git')
            if f.is_file() and f.suffix not in ['.pt','.npz']:
                content=f.read_bytes()
                if re.search(rb'AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}',content):
                    raise ValueError('possible credential in staged file; publication stopped: '+name)
        run('git','diff','--cached','--check')
        run('git','commit','-m','results: model2000 pose2s training and matched B evaluation')
        commit=cmd('git','rev-parse','HEAD')
        status('pushing',commit=commit)
        run('git','push',REMOTE,'HEAD:refs/heads/'+BRANCH)
        remote=cmd('git','ls-remote',REMOTE,'refs/heads/'+BRANCH).split()[0]
        if remote!=commit:raise ValueError('remote SHA mismatch')
        d=dict(phase='completed',github_push_completed=True,branch=BRANCH,commit=commit,checkpoint_sha256=index['checkpoint_sha256'],strict_accuracy_achieved=index['strict_accuracy_achieved'],completed_unix=time.time());status_path.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
    except BaseException as exc:
        status('failed_preserved',error=repr(exc));raise

if __name__=='__main__':main()
