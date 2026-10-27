"""Wait for the fixed R4 suite, then independently verify and summarize it."""
from __future__ import annotations
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eat_nav_r4_run import ROOT,BASE,REPO
from eat_nav_r4_audit import audit_evaluation
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,verify_derived,seal_derived,require

DEFAULT=REPO/'docs/eat-nav/r4-linear-only-evaluation-review-20261026'

def finish(root,output):
    torch.set_num_threads(1);verify_derived(root);verify_derived(BASE)
    manifest=read(root/'manifest.json');results=read(root/'suite-results.json')
    require(len(results)==len(manifest['runs'])==36,'incomplete R4 suite')
    checks=[]
    for e in results:
        if e['state']!='completed':continue
        item=e['assignment'];require(sha(Path(item['model']))==item['model_sha256'],'checkpoint changed')
        audit=audit_evaluation(root/e['folder'],item['model'])
        require(audit==e['audit'],'audit replay changed: '+e['folder'])
        checks.append(dict(folder=e['folder'],eligible=e['comparison_eligible'],replay_errors=audit['replay_errors'],decisions=audit['inference_decisions']))
    a=read(root/'aggregates.json');kr={x['condition']:x for x in a if x['field']=='k_rail_field'}
    prior=read(BASE/'suite-results.json');matched={(e['assignment']['field'],e['assignment']['seed'],e['assignment']['model_seed']) for e in results if e.get('comparison_eligible')}
    elapsed=[]
    for field in manifest['fields']:
        for condition in ('R1','R2','R3','R4'):
            xs=[e for e in (results if condition=='R4' else prior) if e['assignment']['field']==field and e['assignment']['condition']==condition and e.get('comparison_eligible') and (field,e['assignment']['seed'],e['assignment']['model_seed']) in matched]
            if xs:elapsed.append(dict(field=field,condition=condition,mean_observed_sec=float(np.mean([e['result']['control_steps']/20 for e in xs]))))
    write(output/'verification.json',dict(suite_root=str(root),suite_ledger_sha256=sha(root/'sha-ledger.json'),
        completed_runs=len(checks),eligible_runs=sum(e.get('comparison_eligible',False) for e in results),failed_runs=sum(e['state']=='failed' for e in results),
        audits=checks,models_retrained=False,reserved_test_opened=False,mean_observed_duration=elapsed))
    lines=['# R4追加比較の結果・終了後監査','',
        f"36予定を全件処理。走行処理完了{len(checks)}、比較有効{sum(e.get('comparison_eligible',False) for e in results)}、実行失敗{sum(e['state']=='failed' for e in results)}。既存12モデルをそのまま使用し、再学習はしていない。",'',
        'R4は直線スタック予測のみ有効、旋回/転倒/near-fall予測と実測45度STOP/35度復帰を無効化。候補保持・切替マージン・前後進確認・予測STOP時の最短2秒/2回確認・Actor/追従/goal条件等は共通。','',
        '## K-Railの対応比較','',
        '| 条件 | 比較数 | 転倒観測走行 | 平均route進捗m | 平均STOP秒 | 平均観測時間秒 |','|---|---:|---:|---:|---:|---:|']
    for mode in ('R1','R2','R3','R4'):
        if mode not in kr:continue
        x=kr[mode];duration=next(v['mean_observed_sec'] for v in elapsed if v['field']=='k_rail_field' and v['condition']==mode)
        lines.append(f"| {mode} | {x['runs']} | {x['tip_over_runs']} | {x['mean_progress_m']:.3f} | {x['mean_stop_sec']:.2f} | {duration:.2f} |")
    if 'R4' in kr and 'R2' in kr:
        x,y=kr['R4'],kr['R2']
        lines+=['',f"R2→R4の実測差: 転倒観測{x['tip_over_runs']-y['tip_over_runs']:+d}走行、平均進捗{x['mean_progress_m']-y['mean_progress_m']:+.3f}m、平均STOP{x['mean_stop_sec']-y['mean_stop_sec']:+.2f}秒。差は同じ有効case集合で計算した。",'',
            f"K-Railの独立した走行seedは{x['rollout_seeds']}個。{x['runs']}比較を独立した地形seed数へ読み替えない。R4は旋回riskと姿勢STOPを同時に外しており、45度STOPだけの因果効果は分離していない。"]
        fig,axes=plt.subplots(1,3,figsize=(12,4));modes=[m for m in ('R1','R2','R3','R4') if m in kr]
        for ax,metric,label in zip(axes,('tip_over_runs','mean_progress_m','mean_stop_sec'),('Observed tip-over runs','Mean route progress (m)','Mean STOP time (s)')):
            bars=ax.bar(modes,[kr[m][metric] for m in modes],color=['#777777','#168f79','#d55e00','#0072b2'][:len(modes)])
            ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
            for bar,m in zip(bars,modes):
                val=kr[m][metric];text=f"{val}/{kr[m]['runs']}" if metric=='tip_over_runs' else f'{val:.2f}'
                ax.annotate(text,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
            ax.margins(y=.2)
        fig.suptitle('K-Rail | matched cases; optimization seeds repeat the same rollout seeds');fig.tight_layout();fig.savefig(output/'krail-comparison.png',dpi=160);plt.close(fig)
        lines+=['','![K-Rail比較](krail-comparison.png)']
    lines+=['','## 判断上の制約','',
        '- STOP減少だけでは改善としない。早期転倒で観測時間が短くなる場合を含め、転倒・最終進捗・観測時間・動作中stallを併せて読む。',
        '- 新規K-Rail validation不成立のため既存validationで学習epochを選んだ制約、既存PhysX初期化警告、development seedの少なさは引き継ぐ。',
        '- 起動時settling失敗はモデルの候補選択前のため、転倒リスク回避の性能へ加算しない。seedを入れ替えていない。',
        '- 観測結果を見てモデル・閾値を変更していない。reserved test・実機評価は未実施。',
        '- 主実験rootはsealedのまま保持し、本directoryで全checkpoint SHAと完了走行の保存入力推論・全guard状態再現を再確認した。',
        '',f"[全field・全caseと動作中stallの比較](../{root.name}/comparison-report.md)、[終了後監査](verification.json)。"]
    (output/'result-report.md').write_text('\n'.join(lines)+'\n');seal_derived(output);verify_derived(output)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--wait',action='store_true');args=p.parse_args();output=args.output.resolve();require(not output.exists(),'fresh review output required');output.mkdir(parents=True)
    shutil.copy2(__file__,output/'finish.py')
    try:
        if args.wait:
            deadline=time.monotonic()+8*3600
            while not (args.root/'derived-seal.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('R4 suite did not finish within8hours; inspect progress and runner log')
                time.sleep(2)
            time.sleep(1)
        finish(args.root.resolve(),output)
        print('R4_FINAL_REVIEW_COMPLETE '+str(output/'result-report.md'),flush=True)
    except Exception as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
