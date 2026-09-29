"""Static report figures for frozen Stage A/B diagnostics."""
from __future__ import annotations
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visualnav_transformer.eat_nav.command_preview import commands_to_waypoints
ROOT=Path('/tmp/eatnav-candidate-coverage');OUT=ROOT/'command-preview-long-horizon';FIG=OUT/'figures'
def read(p):return json.loads(Path(p).read_text())
def finish(name):
 plt.tight_layout();plt.savefig(FIG/name,dpi=150);plt.close()
def main():
 FIG.mkdir(exist_ok=True)
 f=read(OUT/'preview-command-fidelity.json');e=read(OUT/'preview-evaluator-results.json')
 sid='coverage-001';cid='nomad-seed-1002';p=read(OUT/'preview-candidates'/f'{sid}--{cid}.json')
 with gzip.open(ROOT/'pilot/runs'/p['source_run_id']/'live-0.trace.json.gz','rt') as g:t=json.load(g)
 recorded=np.asarray([r['command'] for r in t]);real=np.asarray([r['post_root_state'][:2] for r in t])
 real=real-real[0]
 plt.figure(figsize=(6,5))
 for points,label in [(np.asarray(p['local_path']),'planned Path'),(np.asarray(p['closed_trajectory']),'preview command'),
                      (commands_to_waypoints(recorded),'recorded command'),(real[4::5],'realized XY')]:
  plt.plot(points[:,0],points[:,1],marker='.',label=label)
 plt.axis('equal');plt.xlabel('body x (m)');plt.ylabel('body y (m)');plt.legend();finish('path-preview-recorded-realized.png')
 groups={s:[r for r in f['candidate_rows'] if r['state_id']==s] for s in sorted({r['state_id'] for r in f['candidate_rows']})}
 names=list(groups);x=np.arange(len(names));plt.figure(figsize=(9,4))
 plt.bar(x-.2,[np.mean([r['closed']['linear_rmse_mps'] for r in groups[s]]) for s in names],.4,label='linear m/s')
 plt.bar(x+.2,[np.mean([r['closed']['angular_rmse_radps'] for r in groups[s]]) for s in names],.4,label='angular rad/s')
 plt.xticks(x,names,rotation=45,ha='right');plt.legend();plt.ylabel('command RMSE');finish('state-command-preview-error.png')
 metric=e['aggregate'];old=e['prior_aggregate']
 plt.figure(figsize=(7,4))
 plt.bar(['P','C-hat','C','R'],[old['P']['state_macro_pairwise_accuracy'],metric['state_macro_pairwise_accuracy'],
                               old['C']['state_macro_pairwise_accuracy'],old['R']['state_macro_pairwise_accuracy']])
 plt.axhline(.5,color='black',linestyle=':');plt.ylabel('state-macro pairwise accuracy');finish('gt-p-chat-c-r.png')
 plt.figure(figsize=(6,5))
 for s in e['states']:
  c=[r for r in e['candidate_rows'] if r['state_id']==s['state_id']]
  plt.scatter([r['delta_v'] for r in c],[r['gt_c_hat_mean'] for r in c],label=s['state_id'],s=12)
 plt.xlabel('true Delta V');plt.ylabel('GT(C-hat) predicted Delta V');plt.legend(fontsize=6);finish('true-vs-preview-predicted.png')
 plt.figure(figsize=(6,4))
 for s in e['states']:
  rs=groups[s['state_id']];r=np.mean([x['closed']['linear_rmse_mps']/.03+x['closed']['angular_rmse_radps']/.05 for x in rs])
  plt.scatter(r,s['gt_c_hat']['pairwise_accuracy']);plt.annotate(s['state_id'],(r,s['gt_c_hat']['pairwise_accuracy']),fontsize=7)
 plt.xlabel('normalized command RMSE');plt.ylabel('within-state pairwise accuracy');finish('fidelity-vs-ranking.png')
 if (OUT/'long-horizon-results.json').exists():
  h=read(OUT/'long-horizon-results.json');states=h['states'];x=np.arange(len(states));names=[s['state_id'] for s in states]
  plt.figure(figsize=(8,4))
  for j,label in enumerate(('nomad','bank')):
   plt.bar(x+(j-.5)*.3,[s['120'][label]['delta_v'] if 'delta_v' in s['120'][label] else np.nan for s in states],.3,label=label)
  plt.xticks(x,names);plt.ylabel('120-step Delta V');plt.legend();finish('nominal-nomad-bank-long-horizon.png')
  plt.figure(figsize=(8,4))
  for j,label in enumerate(('nomad','bank')):
   plt.plot(x,[s['40'][label]['delta_v'] if 'delta_v' in s['40'][label] else np.nan for s in states],marker='o',label=label+' 40')
   plt.plot(x,[s['120'][label]['delta_v'] if 'delta_v' in s['120'][label] else np.nan for s in states],marker='s',label=label+' 120')
  plt.xticks(x,names);plt.ylabel('Delta V');plt.legend();finish('40-vs-120-outcome.png')
  fig,axs=plt.subplots(2,2,figsize=(9,7))
  for ax,key,label in zip(axs.flat,('stack','maximum_tilt_rad','forward_goal_progress_m','drawdown'),('stack','tilt rad','progress m','drawdown')):
   for j,kind in enumerate(('nomad','bank')):
    ax.bar(x+(j-.5)*.3,[float(s['120'][kind][key]) if key in s['120'][kind] else np.nan for s in states],.3,label=kind)
   ax.set_xticks(x,names,rotation=30);ax.set_title(label)
  axs[0,0].legend();finish('stack-tilt-progress-drawdown.png')
 print('FIGURES_COMPLETE',len(list(FIG.glob('*.png'))))
if __name__=='__main__':main()
