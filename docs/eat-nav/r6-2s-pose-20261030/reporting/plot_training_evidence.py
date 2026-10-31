"""Descriptive validation figures; no threshold fitting or test/bag reads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root,output):
 output.mkdir(parents=True,exist_ok=True)
 source=root/'final-dataset-audit/validation-records.json'
 records=json.loads(source.read_text());profiles=json.loads((root/'production-profiles/index.json').read_text())['profiles']
 if any(r['split']!='validation' for r in records):raise ValueError('validation-only descriptive evidence required')
 rows=[r for r in records if r['complete']]
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 fields=[('max_abs_pitch_deg','Max |pitch| (deg)'),('max_abs_roll_deg','Max |roll| (deg)'),('step_cm','Causal DEM step (cm)'),('max_pitch_rate_degps','Max |pitch rate| (deg/s)'),('max_roll_rate_degps','Max |roll rate| (deg/s)')]
 for ax,(key,label) in zip(axes.flat,fields):
  for failed,color in ((False,'#2864a0'),(True,'#bd492e')):
   values=np.asarray([r[key] for r in rows if r['failed']==failed and r[key] is not None]);values.sort()
   if len(values):ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,label=f"{'event observed' if failed else 'event-free 2s'} (n={len(values)})")
  ax.set(xlabel=label,ylabel='Empirical cumulative fraction',ylim=(0,1.02));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
 ax=axes.flat[-1]
 chosen={0,2,3,4,5,6,8,9,10,11}
 for i,p in enumerate(profiles):
  ax.scatter(p['pitch_deg'],p['roll_deg'],s=35+4*p['step_cm'],marker='o' if i in chosen else 'x',c='#2864a0' if i in chosen else '#777777')
  offset=(-5,8) if i//3==0 else (3,-12) if i//3==1 else (3,7)
  ax.annotate(p['id'].removeprefix('pose_'),(p['pitch_deg'],p['roll_deg']),xytext=offset,textcoords='offset points',ha='right' if i//3==0 else 'left',fontsize=8)
 ax.set(xlabel='Candidate pitch limit (deg)',ylabel='Candidate roll limit (deg)',title='12 derived settings; circles: 10 compared')
 ax.grid(alpha=.2);ax.margins(x=.15,y=.15)
 fig.suptitle('Validation 2-second event outcomes and threshold candidates\nCorrelated branches; event-free is not a 2m climb success or a fall-safety boundary',fontsize=12)
 for suffix in ('png','pdf'):fig.savefig(output/f'validation-distributions.{suffix}',dpi=180)
 plt.close(fig)
 main=json.loads((root/'final-model/metrics.json').read_text())['validation']['trajectory'];state=json.loads((root/'state-only-diagnostic/metrics.json').read_text())['validation']['trajectory']
 fig,axes=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
 for ax,keys,title,unit in [(axes[0],['ade_3d_m','fde_3d_m'],'Future position','Error (m)'),(axes[1],['rotation_geodesic_deg','max_tilt_mae_deg'],'Future orientation','Error (deg)')]:
  x=np.arange(2);ax.bar(x-.17,[main[k] for k in keys],.34,label='DEM + state',color='#2864a0');ax.bar(x+.17,[state[k] for k in keys],.34,label='State only',color='#c7822e');ax.set(xticks=x,xticklabels=['ADE','FDE'] if unit=='Error (m)' else ['Rotation','Max tilt'],ylabel=unit,title=title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Validation accuracy: DEM advantage not established',fontsize=12)
 for suffix in ('png','pdf'):fig.savefig(output/f'validation-model-comparison.{suffix}',dpi=180)
 plt.close(fig)
 metadata=dict(schema='eatnav_r6_descriptive_validation_figures_v1',records_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),complete_candidates=len(rows),failures=sum(r['failed'] for r in rows),independent_safety_claim=False,notes=['14branches/anchors share source groups; empirical CDFs descriptive','Validation has zero tip/near positives; observed events here do not establish a fall boundary','Threshold dot size represents global step threshold, not climb probability','No held-out test or model bag results read or used'])
 metadata['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,root/'production-profiles/index.json',root/'final-model/metrics.json',root/'state-only-diagnostic/metrics.json')}
 metadata['libraries']=dict(numpy=np.__version__,matplotlib=matplotlib.__version__)
 (output/'sources.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.root,a.output)
