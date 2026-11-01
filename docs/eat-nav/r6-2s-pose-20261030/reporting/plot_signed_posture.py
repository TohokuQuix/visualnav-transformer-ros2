"""Descriptive signed validation posture/speed only; no fitting or test/bag input."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root,output):
    source=root/'final-dataset-audit/validation-records.json'
    records=json.loads(source.read_text())
    if any(r['split']!='validation' for r in records):
        raise ValueError('validation-only descriptive records required')
    rows=[r for r in records if r['complete']]
    if not rows:raise ValueError('nonempty observed validation records required')
    def signed_peak(row,axis):
        values=np.asarray([row[axis+'_min_deg'],row[axis+'_max_deg']])
        return float(values[np.argmax(np.abs(values))])
    def speed(row):return float(np.linalg.norm(np.asarray(row['initial_body_state'])[3:5]))
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.5),layout='constrained')
    features=[(lambda r:signed_peak(r,'pitch'),'Signed largest-|pitch| sample (deg)'),
              (lambda r:signed_peak(r,'roll'),'Signed largest-|roll| sample (deg)'),
              (speed,'Initial body-frame planar speed (m/s)')]
    for ax,(feature,label) in zip(axes,features):
        for failed,color in ((False,'#2864a0'),(True,'#bd492e')):
            values=np.sort([feature(r) for r in rows if bool(r['failed'])==failed])
            if not np.isfinite(values).all():raise ValueError('nonfinite descriptive values')
            if len(values):ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,
                label=f"{'event observed' if failed else 'event-free 2s'} (n={len(values)})")
        ax.set(xlabel=label,ylabel='Empirical cumulative fraction',ylim=(0,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    cuts=np.quantile([speed(r) for r in rows],[.25,.5,.75])
    for cut in cuts:axes[2].axvline(cut,linestyle=':',color='#777777',linewidth=.9)
    fig.suptitle('Validation signed posture and initial speed (actual 2-second outcomes)\n'
        'Correlated branches; event-free is not 2m climbing success or an inferred fall boundary',fontsize=11)
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(output/f'signed-posture-speed.{suffix}',dpi=180)
    plt.close(fig)
    metadata=dict(schema='eatnav_r6_signed_posture_figures_v1',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        rows=len(rows),positive=sum(bool(r['failed']) for r in rows),episode_groups=len({r['group_id'] for r in rows}),
        signed_peak_rule='signed value at largest absolute extremum of observed future roll/pitch; min chosen on ties',
        speed_definition='sqrt(initial body_state[3]^2 + body_state[4]^2), m/s',speed_quartiles_mps=cuts.tolist(),
        libraries=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['No validationtip/near positives; event outcomes are stall dominated','Unequal class sizes are normalized separately for descriptive CDFs, not a conditional failure probability','Repeated candidates/anchors share sourcegroups; no independent-safety claim','No thresholds or modelweights fitted; no test/bag reads'])
    (output/'signed-posture-sources.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.root,a.output)
