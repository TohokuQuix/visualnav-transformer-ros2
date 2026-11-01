"""Plot three observed validation terminal examples; no inference, fitting or test/bag."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eat_nav_r6_storage import read_trace
from visualnav_transformer.eat_nav.fall_stall_targets import post_state,tilt


def render(root,output):
    examples=[('pose_q1_1',0),('pose_q1_1',4),('pose_q0_2',18)]
    fig,axes=plt.subplots(3,1,figsize=(11.5,9),layout='constrained')
    provenance=[]
    for ax,(condition,env) in zip(axes,examples):
        base=root/'validation-evaluation'/f'{condition}-k_rail_field-l0-batch00'/'sim-output'
        folder=base/f'group-{env:03d}'
        read=lambda p:json.loads(p.read_text())
        group=read(folder/'group.json');result=read(folder/'result.json')
        assert group['role']=='validation' and group['initial_valid'] and result['tip_count']==1
        storage=read(base/'trace-storage.json')[f'group-{env:03d}/trace.jsonl']
        trace=folder/'trace.jsonl.gz'
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==storage['compressed_sha256']
        rows=read_trace(folder/'trace.jsonl')
        times=np.asarray([(r['control_step']+1)*.05 for r in rows]);angles=np.asarray([tilt(post_state(r)) for r in rows])
        assert np.isfinite(angles).all() and len(rows)==round(result['observed_sec']/.05)
        ax.plot(times,angles,color='#174c75',linewidth=1.1,label='Observed gravity tilt (20Hz)')
        stop=np.asarray([bool(r['protective_stop']) for r in rows]);left=None
        for i,active in enumerate(np.r_[stop,False]):
            if active and left is None:left=i*.05
            if not active and left is not None:
                ax.axvspan(left,i*.05,color='#969696',alpha=.25,linewidth=0);left=None
        predictions=[];decision_sources={}
        for file in sorted(folder.glob('decision-*.json')):
            d=read(file);slot=d['selected_index'];decision_sources[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            if slot is None:continue
            assert all(t<=d['decision_stamp_ns'] for t in d['dem_stamps_ns'])
            a=np.asarray(d['predicted_trajectory'][slot],float);assert a.shape==(20,9) and np.isfinite(a).all()
            peak=float(np.degrees(np.arccos(np.clip(a[:,6]*a[:,8],-1,1))).max())
            predictions.append((d['decision_step']*.05,peak))
        if predictions:
            a=np.asarray(predictions);ax.scatter(a[:,0],a[:,1],s=24,color='#d47415',zorder=4,label='Forecast max over next 2s (at decision)')
        ax.axhline(85,color='#a82b2b',linestyle='--',linewidth=.9,label='85 deg observed tip criterion')
        ax.scatter(times[-1],angles[-1],marker='x',color='#a82b2b',s=50,zorder=5)
        seed=group['group_id'].split('/seed')[1].split('/')[0]
        ax.set(title=f'{condition} | K-Rail level 0 | parent seed {seed} | terminal {times[-1]:.2f}s',xlabel='Logical control time (s)',ylabel='Gravity tilt (deg)',xlim=(0,times[-1]+.7),ylim=(0,125))
        ax.grid(alpha=.18);ax.legend(loc='upper left',fontsize=8)
        provenance.append(dict(condition=condition,group_id=group['group_id'],observed_rows=len(rows),source_folder=str(folder),compressed_trace_sha256=storage['compressed_sha256'],decision_sha256=decision_sources))
    fig.suptitle('Observed terminal examples from incomplete validation\nGrey: requested protective STOP; fixed low-level RL and command latency stay active. No causal claim.',fontsize=11)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(output/f'terminal-tilt-examples.{ext}',dpi=180)
    plt.close(fig)
    (output/'terminal-tilt-sources.json').write_text(json.dumps(dict(schema='eatnav_r6_terminal_example_figure_v1',examples=provenance,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),libraries=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),limitations=['Three selected observed failures, not a random sample or comparative rate estimate','Forecast dots are maxima over next2s, not current-pose forecasts or observed future after terminal','Grey shows requested STOP, not instantaneous zero Actor input: existing command-latency interpolation can retain prior input in first50ms control interval; no physical rest or causal guarantee','Terminal pre-reset state retained; no future imputation or new inference/fitting','No test/bag reads; no runtime/model/profile changes']),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.root,a.output)
