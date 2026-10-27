"""Figures from persisted v2 teachers and numerical cycle evidence."""
import json
import math

from .fall_stall_targets import path_arc, post_state, world_path, xy


def write_figures(output,records,differences,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory=output/'figures';directory.mkdir()
    tilt_records=[r for r in records if r['targets']['event_counts']['near_fall'] or r['targets']['event_counts']['tip_over']]
    if tilt_records:
        fig,axes=plt.subplots(len(tilt_records),2,figsize=(13,2.6*len(tilt_records)),squeeze=False)
        for pair,record in zip(axes,tilt_records):
            targets=record['targets'];time=[(i+1)/20 for i in range(targets['row_count'])]
            pair[0].plot(time,targets['diagnostics']['effective_tilt_deg'])
            pair[0].axhline(60,color='orange',ls='--');pair[0].axhline(85,color='red',ls='--')
            pair[1].plot(time,targets['diagnostics']['progress_m'])
            for event in targets['events']:
                if event['event']=='near_fall':
                    pair[0].axvline((event['onset_step']+1)/20,color='orange',ls=':')
                    pair[0].axvline(event['event_sec'],color='darkorange',label='confirmation')
                elif event['event']=='tip_over':pair[0].axvline(event['event_sec'],color='red')
            pair[0].set_title(f'{record["state_group"]} {record["candidate"]}',fontsize=8)
            pair[0].set_ylim(0,105);pair[0].set_ylabel('Tilt (deg)');pair[1].set_ylabel('Signed path progress (m)')
            for ax in pair:ax.set_xlim(0,10);ax.grid(alpha=.2);ax.set_xlabel('Post-action elapsed time (s)')
        fig.tight_layout();fig.savefig(directory/'tilt-confirmation-and-recovery.png',dpi=140);plt.close(fig)
    event_records=[r for r in records if any(r['targets']['event_counts'].values())]
    if event_records:
        fig,axes=plt.subplots(math.ceil(len(event_records)/3),3,figsize=(15,2.5*math.ceil(len(event_records)/3)),squeeze=False)
        for ax,record in zip(axes.flat,event_records):
            targets=record['targets']
            ax.plot([(i+1)/20 for i in range(targets['row_count'])],targets['diagnostics']['progress_m'])
            for event in targets['events']:
                color={'near_fall':'orange','tip_over':'black','linear_stall':'red','turn_stall':'purple'}[event['event']]
                ax.axvline(event['event_sec'],color=color,ls='--',alpha=.6)
            ax.set_title(f'{record["state_group"]}\n{record["candidate"]}',fontsize=7)
            ax.set_xlim(0,10);ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(event_records):]:ax.set_visible(False)
        fig.tight_layout();fig.savefig(directory/'all-event-trajectories.png',dpi=130);plt.close(fig)
    evidence=json.loads((output/'cycle-measurement-comparison.json').read_text())
    coarse=[r for r in evidence if r['diagnostic_best_rule_label']==1 and r['v2']['linear_stall']['label']==0]
    if coarse:
        fig,axes=plt.subplots(len(coarse),2,figsize=(13,2.5*len(coarse)),squeeze=False)
        for pair,record in zip(axes,coarse):
            rows=[json.loads(line) for line in (source/record['trace']).read_text().splitlines()]
            body=json.loads((source/record['trace']).with_name('input-manifest.json').read_text())['path_body_xy_m']
            cycle=record['v2'];chunk=rows[cycle['start_step']:cycle['end_step']+1]
            path=world_path(body,chunk[0]['root_state'])
            command,actual,best=0.,0.,0.
            start=path_arc(xy(chunk[0]['root_state']),path)
            commands,actuals,bests=[],[],[]
            for row in chunk:
                command+=abs(row['follower_command'][0])*.05
                actual+=math.dist(xy(row['root_state']),xy(post_state(row)))
                best=max(best,path_arc(xy(row['root_state']),path)-start,path_arc(xy(post_state(row)),path)-start)
                commands.append(command);actuals.append(actual);bests.append(best)
            time=[(i+1)/20 for i in range(len(chunk))]
            pair[0].plot(time,commands,label='command integral');pair[0].plot(time,actuals,label='actual XY distance')
            pair[0].legend(fontsize=7);pair[1].plot(time,bests,label='continuous best progress')
            pair[1].axhline(.05,color='red',ls='--');pair[1].set_ylabel('Best progress (m)')
            pair[0].set_title(f'{record["state_group"]} {record["candidate"]} cycle{record["cycle"]}',fontsize=8)
            for ax in pair:ax.grid(alpha=.2);ax.set_xlabel('Cycle post-action time (s)')
        fig.tight_layout();fig.savefig(directory/'coarse-best-counterexamples.png',dpi=140);plt.close(fig)
