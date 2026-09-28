"""Reproduce the eight remediation diagnostic figures from frozen JSON results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/phase-c-remediation'
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)

def read(name):return json.loads((ROOT/name).read_text())
def save(name):
    plt.tight_layout();plt.savefig(OUT/name,dpi=160,bbox_inches='tight');plt.close()

history=read('reference-resume-history.json')
plt.figure(figsize=(8,4));plt.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='train')
plt.plot([x['epoch'] for x in history],[x['validation_loss'] for x in history],label='validation')
plt.axvline(120,color='k',ls=':',label='best epoch 120');plt.xlabel('epoch');plt.ylabel('diffusion loss');plt.legend();save('reference-learning-curve.png')

first=read('first-divergence-report.json')
names=['command','pre_noise','post_noise','actor_input','actor_latent','actor_action','processed_action','root_state','contact_force','raycast','critic_obs','reward']
steps=[first[n]['first_step'] if first[n]['first_step'] is not None else 41 for n in names]
plt.figure(figsize=(9,5));plt.barh(names,steps);plt.xlim(0,42);plt.xlabel('first control step above frozen tolerance (41 = none)');save('first-divergence-by-signal.png')

controls=[('baseline',first),('A command',read('recorded-command-control.json')),('B Actor input',read('recorded-observation-control.json')),('C 6-D action',read('recorded-action-control.json'))]
fixed=read('fixed-branch-repeat.json');serial=read('serial-replay-validation.json')
labels=[x[0] for x in controls]+['D same branch','D parallel','serial same env']
values=[x[1]['xy']['rms'] for x in controls]+[fixed['signals']['xy']['same_branch_across_run_rms'],fixed['signals']['xy']['parallel_branch_rms'],serial['post_anchor']['xy']['rms_to_first']]
plt.figure(figsize=(9,4));plt.bar(labels,values);plt.xticks(rotation=35,ha='right');plt.ylabel('same-Path XY trajectory RMS [m]');save('controls-a-to-d-duplicate-rms.png')

plt.figure(figsize=(6,4));plt.bar(['parallel clone\nrecorded action','same-env serial\nfixed Path'],[read('recorded-action-control.json')['xy']['rms'],serial['post_anchor']['xy']['rms_to_first']]);plt.ylabel('duplicate XY RMS [m]');save('parallel-vs-serial.png')

offline=read('candidate-diversity-offline.json');candidate=read('candidate-diversity.json');threshold=read('candidate-thresholds.json')
path_rms=[v for row in offline['rows'] for v in row['pairwise']['trajectory_rms']]
endpoint=[v for row in offline['rows'] for v in row['pairwise']['endpoint_distance_m']]
plt.figure(figsize=(8,4));plt.hist(path_rms,bins=35,alpha=.7,label='trajectory RMS');plt.hist(endpoint,bins=35,alpha=.5,label='endpoint distance');plt.axvline(threshold['path_rms_min_m'],ls=':',color='k',label='frozen 0.05 m threshold');plt.xlabel('candidate pair difference [m]');plt.ylabel('pair count');plt.legend();save('natural-history-diversity.png')

angular=[v for row in candidate['rows'] for v in row['follower_angular_pairwise_difference_radps']]
plt.figure(figsize=(6,4));plt.scatter(path_rms,angular,s=8,alpha=.45);plt.axvline(threshold['path_rms_min_m'],ls=':',color='k');plt.axhline(threshold['command_angular_min_radps'],ls=':',color='gray');plt.xlabel('candidate Path RMS [m]');plt.ylabel('follower angular command difference [rad/s]');save('path-vs-follower-command.png')

terrain=sorted({r['state']['terrain_type'] for r in candidate['rows']})
by_type={t:[r for r in candidate['rows'] if r['state']['terrain_type']==t] for t in terrain}
plt.figure(figsize=(9,4));x=np.arange(len(terrain));plt.bar(x-.2,[sum(r['path_cluster_count']==1 for r in by_type[t])/len(by_type[t]) for t in terrain],width=.4,label='Path collapse');plt.bar(x+.2,[sum(r['command_cluster_count']==1 for r in by_type[t])/len(by_type[t]) for t in terrain],width=.4,label='command collapse');plt.xticks(x,terrain,rotation=35,ha='right');plt.ylim(0,1.05);plt.ylabel('state fraction (n=4–5 per terrain)');plt.legend();save('terrain-state-collapse.png')

comparison=ROOT/'duplicate-vs-candidate.json'
if comparison.exists():
    d=json.loads(comparison.read_text())
    plt.figure(figsize=(6,4));plt.bar(['serial duplicate','candidate observed'],[d['duplicate_xy_rms_m'],d['candidate_observed_xy_rms_m']]);plt.ylabel('XY trajectory RMS [m]');save('duplicate-vs-candidate.png')
print(sorted(p.name for p in OUT.glob('*.png')))
