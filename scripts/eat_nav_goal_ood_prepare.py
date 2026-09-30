"""Read-only frozen source audit and train-only goal support fitting."""
from __future__ import annotations
import collections,gzip,hashlib,json,math,pathlib,time
import numpy as np,yaml
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView

REPO=pathlib.Path('/tmp/eatnav-vnt-goal-ood')
DOC=REPO/'docs/quality-identifiability-gate/candidate-coverage/goal-ood'
RAW=pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood')
MANIFEST=RAW/'input-copy/manifest.json'
REF=REPO/'docs/quality-identifiability-gate/phase-c'
CHECKPOINT=pathlib.Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def write(path,value,mutable=False):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists() and not mutable:
        if path.read_text()!=content:raise FileExistsError(str(path))
        return
    temp=path.with_name(path.name+'.tmp');temp.write_text(content);temp.replace(path)
def polar(goals,edges,bins):
    g=np.asarray(goals).reshape(-1,2);r=np.linalg.norm(g,axis=1);a=np.arctan2(g[:,1],g[:,0]);
    return np.clip(np.searchsorted(edges,r,side='right')-1,0,len(edges)-2),np.floor(((a+np.pi)%(2*np.pi))*bins/(2*np.pi)).astype(int)%bins
def fit_support(goals,episodes,settings):
    edges=settings['polar_radius_edges_m'];bins=settings['polar_angle_bins'];ri,ai=polar(goals,edges,bins)
    members=collections.defaultdict(set)
    for i,key in enumerate(episodes):members[(int(ri[i]),int(ai[i]))].add(tuple(key))
    distances=cKDTree(goals).query(goals,k=settings['knn_k']+1,workers=-1)[0][:,-1]
    q95,q99=np.quantile(distances,[settings['knn_near_quantile'],settings['knn_far_quantile']])
    if not 0<q95<=q99:raise ValueError('degenerate train-only kNN support')
    return {'radius_edges_m':edges,'angle_bins':bins,'min_episodes':settings['minimum_independent_episodes_per_bin'],
        'k':settings['knn_k'],'near_quantile':settings['knn_near_quantile'],'far_quantile':settings['knn_far_quantile'],
        'near_distance_m':float(q95),'far_distance_m':float(q99),
        'polar_episode_support':{f'{i}:{j}':len(members.get((i,j),())) for i in range(len(edges)-1) for j in range(bins)},
        'train_self_knn_quantiles_m':np.quantile(distances,[0,.5,.9,.95,.99,1]).tolist()}
def classify(goals,train,support):
    g=np.asarray(goals).reshape(-1,2);d=cKDTree(train).query(g,k=support['k'],workers=-1)[0][:,-1]
    ri,ai=polar(g,support['radius_edges_m'],support['angle_bins']);labels=[]
    for distance,i,j in zip(d,ri,ai):
        count=support['polar_episode_support'][f'{i}:{j}']
        if 0<count<support['min_episodes']:label='unsupported'
        elif count>=support['min_episodes'] and distance<=support['near_distance_m']:label='ID'
        elif distance<=support['far_distance_m']:label='near_ood'
        else:label='far_ood'
        labels.append(label)
    return labels,d.tolist()
def corr(a,b):
    value=float(spearmanr(a,b).statistic)
    return value if np.isfinite(value) else None

def main():
    started=time.monotonic();setting=yaml.safe_load((DOC/'goal-ood-config.yaml').read_text())
    expected={'manifest':(MANIFEST,setting['manifest_sha256']),
        'checkpoint':(CHECKPOINT,setting['reference_checkpoint_sha256']),
        'config':(REF/'reference-eat.yaml','caa88735e765fc0358930dfa41579ecaac9eb96dc4052dc01a81390377279992'),
        'complete_ids':(REF/'complete-window-ids.jsonl.gz',setting['complete_ids_sha256'])}
    for name,(path,wanted) in expected.items():
        if sha(path)!=wanted:raise ValueError(name+' SHA mismatch')
    manifest=read(MANIFEST)
    for chunk in manifest['chunks']:
        path=MANIFEST.parent/chunk['file']
        if path.stat().st_size!=chunk['bytes'] or sha(path)!=chunk['sha256']:raise ValueError('chunk SHA mismatch '+chunk['file'])
    cfg=load_eat_config(REF/'reference-eat.yaml')
    if cfg.quality_enabled or cfg.len_traj_pred!=8 or cfg.context_size!=3 or cfg.goal_max_radius_m!=.8:raise ValueError('reference config drift')
    trainset=EATDataset(MANIFEST,'train',cfg,split_seed=123);train=CompleteOnlyView(trainset,8)
    valset=EATDataset(MANIFEST,'val',cfg,split_seed=123);val=CompleteOnlyView(valset,8)
    if (len(train),len(val))!=(49598,6171):raise ValueError('complete count drift')
    split_hash=hashlib.sha256(json.dumps(trainset.split_map,sort_keys=True).encode()).hexdigest()
    if split_hash!=setting['split_hash'] or trainset.split_map!=valset.split_map:raise ValueError('split hash drift')
    with gzip.open(REF/'complete-window-ids.jsonl.gz','rt') as f:frozen={(r['split'],tuple(r['episode_key']),r['step']) for r in map(json.loads,f)}
    actual={(name,tuple(w['episode_key']),int(w['step'])) for name,view in (('train',train),('val',val)) for w in (view.dataset.windows[i] for i in view.indices)}
    if frozen!=actual or len(actual)!=len(train)+len(val):raise ValueError('complete ID mismatch')
    windows=[trainset.windows[i] for i in train.indices]
    goals=np.asarray([w['goal'] for w in windows],dtype=np.float32);actions=np.asarray([w['action'] for w in windows],dtype=np.float32)
    endpoints=actions.cumsum(axis=1)[:,-1];rpre=np.linalg.norm(endpoints,axis=1)
    restored=endpoints*np.minimum(1.,.8/np.maximum(rpre,1e-20))[:,None]
    if not np.allclose(goals,restored,atol=2e-7,rtol=0):raise ValueError('goal clip reconstruction failed')
    mean=actions.reshape(-1,2).mean(axis=0).astype(np.float32);std=actions.reshape(-1,2).std(axis=0).astype(np.float32)
    audit=read(REF/'reference-data-audit.json')
    if not np.allclose(mean,audit['train_action_mean'],atol=1e-6) or not np.allclose(std,audit['train_action_std'],atol=1e-6):raise ValueError('train action normalization drift')
    episodes=[tuple(w['episode_key']) for w in windows];support=fit_support(goals,episodes,setting);labels,knn=classify(goals,goals,support)
    gr=np.linalg.norm(goals,axis=1);angle=np.arctan2(goals[:,1],goals[:,0]);rough=[];critic=[];curvature=[]
    for w in windows:
        dem=np.asarray(w['dem']);valid=dem[:,1]>.5
        rough.append(float(np.std(dem[:,0][valid])) if valid.any() else 0.)
        critic.append(float(w['quality_raw']));a=np.unwrap(np.arctan2(w['action'][:,1],w['action'][:,0]));curvature.append(float(np.abs(np.diff(a)).sum()))
    terrain={}
    types=np.asarray([w['terrain_type'] for w in windows])
    for kind in sorted(set(types)):
        ix=np.flatnonzero(types==kind)
        terrain[kind]={'windows':len(ix),'episodes':len({episodes[i] for i in ix}),
            'goal_radius_quantiles_m':np.quantile(gr[ix],[0,.1,.5,.9,1]).tolist(),
            'goal_angle_quantiles_rad':np.quantile(angle[ix],[0,.1,.5,.9,1]).tolist(),
            'support_classes':dict(collections.Counter(labels[i] for i in ix))}
    episode_stats={}
    for key in sorted(set(episodes)):
        ix=[i for i,e in enumerate(episodes) if e==key]
        episode_stats[str(key)]={'windows':len(ix),'goal_radius_median_m':float(np.median(gr[ix])),'goal_angle_median_rad':float(np.median(angle[ix]))}
    distribution={'complete_train_windows':len(train),'validation_windows':len(val),'train_episode_count':len(set(episodes)),
        'x_quantiles_m':np.quantile(goals[:,0],[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
        'y_quantiles_m':np.quantile(goals[:,1],[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
        'preclip_radius_quantiles_m':np.quantile(rpre,[0,.1,.25,.5,.75,.9,.99,1]).tolist(),
        'postclip_radius_quantiles_m':np.quantile(gr,[0,.1,.25,.5,.75,.9,.99,1]).tolist(),
        'angle_quantiles_rad':np.quantile(angle,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
        'clip_count':int(np.sum(rpre>.8)),'clip_rate':float(np.mean(rpre>.8)),
        'direction_fractions':{'forward':float(np.mean((goals[:,0]>=0)&(np.abs(goals[:,0])>=np.abs(goals[:,1])))),
            'backward':float(np.mean((goals[:,0]<0)&(np.abs(goals[:,0])>=np.abs(goals[:,1])))),
            'left':float(np.mean((goals[:,1]>0)&(np.abs(goals[:,1])>np.abs(goals[:,0])))),
            'right':float(np.mean((goals[:,1]<0)&(np.abs(goals[:,1])>np.abs(goals[:,0]))))},
        'goal_endpoint_distance_quantiles_m':np.quantile(np.linalg.norm(goals-endpoints,axis=1),[0,.5,.9,.99,1]).tolist(),
        'support_class_counts':dict(collections.Counter(labels)),
        'spearman':{'radius_roughness':corr(gr,rough),'radius_critic':corr(gr,critic),'radius_curvature':corr(gr,curvature),
            'angle_roughness':corr(angle,rough),'angle_critic':corr(angle,critic),'angle_curvature':corr(angle,curvature)},
        'terrain':terrain,'episodes':episode_stats,'polar_episode_support':support['polar_episode_support']}
    write(DOC/'source-provenance.json',{'branch':'exp/eatnav-goal-ood','base_commit':'49a0c5a770c3c918f7d9e0c2b30b444ccff93b60',
        'input_sha256':{name:sha(path) for name,(path,_) in expected.items()},'chunk_count':len(manifest['chunks']),
        'split_hash':split_hash,'model_metadata':cfg.checkpoint_metadata(),'train_action_mean':mean.tolist(),'train_action_std':std.tolist(),
        'heldout_used':False,'simulator_started':False})
    write(DOC/'goal-definition-audit.json',{'source':'dataset.py:_build_window','frame':'robot local x forward y left',
        'endpoint_source':'last valid future waypoint; complete-only uses eighth point at t+2.0s',
        'saved_goal':'post radial clip at 0.8m','preclip_reconstruction':'cumulative action endpoint',
        'max_reconstruction_error_m':float(np.linalg.norm(goals-restored,axis=1).max()),
        'nonmatching_endpoint_reason':'0.8m radial clipping for complete-only windows',
        'partial_terminal':'last valid future; invalid increments zero; zero-valid goal zero; both excluded from complete-only',
        'clip_rate':distribution['clip_rate']})
    write(DOC/'train-goal-distribution.json',distribution)
    write(DOC/'goal-support-thresholds.json',{'source_split':'complete_only_train',**support,
        'rule':'ID if polar episode support >=5 and kNN <= train q95; near if <= q99; far otherwise; 1..4 episodes unsupported',
        'angle_wrap':'modulo 2pi'})
    np.savez_compressed(RAW/'train-derived.npz',goals=goals,actions=actions,endpoints=endpoints,
        episode_keys=np.asarray(episodes,dtype=np.int64),terrain=types,dem_roughness=rough,critic=critic,curvature=curvature)
    write(RAW/'g1.done.json',{'stage':'G1_COMPLETE','source_sha256':sha(DOC/'source-provenance.json'),
        'support_sha256':sha(DOC/'goal-support-thresholds.json'),'train_derived_sha256':sha(RAW/'train-derived.npz')})
    write(RAW/'status.json',{'stage':'G1_COMPLETE','elapsed_sec':time.monotonic()-started},mutable=True)
    print('G1_COMPLETE',len(train),len(val),'clip_rate',distribution['clip_rate'],'class_counts',distribution['support_class_counts'],flush=True)
if __name__=='__main__':main()
