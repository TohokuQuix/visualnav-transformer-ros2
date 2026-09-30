"""Exact DEM duplicates plus verified near-history goal diversity on a frozen train probe."""
from __future__ import annotations
import collections,hashlib,time
import numpy as np
from scipy.spatial import cKDTree
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from eat_nav_goal_ood_prepare import DOC,RAW,REF,MANIFEST,sha,read,write,polar

def main():
    started=time.monotonic();source=read(DOC/'source-provenance.json')
    if source['input_sha256']['manifest']!=sha(MANIFEST):raise ValueError('manifest drift')
    config=load_eat_config(REF/'reference-eat.yaml');view=CompleteOnlyView(EATDataset(MANIFEST,'train',config,split_seed=123),8)
    if len(view)!=49598:raise ValueError('train count')
    windows=[view.dataset.windows[i] for i in view.indices]
    groups=collections.defaultdict(list);embedding=np.zeros((len(windows),128),dtype=np.float32)
    for i,w in enumerate(windows):
        d=np.asarray(w['dem'],dtype=np.float32)
        groups[hashlib.sha256(d.tobytes()).hexdigest()].append(i)
        # 4 frames x 2 channels x 4x4 pooled grid, projected only for shortlist.
        small=d.reshape(4,2,4,5,4,5).mean(axis=(3,5)).reshape(-1)
        embedding[i]=small
    rng=np.random.default_rng(20261001);proj=rng.normal(size=(128,16)).astype(np.float32)/np.sqrt(16)
    tree=cKDTree(embedding@proj);probes=np.sort(rng.choice(len(windows),size=1024,replace=False));results=[]
    for i in probes:
        distances,candidates=tree.query((embedding[i]@proj)[None],k=65)
        a=np.asarray(windows[i]['dem']);am=a[:,1]>.5;near=[]
        for j in candidates[0]:
            if j==i:continue
            b=np.asarray(windows[j]['dem']);bm=b[:,1]>.5;union=np.logical_or(am,bm).sum()
            iou=float(np.logical_and(am,bm).sum()/union) if union else 0.
            overlap=am&bm
            rmse=float(np.sqrt(np.mean((a[:,0][overlap]-b[:,0][overlap])**2))) if overlap.any() else float('inf')
            if iou>=.8 and rmse<=.05:near.append(int(j))
        distinct=[j for j in near if tuple(windows[j]['episode_key'])!=tuple(windows[i]['episode_key'])]
        goals=np.asarray([windows[j]['goal'] for j in distinct+[i]])
        ri,ai=polar(goals,[0,.2,.4,.6,.8,1.1],12)
        results.append({'probe_index':int(i),'near_count':len(near),'cross_episode_near_count':len(distinct),
            'cross_episode_goal_bins':len(set(zip(ri.tolist(),ai.tolist()))),
            'cross_episode_goal_max_separation_m':float(np.max(np.linalg.norm(goals-goals[-1],axis=1)))})
    duplicates=[indices for indices in groups.values() if len(indices)>1]
    out={'source_split':'complete_only_train','window_count':len(windows),'exact_duplicate_groups':len(duplicates),
         'windows_in_exact_duplicate_groups':sum(len(x) for x in duplicates),
         'probe_count':len(results),'shortlist_k':64,'near_rule':{'mask_iou_min':.8,'height_rmse_max_m':.05},
         'near_search':'random projection shortlist then exact DEM verification; sampled probe, not exhaustive nearest-neighbor guarantee',
         'probes_with_cross_episode_near':sum(x['cross_episode_near_count']>0 for x in results),
         'probes_with_multiple_goal_bins':sum(x['cross_episode_goal_bins']>1 for x in results),
         'probe_results':results}
    write(DOC/'dem-goal-neighborhood.json',out)
    write(RAW/'dem-neighborhood.done.json',{'stage':'DEM_NEIGHBORHOOD_COMPLETE','sha256':sha(DOC/'dem-goal-neighborhood.json'),'elapsed_sec':time.monotonic()-started})
    print('DEM_NEIGHBORHOOD_COMPLETE',out['probes_with_cross_episode_near'],out['probes_with_multiple_goal_bins'],flush=True)
if __name__=='__main__':main()
