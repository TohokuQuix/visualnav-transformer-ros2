"""Predeclared equal-radius angular and equal-angle radial paired checks."""
from __future__ import annotations
import numpy as np
from eat_nav_goal_ood_prepare import DOC,RAW,read,write,sha

def main():
    data=read(DOC/'goal-sweep-inputs.json');states=read(DOC/'state-selection.json')['states'];rows=[]
    pair_names=(('near_left_vs_right',6,5,'y'),('far_left_vs_right',10,9,'y'),('ID_long_vs_short',2,1,'radius'))
    for state in states:
        p=np.load(RAW/'inference'/f"{state['state_id']}.npz")['paths']
        row={'state_id':state['state_id'],'split':state['split'],'terrain':state['terrain'],'pairs':{}}
        for name,left,right,axis in pair_names:
            a,b=p[left,:,-1,:],p[right,:,-1,:]
            diff=a[:,1]-b[:,1] if axis=='y' else np.linalg.norm(a,axis=1)-np.linalg.norm(b,axis=1)
            row['pairs'][name]={'success_fraction':float(np.mean(diff>0)),
                'endpoint_difference_median_m':float(np.median(diff)),
                'same_seed_count':len(diff)}
        rows.append(row)
    summary={}
    for split in ('train','val'):
        subset=[r for r in rows if r['split']==split]
        summary[split]={name:{'states':len(subset),'state_macro_success_fraction':float(np.mean([r['pairs'][name]['success_fraction'] for r in subset])),
            'state_median_endpoint_difference_m':float(np.median([r['pairs'][name]['endpoint_difference_median_m'] for r in subset]))} for name,*_ in pair_names}
    write(DOC/'goal-direction-pairs.json',{'predeclared_pairs':[x[0] for x in pair_names],
        'same_goal_radius_angle_pairs':'near left/right both 0.1m; far left/right both 0.3m; ID short/long both 0deg',
        'states':rows,'summary':summary})
    print(summary,flush=True)
if __name__=='__main__':main()
