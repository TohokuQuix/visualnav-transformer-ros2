#!/usr/bin/env python3
"""Render the pre-rollout P1 role gate; no candidate Outcome is loaded."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.eat_nav_curvature_attainable import sha
from scripts.eat_nav_role_aware_gate import OUTPUT, write


def render(root=OUTPUT):
    root=Path(root)
    gate=json.loads((root/'role-aware-terrain-gate.json').read_text())
    rows=gate['anchors']
    names=['flat-only','graded','transition','differential','unknown']
    data=np.array([[names.index(p['class_proxy']) for p in a['paths']] for a in rows])
    fig,ax=plt.subplots(figsize=(9,4.8))
    from matplotlib.colors import ListedColormap,BoundaryNorm
    colors=['#b8c4cc','#798a55','#ddae5e','#bf677b','#333333']
    ax.imshow(data,aspect='auto',cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(6)-0.5,5))
    ax.set_yticks(range(len(rows)),[a['anchor_id'] for a in rows])
    ax.set_xticks(range(5),[p['candidate_id'] for p in rows[0]['paths']])
    ax.set_xlabel('Frozen primitive Path')
    ax.set_title('Known 0–0.60 m Path-aligned terrain classifications')
    for i in range(len(rows)):
        for j in range(5):
            ax.text(j,i,names[data[i,j]][0].upper(),ha='center',va='center',color='white' if data[i,j]>=3 else '#222')
    fig.text(.06,.02,'F flat  ·  G graded  ·  T transition  ·  D differential; all 40 Paths known',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1])
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'01-path-terrain-roles.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5))
    roles=['R-differential','R-transition-common','R-graded-common','R-flat-control']
    counts=[sum(a['primary_role']==r for a in rows) for r in roles]
    ax.barh(range(4),counts,color=['#bf677b','#ddae5e','#798a55','#b8c4cc'])
    ax.set_yticks(range(4),roles)
    ax.set_xlim(0,5)
    ax.set_xlabel('Eligible primary-role anchors among eight')
    ax.invert_yaxis()
    for i,n in enumerate(counts):ax.text(n+.1,i,str(n),va='center')
    ax.set_title('N2 gate: graded and flat slots remain empty')
    fig.tight_layout()
    fig.savefig(folder/'12-gate-decision.png',dpi=180)
    plt.close(fig)
    files={str(p.relative_to(root)):{'sha256':sha(p),'size_bytes':p.stat().st_size}
           for p in root.rglob('*') if p.is_file() and p.name!='artifact-sha-ledger.json'}
    write(root/'artifact-sha-ledger.json',{'files':files})
    return counts


if __name__=='__main__':
    print(render())
