from pathlib import Path
import hashlib,json,csv
import numpy as np
O=Path(__file__).resolve().parent
P=Path('/home/ubuntu/eatnav-student-share-20261018')
with np.load(P/'training-evidence/grouped.npz') as a:
 rows=[]
 for i in range(14):
  y=np.rad2deg(a['yaw'][a['mask'][:,i],i].astype(np.float64))
  rows.append(dict(candidate_id=i,n=len(y),mean_yaw_deg=float(y.mean()),median_yaw_deg=float(np.median(y))))
with (O/'original-oct14-yaw-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(grouped_npz_sha256=hashlib.sha256((P/'training-evidence/grouped.npz').read_bytes()).hexdigest(),actor_sha256=hashlib.sha256((P/'rl_actor_model_10000.pt').read_bytes()).hexdigest(),actor_env_sha256=hashlib.sha256((P/'rl_actor_env.yaml').read_bytes()).hexdigest(),source_handoff=(P/'SOURCE_COMMIT.txt').read_text(),units='degrees; original grouped yaw radians converted',scope='old observed simulator yaw, not newly simulated outcomes',rows=rows)
(O/'original-shared-evidence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
