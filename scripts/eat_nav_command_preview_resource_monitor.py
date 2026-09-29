"""Detached resource sample for the fixed long-horizon runner."""
from __future__ import annotations
import csv,subprocess,time
from pathlib import Path
ROOT=Path('/tmp/eatnav-candidate-coverage/command-preview-long-horizon/long-horizon')
ROOT.mkdir(parents=True,exist_ok=True)
with (ROOT/'resource-samples.csv').open('a',newline='') as f:
 w=csv.writer(f)
 if f.tell()==0:w.writerow(['unix_sec','sim_rss_mib','gpu_used_mib'])
 while not (ROOT/'stage-b-runs-complete.json').exists():
  try:
   data=subprocess.check_output(['ps','-eo','rss,args'],text=True)
   rss=max((int(line.split(None,1)[0]) for line in data.splitlines()[1:] if '/isaac-sim/kit/python/bin/python3' in line and '/long_sim.py' in line),default=0)/1024
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0].strip()
   w.writerow([time.time(),round(rss,2),gpu]);f.flush()
  except Exception as exc:w.writerow([time.time(),'ERROR',repr(exc)]);f.flush()
  if (ROOT/'status.json').exists():
   import json
   if json.loads((ROOT/'status.json').read_text()).get('state') in ('stopped','budget_stop','complete'):break
  time.sleep(5)
