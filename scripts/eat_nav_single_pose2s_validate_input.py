"""Validate offline current-only single-DEM inputs; never publish robot commands."""
import argparse,json
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.single_pose2s_contract import INPUT_SCHEMA,PROFILE_SHA,validate_input
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    with np.load(args.input,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    if str(np.asarray(a.get('input_schema','')).item())!=INPUT_SCHEMA or str(np.asarray(a.get('profile_sha256','')).item())!=PROFILE_SHA:raise ValueError('input provenance mismatch')
    if not len(a['dem']):raise ValueError('empty batch')
    for i in range(len(a['dem'])):
        sample={k:v[i] for k,v in a.items() if k not in ['q0_wxyz','input_schema','profile_sha256']};sample['schema']=INPUT_SCHEMA;validate_input(sample)
    print(json.dumps(dict(valid=True,schema=INPUT_SCHEMA,profile_sha256=PROFILE_SHA,batch=len(a['dem']),candidates=a['paths'].shape[1],DEM_frames=1,DEM_update_hz=4),indent=2))
