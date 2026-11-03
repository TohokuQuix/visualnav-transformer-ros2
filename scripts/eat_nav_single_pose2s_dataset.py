"""Build the approved single-frame dataset and audit without opening final results."""
import argparse,json
from pathlib import Path
from visualnav_transformer.eat_nav.single_pose2s_dataset import build_dataset
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=build_dataset(a.raw_root,json.loads(a.manifest.read_text()),a.output)
    print(json.dumps({k:report[k] for k in ['planned_parents','observed_parents','planned_windows','observed_windows','valid_pose_windows','complete_parent_coverage']}))
