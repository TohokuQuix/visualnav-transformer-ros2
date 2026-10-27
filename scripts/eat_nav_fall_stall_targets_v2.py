"""Generate read-only-source v2 teachers and a sealed differential audit."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from visualnav_transformer.eat_nav.fall_stall_audit import audit_pilot


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh derived directory outside source evidence')
    parser.add_argument('--figures',action='store_true')
    args=parser.parse_args()
    print(json.dumps(audit_pilot(args.source,args.output,make_figures=args.figures),indent=2))
