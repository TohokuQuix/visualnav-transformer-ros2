"""Retrieve the pinned paired Actor without mixing moving main-branch files."""
from pathlib import Path
import argparse,hashlib,urllib.request
from visualnav_transformer.eat_nav.model2000_policy import HASHES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'checkpoints/eat_nav/rl_actor_model2000')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    base='https://raw.githubusercontent.com/TohokuQuix/onix_automation/a43de8de76fcfe2113eac8708c9b21a36f2af2bd/onix_policy_bridge/'
    for name,expected in HASHES.items():
        dst=a.output/name
        if dst.exists():
            if hashlib.sha256(dst.read_bytes()).hexdigest()!=expected:raise ValueError('existing bundle mismatch; preserved '+str(dst))
        else:
            remote=('checkpoints/' if name.endswith('.pt') else 'config/')+'onix_rough_standard/'+name
            with urllib.request.urlopen(base+remote,timeout=120) as response:data=response.read()
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('download identity mismatch '+name)
            tmp=dst.with_suffix(dst.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(dst)
        print(name+' '+expected)

if __name__=='__main__':main()
