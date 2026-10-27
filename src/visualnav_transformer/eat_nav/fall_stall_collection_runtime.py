"""Read-only native preflight and staged-source parity around the frozen host."""
import json
import subprocess
from pathlib import Path
from .fall_stall_audit import read,sha,require,write
from .fall_stall_collection_contract import verify_sources
from .fall_stall_blind_verification import BlindVerificationError

CONTAINER='isaac-lab-onix'


def native(*args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout


def container_hashes(files,runner=native):
    code='import hashlib,json; files='+repr(files)+'; print(json.dumps({k:hashlib.sha256(open(v,"rb").read()).hexdigest() for k,v in files.items()},sort_keys=True))'
    return json.loads(runner('docker','exec',CONTAINER,'/isaac-sim/python.sh','-c',code).strip().splitlines()[-1])


def probe_runtime(contract):
    from scripts.eat_nav_motion_14_collection_host import sample_resources,CAP
    verify_sources(contract)
    inspection=json.loads(native('docker','inspect',CONTAINER))[0]
    require(inspection['State']['Running'] is True,'container not running')
    code='''import json,sys,pathlib,importlib.metadata as m,numpy,torch
versions={}
for name in ('isaaclab','isaaclab_rl','rsl-rl','onix-isaac-lab'):
    try:versions[name]=m.version(name)
    except m.PackageNotFoundError:versions[name]=None
version_file=pathlib.Path('/isaac-sim/VERSION')
print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':str(torch.__version__),'cuda':torch.cuda.is_available(),'isaac_version':version_file.read_text().strip() if version_file.exists() else None,'package_versions':versions}))'''
    runtime=json.loads(native('docker','exec',CONTAINER,'/isaac-sim/python.sh','-c',code).strip().splitlines()[-1])
    require(runtime['cuda'] is True and runtime['isaac_version'] is not None and runtime['isaac_version'].startswith('5.1'),'Isaac/CUDA runtime mismatch')
    runtime['container_image']=inspection['Image'];runtime['container_id']=inspection['Id']
    runtime['gpu']=native('nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits').strip()
    resources=sample_resources(Path(contract['manifest']['output_root']))
    require(resources['gpu_device_total_mib']<=CAP['gpu_device_total_mib'] and resources['host_mem_available_kib']>=CAP['host_mem_available_min_kib'],'resource preflight failed')
    runtime['resource_preflight']=resources
    sources=contract['sources']
    paths={p:'/workspace/isaaclab_onix/source/onix_isaac_lab/'+p for p in sources['onix_source_files_sha256']}
    require(container_hashes(paths)==sources['onix_source_files_sha256'],'host/container Onix source mismatch')
    runtime['onix_source_parity']='pass'
    return runtime


def execute_frozen_host(target,contract,binding,**kwargs):
    from scripts import eat_nav_motion_14_collection_host as host
    verify_sources(contract)
    current=json.loads(native('docker','inspect',CONTAINER))[0]
    require(current['Image']==binding['runtime']['container_image'] and current['Id']==binding['runtime']['container_id'],'runtime binding changed')
    stage=host.container_stage(target);original=host.run;parity_checked=False
    def checked_run(*args):
        nonlocal parity_checked
        result=original(*args)
        if tuple(args)==('docker','exec',CONTAINER,'sha256sum',stage+'/sim.py'):
            expected=dict(contract['sources']['repo_files_sha256'])
            expected.update({p:d for p,d in binding['source_sha256'].items() if p.startswith('src/')})
            paths={p:stage+'/'+p for p in expected if p.startswith('src/')}
            paths.update({'sim.py':stage+'/sim.py','config':stage+'/config/eat_nav.yaml','actor':stage+'/config/rl_actor.pt','actor_env':stage+'/config/actor_env.yaml'})
            want={p:expected[p] for p in paths if p in expected}
            critical=contract['sources']['pilot_critical_sha256']
            want.update({'sim.py':critical['sim_runner'],'config':critical['config'],'actor':critical['actor'],'actor_env':critical['actor_env']})
            actual=container_hashes(paths)
            require(actual==want,'staged source identity mismatch')
            onix=contract['sources']['onix_source_files_sha256']
            require(container_hashes({p:'/workspace/isaaclab_onix/source/onix_isaac_lab/'+p for p in onix})==onix,'Onix source changed')
            write(Path(target)/'stage-parity.json',{'pass':True,'files_sha256':actual,'onix_source_parity':'pass'})
            parity_checked=True
        return result
    host.run=checked_run
    try:
        host.execute(target,**kwargs)
        require(parity_checked,'staging parity hook did not execute')
    finally:host.run=original
