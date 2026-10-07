"""One-shot host lifecycle for the candidate-free RL-reset five-pose contact smoke."""
from __future__ import annotations
import hashlib,json,os,subprocess,time,traceback,sys
from pathlib import Path

ROOT=Path(os.environ['EATNAV_FIVE_POSE_ROOT'])
WORKTREE=Path('/tmp/eatnav-vnt-radial-112-sharded')
CONTAINER='isaac-lab-onix'
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.contact_host_gate import evaluate_run_markers
from visualnav_transformer.eat_nav.radial_shard_inventory import verify_shard_artifacts

def process_matches(owned,pid,start_tick):
    return int(owned['pid'])==int(pid) and int(owned['proc_start_ticks'])==int(start_tick)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)

def docker(*args,check=True):
    return subprocess.run(['docker',*map(str,args)],capture_output=True,text=True,check=check)

def rsha(path):
    p=docker('exec',CONTAINER,'sha256sum',path,check=False)
    return p.stdout.split()[0] if p.returncode==0 else None

def readremote(path):
    p=docker('exec',CONTAINER,'cat',path,check=False)
    return json.loads(p.stdout) if p.returncode==0 else None

def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return max(int(s.strip()) for s in p.stdout.splitlines() if s.strip())

def hostmem():
    return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024

def anon():
    text=docker('exec',CONTAINER,'cat','/sys/fs/cgroup/memory.stat').stdout
    d={x.split()[0]:int(x.split()[1]) for x in text.splitlines() if len(x.split())==2}
    return d.get('anon',0)+d.get('shmem',0)

def storage():
    scope=ROOT.parent.parent
    remote=docker('exec',CONTAINER,'du','-sb',scope,check=False)
    r=int(remote.stdout.split()[0]) if remote.returncode==0 else 0
    h=sum(p.stat().st_size for p in scope.rglob('*') if p.is_file())
    return r+h

def owned():
    return readremote(ROOT/'run/owned-process.json')

def owned_alive(item):
    if not item:return False
    stat=docker('exec',CONTAINER,'cat',f"/proc/{item['pid']}/stat",check=False)
    if stat.returncode!=0:return False
    try:tick=int(stat.stdout.split(') ',1)[1].split()[19])
    except (ValueError,IndexError):raise RuntimeError('owned PID start tick unreadable')
    return process_matches(item,item['pid'],tick)

def stop_owned(client):
    item=owned()
    if owned_alive(item):
        docker('exec',CONTAINER,'kill','-TERM',item['pid'],check=False)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and owned_alive(item):time.sleep(.25)
        if owned_alive(item):
            docker('exec',CONTAINER,'kill','-KILL',item['pid'],check=False)
            time.sleep(.5)
        if owned_alive(item):raise RuntimeError('owned simulator residual after targeted cleanup')
    if client.poll() is None:
        client.terminate()
        try: client.wait(timeout=20)
        except subprocess.TimeoutExpired: client.kill();client.wait(timeout=5)

def copy_inputs(freeze):
    mapping=[(ROOT/'input-freeze.json',ROOT/'input-freeze.json'),
             (ROOT/'pose-manifest.json',ROOT/'pose-manifest.json'),
             (WORKTREE/'scripts/eat_nav_radial_five_pose_sim.py',ROOT/'inputs/eat_nav_radial_five_pose_sim.py'),
             (WORKTREE/'src/visualnav_transformer/eat_nav/radial_contact_audit.py',ROOT/'src/visualnav_transformer/eat_nav/radial_contact_audit.py'),
             (WORKTREE/'src/visualnav_transformer/eat_nav/contact_preflight_protocol.py',ROOT/'src/visualnav_transformer/eat_nav/contact_preflight_protocol.py'),
             (WORKTREE/'src/visualnav_transformer/eat_nav/radial_five_pose_contract.py',ROOT/'src/visualnav_transformer/eat_nav/radial_five_pose_contract.py'),
             (WORKTREE/'src/visualnav_transformer/eat_nav/contact_raw_capture.py',ROOT/'src/visualnav_transformer/eat_nav/contact_raw_capture.py')]
    mapping.append((WORKTREE/'src/visualnav_transformer/eat_nav/radial_lean_log.py',
                    ROOT/'src/visualnav_transformer/eat_nav/radial_lean_log.py'))
    mapping.append((WORKTREE/'src/visualnav_transformer/eat_nav/radial_shard_inventory.py',
                    ROOT/'src/visualnav_transformer/eat_nav/radial_shard_inventory.py'))
    for local,remote in mapping:
        docker('exec',CONTAINER,'mkdir','-p',remote.parent)
        docker('cp',local,f'{CONTAINER}:{remote}')
        if sha(local)!=rsha(remote): raise RuntimeError('copy SHA mismatch: '+str(local))
    for label,path in freeze['source_paths'].items():
        if rsha(path)!=freeze['source_sha256'][label]:
            raise RuntimeError('frozen remote SHA mismatch: '+label)

def main():
    freeze=json.loads((ROOT/'input-freeze.json').read_text())
    pre=json.loads((ROOT/'prelaunch.json').read_text())
    if (ROOT/'host-launch.json').exists():raise RuntimeError('one-shot already launched')
    if sha(ROOT/'input-freeze.json')!=pre['freeze_sha256'] or sha(ROOT/'pose-manifest.json')!=freeze['pose_manifest_sha256']:
        raise RuntimeError('host freeze SHA mismatch')
    for path,expected in pre['source_and_test_sha256'].items():
        if sha(path)!=expected:
            raise RuntimeError('source SHA mismatch before launch: '+path)
    copy_inputs(freeze)
    cmd=pre['command']
    atomic(ROOT/'host-launch.json',{'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
            'command':cmd,'freeze_sha256':sha(ROOT/'input-freeze.json'),'candidate_free':True})
    atomic(ROOT/'status.json',{'stage':'running','physics_steps':0,'decision':None})
    started=time.monotonic(); client=None
    try:
        with (ROOT/'runner.log').open('wb') as log:
            client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            while client.poll() is None:
                hb=readremote(ROOT/'run/heartbeat.json')
                sample={'elapsed_sec':time.monotonic()-started,'gpu_device_used_mib':gpu(),
                        'host_available_bytes':hostmem(),'container_anon_shmem_bytes':anon(),
                        'storage_bytes':storage(),'heartbeat':hb,'wall_ns':time.time_ns()}
                atomic(ROOT/'resource-latest.json',sample)
                with (ROOT/'resource-series.jsonl').open('a') as f: f.write(json.dumps(sample,sort_keys=True)+'\n')
                cap=freeze['limits'];reasons=[]
                if sample['gpu_device_used_mib']>cap['gpu_device_total_cap_mib']:reasons.append('gpu_cap')
                if sample['host_available_bytes']<cap['host_mem_available_floor_bytes']:reasons.append('host_memory_floor')
                if sample['container_anon_shmem_bytes']>cap['anon_plus_shmem_cap_bytes']:reasons.append('anon_shmem_cap')
                if sample['storage_bytes']>cap['storage_cap_bytes']:reasons.append('storage_cap')
                if sample['elapsed_sec']>cap['process_timeout_sec']:reasons.append('runtime_cap')
                if hb is None and sample['elapsed_sec']>cap['startup_cap_sec']:reasons.append('startup_cap')
                if hb is not None and (time.monotonic_ns()-hb['monotonic_ns'])/1e9>cap['heartbeat_cap_sec']:reasons.append('heartbeat_cap')
                if reasons:
                    stop_owned(client)
                    atomic(ROOT/'status.json',{'stage':'resource_or_lifecycle_stop','reasons':reasons,'sample':sample})
                    break
                time.sleep(2)
        docker('exec',CONTAINER,'mkdir','-p',ROOT/'run',check=False)
        docker('cp',f'{CONTAINER}:{ROOT}/run/.',ROOT/'run',check=False)
        if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text())['stage']=='resource_or_lifecycle_stop':
            raise RuntimeError('resource or lifecycle cap')
        manifest=json.loads((ROOT/'pose-manifest.json').read_text())
        pose_ids=[item['pose_id'] for item in manifest['poses']]
        if len(pose_ids)!=freeze['expected_pose_count']:
            raise RuntimeError('frozen pose count mismatch')
        outcome=evaluate_run_markers(ROOT/'run',freeze['run_kind'],client.returncode,
            expected_pose_count=len(pose_ids))
        if outcome=='invalid':
            raise RuntimeError(f'invalid simulator markers: exit={client.returncode}, mode={freeze["run_kind"]}')
        artifact_gate=verify_shard_artifacts(ROOT/'run',pose_ids)
        files=[p for p in (ROOT/'run').rglob('*') if p.is_file()]
        mismatches=[str(p) for p in files if sha(p)!=rsha(p)]
        if mismatches:raise RuntimeError('host/container SHA mismatch: '+repr(mismatches))
        item=owned()
        if owned_alive(item):
            raise RuntimeError('owned simulator process remains')
        atomic(ROOT/'sha-ledger.json',{'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]})
        atomic(ROOT/'seal.json',{'schema':'eatnav_radial_shard_seal_v1','shard_id':freeze['shard_id'],
            'pose_ids':pose_ids,'artifact_gate':artifact_gate,'native_exit_code':client.returncode,
            'owned_process':item,'owned_residual':False,'run_sha_ledger_sha256':sha(ROOT/'sha-ledger.json'),
            'freeze_sha256':sha(ROOT/'input-freeze.json'),'pose_manifest_sha256':sha(ROOT/'pose-manifest.json'),
            'resource_latest':json.loads((ROOT/'resource-latest.json').read_text())})
        atomic(ROOT/'status.json',{'stage':outcome,'run_kind':freeze['run_kind'],
            'native_exit_code':client.returncode,'owned_residual':False,
            'sha_ledger_sha256':sha(ROOT/'sha-ledger.json'),'seal_sha256':sha(ROOT/'seal.json')})
    except BaseException as e:
        if client and client.poll() is None:stop_owned(client)
        docker('cp',f'{CONTAINER}:{ROOT}/run/.',ROOT/'run',check=False)
        hb=readremote(ROOT/'run/heartbeat.json')
        decision='five_pose_measurement_partial_review_required' if hb and hb.get('physics_step',0)>0 else 'five_pose_measurement_invalid_stop'
        atomic(ROOT/'status.json',{'stage':'stopped','decision':decision,'error':repr(e),
                'traceback':traceback.format_exc(),'heartbeat':hb,
                'native_exit_code':None if not client else client.poll()})
        raise

if __name__=='__main__':main()
