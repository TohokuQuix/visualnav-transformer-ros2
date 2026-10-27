"""Native stage-only preparation and fresh runtime checks for batch02."""
from pathlib import Path
import uuid
import json

from .fall_stall_audit import require,write
from .fall_stall_batch02_contract import verify_sources
from .fall_stall_collection_runtime import (
    CONTAINER, native, container_hashes, probe_runtime,
)
from .fall_stall_blind_verification import BlindVerificationError


def expected_stage_hashes(contract, source_sha256):
    expected = dict(contract['sources']['repo_files_sha256'])
    expected.update(source_sha256)
    critical = contract['sources']['pilot_critical_sha256']
    expected.update({'sim.py': critical['sim_runner'], 'actor': critical['actor'],
                     'actor_env': critical['actor_env']})
    return expected


def stage_paths(stage, expected):
    aliases = {'actor': 'config/rl_actor.pt', 'actor_env': 'config/actor_env.yaml'}
    return {key: stage + '/' + aliases.get(key, key) for key in expected}


def stage_preflight(contract, source_sha256):
    """Copy/hash a fresh temporary stage; never run sim.py or create a raw root."""
    stage = '/tmp/eatnav-batch02-preflight-' + uuid.uuid4().hex
    expected = expected_stage_hashes(contract, source_sha256)
    native('docker', 'exec', CONTAINER, 'test', '!', '-e', stage)
    native('docker', 'exec', CONTAINER, 'mkdir', '-p', stage + '/src', stage + '/config', stage + '/scripts')
    native('docker', 'cp', str(contract['repo'] / 'src/visualnav_transformer'), CONTAINER + ':' + stage + '/src/')
    for relative in expected:
        if relative.startswith('scripts/') or relative.startswith('config/'):
            native('docker', 'cp', str(contract['repo'] / relative), CONTAINER + ':' + stage + '/' + relative)
    native('docker', 'cp', str(contract['repo'] / 'scripts/eat_nav_motion_14_collection_sim.py'), CONTAINER + ':' + stage + '/sim.py')
    for key, relative in (('actor', 'config/rl_actor.pt'), ('actor_env', 'config/actor_env.yaml')):
        native('docker', 'cp', contract['sources'][key]['path'], CONTAINER + ':' + stage + '/' + relative)
    remove_staged_bytecode(stage)
    check_stage_membership(stage, expected)
    actual = container_hashes(stage_paths(stage, expected))
    require(actual == expected, 'staged source/config/Actor mismatch')
    return {'pass': True, 'stage': stage, 'files_sha256': actual}


def immutable_runtime(runtime):
    result = {key: runtime[key] for key in ('container_image', 'container_id', 'python', 'numpy',
                                          'torch', 'isaac_version', 'cuda', 'package_versions')}
    for key in ('gpu_uuid', 'gpu_capacity_mib'):
        result[key] = runtime['resource_preflight'][key]
    return result


def check_runtime_proof(contract, runtime, source_sha256):
    require(runtime['cuda'] is True and runtime['isaac_version'].startswith('5.1'), 'Isaac/CUDA proof')
    require(runtime['onix_source_parity'] == 'pass' and isinstance(runtime['package_versions'], dict), 'Onix/runtime proof')
    proof = runtime['stage_parity']
    require(proof['pass'] is True and proof['files_sha256'] == expected_stage_hashes(contract, source_sha256), 'staging proof')
    stage = Path(proof['stage'])
    require(stage.parent == Path('/tmp') and stage.name.startswith('eatnav-batch02-preflight-'), 'staging path')
    resources = runtime['resource_preflight']
    cap = contract['sources']['host_resource_cap']
    require(resources['gpu_uuid'] and resources['gpu_capacity_mib'] > 0
            and resources['gpu_device_total_mib'] <= cap['gpu_device_total_mib']
            and resources['host_mem_available_kib'] >= cap['host_mem_available_min_kib'], 'resource proof')


def check_staged_runtime(contract, binding):
    proof = binding['runtime']['stage_parity']
    expected = expected_stage_hashes(contract, binding['source_sha256'])
    check_stage_membership(proof['stage'], expected)
    require(container_hashes(stage_paths(proof['stage'], expected)) == expected, 'prepared stage changed')


def remove_staged_bytecode(stage):
    # Only a fresh /tmp stage is sanitized. Historical evidence/source trees are untouched.
    code = ('from pathlib import Path; root=Path(' + repr(stage + '/src') + '); '
            '[p.unlink() for p in root.rglob("*") if p.is_file() and p.suffix in (".pyc",".pyo")]')
    native('docker', 'exec', CONTAINER, '/isaac-sim/python.sh', '-c', code)


def check_stage_membership(stage, expected):
    code = '''from pathlib import Path
import json
root=Path(STAGE)
prefix=root/'src/visualnav_transformer'
suffixes={'.py','.cpp','.h','.hpp','.yaml','.pyc','.pyo','.so'}
files=sorted(str(p.relative_to(root)) for p in prefix.rglob('*') if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts)
initializers=[name for name in ('src/visualnav_transformer/__init__.py','src/visualnav_transformer/__init__.pyc','scripts/__init__.py','scripts/__init__.pyc') if (root/name).exists()]
bytecode=[str(p.relative_to(root)) for p in (root/'src').rglob('*') if p.is_file() and p.suffix in ('.pyc','.pyo')]
print(json.dumps({'source_files':files,'namespace_initializers':initializers,'bytecode_files':bytecode}))
'''.replace('STAGE', repr(stage))
    actual = json.loads(native('docker', 'exec', CONTAINER, '/isaac-sim/python.sh', '-c', code).strip().splitlines()[-1])
    files = sorted(key for key in expected if key.startswith('src/visualnav_transformer/'))
    require(actual['source_files'] == files and not actual['namespace_initializers']
            and not actual['bytecode_files'], 'staged executable inventory changed')


def recheck_runtime(contract, binding):
    """Startup gate: immutable runtime and prepared stage must still match."""
    try:
        from .fall_stall_batch02_runner import check_binding
        check_binding(contract, binding)
        current = probe_runtime(contract)
        require(immutable_runtime(current) == immutable_runtime(binding['runtime']), 'runtime changed')
        check_staged_runtime(contract, binding)
        return current
    except Exception:
        raise BlindVerificationError('BINDING_MISMATCH') from None


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
            remove_staged_bytecode(stage)
            check_stage_membership(stage,want)
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
