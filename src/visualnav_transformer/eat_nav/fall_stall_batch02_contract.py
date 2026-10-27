"""Batch02-only contract; historical batch01 sources remain immutable."""
from pathlib import Path
from .fall_stall_audit import read,require,sha,verify_derived

REPO=Path(__file__).resolve().parents[3]
FREEZE_LEDGER_SHA256='6e65b2ec0c17dcb8958f3cc58abba722f27a8ac43fce740dc8ccc49e82d2eae0'


def contained_path(root,relative):
    root=Path(root);name=Path(relative)
    require(not root.is_symlink() and not name.is_absolute() and '..' not in name.parts and bool(name.parts),'unsafe evidence path')
    current=root
    for part in name.parts:
        current=current/part
        require(not current.is_symlink(),'symlink in evidence path')
    require(current.resolve().is_relative_to(root.resolve()),'evidence path escapes root')
    return current


def load_freeze(root):
    root=Path(root)
    require(not root.is_symlink(),'symlink freeze root')
    verify_derived(root)
    require(sha(root/'sha-ledger.json')==FREEZE_LEDGER_SHA256,'frozen ledger identity mismatch')
    manifest=read(root/'collection-manifest.json');rows=read(root/'assignment.json')['rows']
    sources=read(root/'source-identities.json');split=read(root/'split.json')
    require(manifest['output_root']=='/home/ubuntu/Projects/onix_isaac_lab/.eatnav-fall-stall-additional-v0.2-batch02','batch02 root identity')
    for name,digest in manifest['files_sha256'].items():require(sha(contained_path(root,name))==digest,'freeze file identity mismatch')
    require(manifest['batch_id']=='eatnav-fall-stall-additional-v0.2-batch02' and manifest['failure_induction'] is False,'unexpected collection contract')
    require(sha(root/'protocol.md')==manifest['protocol_sha256'] and len(rows)==120,'protocol or schedule mismatch')
    require([r['index'] for r in rows]==list(range(120)),'schedule order mismatch')
    for row in rows:
        contained_path(Path(manifest['output_root']),row['output_relative'])
        if row['reference_relative'] is not None:contained_path(Path(manifest['output_root']),row['reference_relative'])
        require(row['role'] in split['roles'] and row['terrain_seed'] in split['roles'][row['role']]['base_seeds'],'seed role mismatch')
    return {'freeze':root.resolve(),'ledger_sha256':FREEZE_LEDGER_SHA256,'manifest':manifest,'rows':rows,
            'sources':sources,'split':split,'repo':REPO}


def verify_sources(contract):
    sources=contract['sources'];repo=contract['repo']
    for relative,digest in sources['repo_files_sha256'].items():require(sha(contained_path(repo,relative))==digest,'frozen source mismatch')
    for relative,digest in sources['onix_source_files_sha256'].items():
        require(sha(contained_path(sources['onix_source_root'],relative))==digest,'frozen Onix source mismatch')
    for name in ('actor','actor_env'):require(sha(sources[name]['path'])==sources[name]['sha256'],'Actor identity mismatch')
    # Namespace packages must remain namespaces, and additions must not bypass SHA pinning.
    from .fall_stall_batch02_runner import ADDED_SOURCES
    prefix='src/visualnav_transformer/'
    suffixes={'.py','.cpp','.h','.hpp','.yaml','.pyc','.pyo','.so'}
    actual={str(p.relative_to(repo)) for p in (repo/prefix).rglob('*')
            if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts}
    expected={p for p in set(sources['repo_files_sha256'])|set(ADDED_SOURCES) if p.startswith(prefix)}
    require(actual==expected,'source inventory membership changed')
    for relative in ('src/visualnav_transformer/__init__.py','src/visualnav_transformer/__init__.pyc',
                     'scripts/__init__.py','scripts/__init__.pyc'):
        require(not (repo/relative).exists(),'namespace initializer added')
    return True
