"""Consume the immutable additional-collection freeze without changing its roles."""
from pathlib import Path
from .fall_stall_audit import read,require,sha,verify_derived

REPO=Path(__file__).resolve().parents[3]
FREEZE_LEDGER_SHA256='84e10c7447ca845925b5714ab4a9d241d95529d78376e4b647272e14d534c27a'


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
    for name,digest in manifest['files_sha256'].items():require(sha(contained_path(root,name))==digest,'freeze file identity mismatch')
    require(manifest['batch_id']=='eatnav-fall-stall-additional-v0.1-batch01' and manifest['failure_induction'] is False,'unexpected collection contract')
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
    return True
