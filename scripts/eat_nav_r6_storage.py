"""Verified storage of newly produced R6 traces; old datasets are never rewritten."""
import gzip,hashlib,json,shutil,subprocess
from pathlib import Path
from eat_nav_r6_run import sha,write


def read_trace(path):
 path=Path(path)
 if path.exists():handle=path.open('rt')
 elif path.with_suffix(path.suffix+'.gz').exists():handle=gzip.open(path.with_suffix(path.suffix+'.gz'),'rt')
 else:raise FileNotFoundError(path)
 with handle:return [json.loads(line) for line in handle if line.strip()]


def compress_traces(root):
 root=Path(root);report={}
 for path in root.glob('group-*/trace.jsonl'):
  digest=sha(path);destination=path.with_suffix(path.suffix+'.gz')
  if destination.exists():raise FileExistsError('preserved compression attempt '+str(destination))
  with path.open('rb') as src,destination.open('xb') as raw:
   with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=3,mtime=0) as dst:shutil.copyfileobj(src,dst,1024*1024)
  verified=hashlib.sha256()
  with gzip.open(destination,'rb') as src:
   for chunk in iter(lambda:src.read(1024*1024),b''):verified.update(chunk)
  if verified.hexdigest()!=digest:raise ValueError('lossless trace compression verification failed')
  report[str(path.relative_to(root))]=dict(uncompressed_sha256=digest,uncompressed_bytes=path.stat().st_size,compressed_sha256=sha(destination),compressed_bytes=destination.stat().st_size,compressed_path=str(destination.relative_to(root)))
  # Newly produced raw bytes remain recoverable exactly from verified gzip.
  path.unlink()
 write(root/'trace-storage.json',report);return report


def verify_copied_tree(container,remote,local):
 remote=Path(remote);local=Path(local)
 # Only a new, task-owned container scratch directory may be cleaned later.
 if not str(remote).startswith('/tmp/eatnav-r6-production-') or '..' in remote.parts:raise ValueError('not owned R6 scratch output')
 code="import pathlib,hashlib,json,sys; r=pathlib.Path(sys.argv[1]); print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()}))"
 actual=json.loads(subprocess.check_output(['docker','exec',container,'/isaac-sim/python.sh','-c',code,str(remote)],text=True))
 expected={str(p.relative_to(local)):sha(p) for p in local.rglob('*') if p.is_file()}
 if not actual or actual!=expected:raise ValueError('container/host copy bytes differ; preserve both')
 write(local.parent/'copied-output-sha256.json',dict(remote=str(remote),files_sha256=expected,verified=True));return expected


def remove_verified_scratch(container,remote):
 remote=Path(remote)
 if not str(remote).startswith('/tmp/eatnav-r6-production-') or len(remote.parts)<5 or '..' in remote.parts:raise ValueError('not owned R6 job scratch')
 subprocess.run(['docker','exec',container,'rm','-rf','--',str(remote)],check=True)
