import gzip,hashlib
from eat_nav_r6_storage import compress_traces,read_trace

def test_lossless_trace_storage_keeps_logical_digest_and_can_be_read(tmp_path):
 p=tmp_path/'group-000';p.mkdir();trace=p/'trace.jsonl';raw=b'{"a":1}\n{"a":2}\n';trace.write_bytes(raw)
 report=compress_traces(tmp_path)
 assert not trace.exists() and gzip.decompress((p/'trace.jsonl.gz').read_bytes())==raw
 assert report['group-000/trace.jsonl']['uncompressed_sha256']==hashlib.sha256(raw).hexdigest()
 assert read_trace(trace)==[{'a':1},{'a':2}]
