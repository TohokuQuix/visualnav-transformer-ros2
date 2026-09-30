"""Durable diagnostic evidence for the first same-stamp command conflict."""
from __future__ import annotations
import json
from pathlib import Path


def write_first_conflict(path, *, branch, stamp_ns, previous, incoming, incoming_event, preceding, publishers):
    if not previous or tuple(previous[0]) == tuple(incoming):
        raise ValueError('two different commands are required')
    matches=[x for x in preceding if x.get('branch')==branch and x.get('stamp_ns')==stamp_ns and x.get('accepted')]
    if not matches:
        raise ValueError('accepted command A context is missing')
    value={'branch':branch,'stamp_ns':stamp_ns,'A':list(previous[0]),'B':list(incoming),
           'A_receive':matches[0],'B_receive':incoming_event,'bridge_preceding_events':preceding[-20:],
           'last_publisher_events':publishers[-80:]}
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        existing=json.loads(target.read_text())
        if existing!=value:raise FileExistsError('first conflict evidence already frozen')
        return existing
    temp=target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(target)
    return value
