import json
import pytest
from visualnav_transformer.eat_nav.tf_race_evidence import write_first_conflict


def test_first_conflict_saves_both_commands_and_context_atomically(tmp_path):
    p=tmp_path/'conflict.json'
    a={'branch':4,'stamp_ns':1_600_000_000,'accepted':True,'command':[.28,0,-.1]}
    result=write_first_conflict(p,branch=4,stamp_ns=1_600_000_000,
        previous=[(.28,0,-.1)],incoming=[.27,0,-.12],incoming_event={'receive_seq':2},preceding=[a],publishers=[{'kind':'tf-publish'}])
    saved=json.loads(p.read_text())
    assert saved==result
    assert saved['A']==[.28,0,-.1] and saved['B']==[.27,0,-.12]
    assert saved['A_receive']==a and saved['B_receive']['receive_seq']==2 and saved['last_publisher_events'][0]['kind']=='tf-publish'
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(FileExistsError):
        write_first_conflict(p,branch=4,stamp_ns=1_600_000_000,
            previous=[(.29,0,-.1)],incoming=[.27,0,-.12],incoming_event={'receive_seq':2},preceding=[a],publishers=[])


def test_missing_a_context_rejected(tmp_path):
    with pytest.raises(ValueError,match='context'):
        write_first_conflict(tmp_path/'bad.json',branch=4,stamp_ns=3,
            previous=[(.1,0,0)],incoming=[(.2,0,0)],incoming_event={},preceding=[],publishers=[])
