"""Exercise all run exports and checkpoint/metric replay with two test epochs."""
import importlib
import json
from pathlib import Path
import pytest
import torch


def test_all_runs_export_verify_reject_overwrite_and_detect_mutation(tmp_path,monkeypatch):
    api=importlib.import_module('visualnav_transformer.eat_nav.fall_stall_comparison')
    repo=Path(__file__).resolve().parents[2]
    dataset=repo/'docs/eat-nav/fall-stall-dataset-freeze-20261019/artifacts'
    config=json.loads((repo/'config/eat_nav/fall_stall_baseline_v1.json').read_text())
    config['epochs']=2
    original_validate=api.validate_config
    monkeypatch.setattr(api,'validate_config',lambda c:original_validate(dict(c,epochs=300)))
    config_path=tmp_path/'test-config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    output=tmp_path/'comparison'
    result=api.run_comparison(dataset,config_path,output,repo)
    assert len(result['runs'])==24 and result['checkpoint_roundtrips']==18
    torch.set_num_threads(2)
    assert api.verify_comparison(output,dataset)['verified_runs']==24
    assert torch.get_num_threads()==config['threads']
    with pytest.raises(FileExistsError):api.run_comparison(dataset,config_path,output,repo)
    predictions=next(output.glob('runs/*/holdout-predictions.json'))
    predictions.write_text('{}\n')
    with pytest.raises(ValueError,match='SHA|ledger'):api.verify_comparison(output,dataset)
