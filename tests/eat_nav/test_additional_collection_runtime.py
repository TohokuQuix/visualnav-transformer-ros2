"""Preflight/staging fail closed at the native Docker boundary."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from visualnav_transformer.eat_nav.fall_stall_collection_contract import load_freeze
from visualnav_transformer.eat_nav.fall_stall_audit import sha


def api():return importlib.import_module('visualnav_transformer.eat_nav.fall_stall_collection_runtime')


def test_container_parity_hashes_actual_file_bytes(tmp_path):
    p=tmp_path/'code.py';p.write_bytes(b'print(123)\n')
    def local_native(*args):return subprocess.run([sys.executable,*args[-2:]],check=True,text=True,capture_output=True).stdout
    assert api().container_hashes({'module':str(p)},runner=local_native)=={'module':sha(p)}


def test_staging_mismatch_prevents_simulator_launch_and_restores_host(monkeypatch,tmp_path):
    from scripts import eat_nav_motion_14_collection_host as host
    c=load_freeze('docs/eat-nav/fall-stall-additional-collection-freeze-20261020/artifacts')
    target=tmp_path/'attempt';target.mkdir();launched=[]
    original_run=lambda *args:SimpleNamespace(stdout='hash sim.py')
    monkeypatch.setattr(host,'run',original_run)
    monkeypatch.setattr(api(),'native',lambda *args:json.dumps([{'Image':'image','Id':'container'}]))
    monkeypatch.setattr(api(),'container_hashes',lambda paths:{})
    def fake_execute(target,**kwargs):
        host.run('docker','exec','isaac-lab-onix','sha256sum',host.container_stage(target)+'/sim.py')
        launched.append(True)
    monkeypatch.setattr(host,'execute',fake_execute)
    binding={'runtime':{'container_image':'image','container_id':'container'},'source_sha256':{}}
    with pytest.raises(ValueError,match='staged'):api().execute_frozen_host(target,c,binding)
    assert launched==[] and host.run is original_run


def test_runtime_probe_rejects_cuda_absence_without_collecting(monkeypatch):
    c=load_freeze('docs/eat-nav/fall-stall-additional-collection-freeze-20261020/artifacts')
    def native(*args):
        if args[1]=='inspect':return json.dumps([{'State':{'Running':True},'Image':'x','Id':'y'}])
        return json.dumps({'cuda':False,'isaac_version':'5.1.0'})
    monkeypatch.setattr(api(),'native',native)
    with pytest.raises(ValueError,match='CUDA'):api().probe_runtime(c)
