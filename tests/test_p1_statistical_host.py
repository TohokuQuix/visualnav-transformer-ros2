import importlib.util
from pathlib import Path
from types import SimpleNamespace


def host_module():
    path=Path(__file__).parents[1]/'scripts/eat_nav_p1_statistical_host.py'
    spec=importlib.util.spec_from_file_location('p1_statistical_host',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_copy_parent_is_created_before_copy():
    host=host_module()
    calls=[]
    def fake_docker(*args):
        calls.append(args)
        return SimpleNamespace(returncode=0,stderr='')
    host.prepare_container_parent(Path('/tmp/p1-statistical/run-v2'),fake_docker)
    assert calls==[('mkdir','-p','/tmp/p1-statistical')]


def test_host_run_accepts_diagnostic_branch_and_params_contract():
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    path=Path(__file__).parents[1]/'scripts/eat_nav_p1_statistical_pilot_host.py'
    spec=importlib.util.spec_from_file_location('p1_statistical_pilot_host',path)
    host=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    import inspect
    assert 'branches' in inspect.signature(host.run).parameters
    assert 'params' in inspect.signature(host.run).parameters
