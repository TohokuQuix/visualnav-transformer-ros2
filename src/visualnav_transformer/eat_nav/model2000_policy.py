"""Pinned Actor bundle; apply only policy interfaces and paired actuators."""
from pathlib import Path
import hashlib
import yaml

HASHES={
 'model_2000.pt':'56c16ecca41b315ef57aea57beae17a05d94c5c0f54ede88066e9688b7f8a770',
 'env.yaml':'8e2a592169d3dc4cf193f049f481d35b3c6ff265ae118bb0f8ecf588551d5e2d',
 'agent.yaml':'1d8186221ad7db89bc41578f1a4fe4fbd7b699f8169c0bb1965b0202e59d11bb',
}

class PrimitiveLoader(yaml.SafeLoader):
    pass
PrimitiveLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:tuple(loader.construct_sequence(node)))
PrimitiveLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice',lambda loader,node:slice(*loader.construct_sequence(node)))


def load_bundle(env_path):
    path=Path(env_path)
    for name,expected in HASHES.items():
        if hashlib.sha256((path.parent/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('policy bundle identity mismatch: '+name)
    return yaml.load(path.read_text(),Loader=PrimitiveLoader),yaml.safe_load(path.with_name('agent.yaml').read_text())


def apply_actuator_values(target,values):
    # IsaacLab's typed updater rejects None→float; retain the paired values literally.
    for key,value in values.items():
        if key=='class_type':continue  # existing ImplicitActuator class is unchanged
        setattr(target,key,value)


def apply_bundle(cfg,env_path):
    from isaaclab.utils import update_class_from_dict,class_to_dict
    paired,agent=load_bundle(env_path)
    update_class_from_dict(cfg.actions,paired['actions'])
    # This extra term exists in the current simulator, not the paired Actor contract.
    if hasattr(cfg.observations.critic,'control_latency'):
        cfg.observations.critic.control_latency=None
    update_class_from_dict(cfg.observations,paired['observations'])
    for name,values in paired['scene']['robot']['actuators'].items():
        apply_actuator_values(cfg.scene.robot.actuators[name],values)
    def canonical(v):
        if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
        if isinstance(v,slice):return [v.start,v.stop,v.step]
        return v
    if canonical(class_to_dict(cfg.actions))!=canonical(paired['actions']):
        raise ValueError('paired actions mismatch')
    if canonical(class_to_dict(cfg.observations.policy))!=canonical(paired['observations']['policy']):
        raise ValueError('paired policy observations mismatch')
    return agent
