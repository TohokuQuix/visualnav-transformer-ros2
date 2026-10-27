"""Live adapter must preserve checkpoint input parity and actually use risk."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('eat_nav_risk_navigation')


def test_risk_changes_path_choice_and_rejects_invalid_prediction():
    # Same progress; direct advances most, but direct is predicted to tip.
    p=np.zeros((4,4));p[0,0]=1
    result=api().select_candidate(p,np.ones((4,4)),[4.,0.])
    assert result['goal_only_index']==0
    assert result['selected_index'] in (1,2)
    p[1,0]=np.nan
    with pytest.raises(ValueError,match='prediction'):
        api().select_candidate(p,np.ones((4,4)),[4.,0.])


def test_live_initial_input_matches_dataset_tensor_and_inference():
    from eat_nav_current_learning import load_current_dataset
    from visualnav_transformer.eat_nav.fall_stall_learning import prepare_inputs
    from visualnav_transformer.eat_nav.fall_stall_training import predict,load_checkpoint
    import torch
    module=api();a,index,fold=load_current_dataset()
    # Adapter exposes body-state packing separately: test actual dataset values,
    # preserving every command component instead of zeroing observations.
    arrays=module.pack_inputs(a['dem'][0],a['actor_state'][0],a['body_state'][0])
    for name in arrays:np.testing.assert_array_equal(arrays[name][0],a[name][0])
    evaluator=module.OnlineEvaluator(module.DEFAULT_CHECKPOINT)
    expected=prepare_inputs(a,[0],evaluator.payload['normalization'])
    actual=prepare_inputs(arrays,[0],evaluator.payload['normalization'])
    for name in expected:torch.testing.assert_close(expected[name],actual[name],rtol=0,atol=0)
    p,progress=evaluator.predict(arrays)
    model,payload=load_checkpoint(module.DEFAULT_CHECKPOINT)
    old_p,old_progress=predict(dict(payload,model=model),a,[0])
    np.testing.assert_array_equal(p,old_p[0]);np.testing.assert_array_equal(progress,old_progress[0])


def test_body_rotation_and_frame_age_are_enforced():
    module=api()
    # Unit quaternion yaw=90deg maps world +x velocity to body -y.
    q=np.sqrt(.5);root=[0,0,1,q,0,0,q,1,0,0,0,0,0]
    np.testing.assert_allclose(module.body_state(root),[0,0,-1,0,-1,0,0,0,0],atol=1e-6)
    module.validate_history([250_000_000,500_000_000,750_000_000,1_000_000_000],1_000_000_000)
    for stamps in ([0,0,0,0],[250_000_000,500_000_000,750_000_000,1_250_000_000]):
        with pytest.raises(ValueError,match='history'):
            module.validate_history(stamps,1_000_000_000)


def test_checkpoint_identity_checked_before_loading(tmp_path):
    p=tmp_path/'bad.pt';p.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='checkpoint SHA'):
        api().OnlineEvaluator(p)


def test_cpu_evaluator_does_not_change_simulator_determinism_setting():
    # CUDA DelayBuffer broadcast assignment in Torch2.7 fails under global
    # deterministic=True. CPU inference must not enable that process-wide flag.
    import torch
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        module=api();module.OnlineEvaluator(module.DEFAULT_CHECKPOINT)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)
