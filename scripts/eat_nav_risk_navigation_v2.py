"""Live command-free Evaluator v2; raw Actor observations remain untouched."""
from pathlib import Path
import torch
from eat_nav_risk_navigation import (REPO,candidate_paths,body_state,validate_history,pack_inputs,
                                     select_candidate,CANDIDATE_IDS,RISK_WEIGHTS)
from eat_nav_command_free_learning import load_checkpoint,predict,INPUT_CONTRACT
from visualnav_transformer.eat_nav.fall_stall_audit import sha,read,require,verify_derived

ARTIFACTS=REPO/'docs/eat-nav/command-free-learning-20261021/artifacts'
DEFAULT_CHECKPOINT=ARTIFACTS/'dem_state_candidate-seed123/checkpoint.pt'


def training_binding():
    verify_derived(ARTIFACTS)
    binding=read(ARTIFACTS/'gui-checkpoint.json')
    require(binding['input_contract']==INPUT_CONTRACT and
            binding['relative_checkpoint']=='dem_state_candidate-seed123/checkpoint.pt',
            'GUI checkpoint binding schema mismatch')
    require(sha(DEFAULT_CHECKPOINT)==binding['checkpoint_sha256'],'GUI checkpoint SHA mismatch')
    return binding


class OnlineEvaluator:
    def __init__(self,checkpoint,*,expected_sha256=None):
        if expected_sha256 is None:
            require(Path(checkpoint).resolve()==DEFAULT_CHECKPOINT.resolve(),'explicit checkpoint SHA required')
            expected_sha256=training_binding()['checkpoint_sha256']
        require(sha(checkpoint)==expected_sha256,'checkpoint SHA mismatch')
        # CPU-only inference. Do not change process-wide CUDA determinism flags.
        torch.set_num_threads(1)
        model,payload=load_checkpoint(checkpoint)
        require(payload['variant']=='dem_state_candidate' and all(payload['support']['active']) and
                all(n>0 for n in payload['normalization']['progress_count']),'unsupported checkpoint')
        self.payload=payload;self.fit=dict(payload,model=model)

    def predict(self,arrays):
        probability,progress=predict(self.fit,arrays,[0])
        return probability[0],progress[0]
