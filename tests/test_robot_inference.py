"""Compare against the actual legacy navigation branch, not a copied algorithm."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from PIL import Image
from std_msgs.msg import Float32MultiArray
from visualnav_transformer.deployment.src import utils
from visualnav_transformer.deployment.src.robot_inference import (
    NoMaDPredictor,
    load_topomap,
)
from visualnav_transformer.train.vint_train.models.nomad.nomad import (
    DenseNetwork,
    NoMaD,
)
from visualnav_transformer.train.vint_train.models.nomad.nomad_vint import NoMaD_ViNT
from visualnav_transformer.train.vint_train.training.train_utils import get_action


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    cfg = dict(
        model_type="nomad",
        vision_encoder="nomad_vint",
        encoding_size=32,
        context_size=0,
        mha_num_attention_heads=2,
        mha_num_attention_layers=1,
        mha_ff_dim_factor=2,
        down_dims=[8, 16],
        cond_predict_scale=False,
        normalize=True,
        image_size=[96, 96],
        len_traj_pred=4,
        num_diffusion_iters=2,
    )
    model = NoMaD(
        NoMaD_ViNT(
            context_size=0,
            obs_encoding_size=32,
            mha_num_attention_heads=2,
            mha_num_attention_layers=1,
            mha_ff_dim_factor=2,
        ),
        ConditionalUnet1D(
            input_dim=2, global_cond_dim=32, down_dims=[8, 16], cond_predict_scale=False
        ),
        DenseNetwork(32),
    )
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)
    topomap = tmp_path / "topomap"
    topomap.mkdir()
    for i, color in enumerate(["red", "green", "blue"]):
        Image.new("RGB", (96, 96), color).save(topomap / f"{i}.png")
    return cfg, ckpt, topomap


def test_seeded_result_matches_existing_navigate(inputs):
    cfg, ckpt, topomap_dir = inputs
    predictor = NoMaDPredictor(
        model_config=cfg,
        checkpoint=str(ckpt),
        topomap=str(topomap_dir),
        device="cpu",
        num_samples=2,
    )
    images = [Image.new("RGB", (96, 96), "yellow")]
    torch.manual_seed(42)
    actual = predictor(images, 0)
    # Execute the original if-model_type==nomad body directly from navigate.py.
    source = Path(utils.__file__).with_name("navigate.py").read_text()
    branch = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "model_params['model_type'] == 'nomad'"
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=branch.body, type_ignores=[])),
        "legacy_navigate_branch",
        "exec",
    )
    sink = SimpleNamespace(publish=lambda msg: None)
    ns = dict(
        np=np,
        torch=torch,
        model_params=cfg,
        context_queue=images,
        model=predictor.model,
        device=torch.device("cpu"),
        closest_node=0,
        goal_node=2,
        topomap=load_topomap(topomap_dir),
        args=SimpleNamespace(radius=4, close_threshold=3, num_samples=2, waypoint=0),
        transform_images=utils.transform_images,
        to_numpy=utils.to_numpy,
        get_action=get_action,
        num_diffusion_iters=2,
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=2,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        ),
        Float32MultiArray=Float32MultiArray,
        waypoint_msg=Float32MultiArray(),
        node=SimpleNamespace(sampled_actions_pub=sink, waypoint_pub=sink),
        MAX_V=1,
        RATE=1,
        loop_start_time=0,
        time=SimpleNamespace(time=lambda: 0.0, sleep=lambda _: None),
    )
    torch.manual_seed(42)
    exec(code, ns)
    np.testing.assert_array_equal(actual.points, ns["naction"])
    assert actual.closest_node == ns["closest_node"]


def test_conditioned_checkpoint_is_rejected(inputs):
    cfg, ckpt, topomap = inputs
    state = torch.load(ckpt, weights_only=True)
    state["switchback_state_embedding.weight"] = torch.zeros(2, 16)
    torch.save(state, ckpt)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        NoMaDPredictor(
            model_config=cfg, checkpoint=str(ckpt), topomap=str(topomap), device="cpu"
        )


def test_topomap_must_be_contiguous(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "1.png")
    with pytest.raises(ValueError, match="contiguous"):
        load_topomap(tmp_path)


def test_single_goal_matches_one_image_inference_without_index_arrival(inputs):
    cfg, ckpt, topomap = inputs
    goal = topomap / "0.png"
    # Use the same one-image goal in the legacy path for exact diffusion comparison.
    for path in (topomap / "1.png", topomap / "2.png"):
        path.unlink()
    legacy = NoMaDPredictor(
        model_config=cfg,
        checkpoint=str(ckpt),
        topomap=str(topomap),
        device="cpu",
        num_samples=2,
    )
    single = NoMaDPredictor(
        model_config=cfg,
        checkpoint=str(ckpt),
        goal_image=str(goal),
        goal_distance_threshold=0.0,
        device="cpu",
        num_samples=2,
    )
    images = [Image.new("RGB", (96, 96), "yellow")]
    # Fix only the distance head output; vision encoder and diffusion remain real.
    with torch.no_grad():
        for model in (legacy.model, single.model):
            for parameter in model.dist_pred_net.parameters():
                parameter.zero_()
    torch.manual_seed(42)
    expected = legacy(images, 0)
    torch.manual_seed(42)
    actual = single(images, 99)
    np.testing.assert_array_equal(actual.points, expected.points)
    assert expected.reached_goal
    assert not actual.reached_goal  # strict distance < threshold, never index == 0
    single.goal_distance_threshold = 1.0
    assert single(images, 0).reached_goal


def test_single_goal_rejects_invalid_image_and_ambiguous_sources(inputs, tmp_path):
    cfg, ckpt, topomap = inputs
    with pytest.raises(ValueError, match="exactly one"):
        NoMaDPredictor(
            model_config=cfg,
            checkpoint=str(ckpt),
            topomap=str(topomap),
            goal_image=str(topomap / "0.png"),
        )
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises((ValueError, OSError)):
        NoMaDPredictor(model_config=cfg, checkpoint=str(ckpt), goal_image=str(bad))
