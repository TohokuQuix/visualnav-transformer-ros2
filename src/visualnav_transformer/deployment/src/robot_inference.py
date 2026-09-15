"""Existing NoMaD inference, isolated from ROS callbacks and output scaling."""

from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image
from visualnav_transformer.deployment.src.robot_interface import Prediction
from visualnav_transformer.deployment.src.utils import (
    load_model,
    to_numpy,
    transform_images,
)
from visualnav_transformer.train.vint_train.training.train_utils import get_action


def load_topomap(directory):
    files = [
        p
        for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    if not files or any(not p.stem.isdecimal() for p in files):
        raise ValueError("Topomap requires numbered images")
    files.sort(key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(len(files))):
        raise ValueError("Topomap image numbers must be contiguous from zero")
    images = []
    for path in files:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


class NoMaDPredictor:
    def __init__(
        self,
        *,
        model_config,
        checkpoint,
        topomap,
        goal_node=-1,
        radius=4,
        close_threshold=3.0,
        num_samples=8,
        device="auto"
    ):
        self.cfg = model_config
        if model_config.get("model_type") != "nomad":
            raise ValueError("Robot interface supports NoMaD only")
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.topomap = load_topomap(topomap)
        self.goal_node = len(self.topomap) - 1 if goal_node == -1 else goal_node
        if not 0 <= self.goal_node < len(self.topomap):
            raise ValueError("goal_node out of range")
        self.radius = radius
        self.close_threshold = close_threshold
        self.num_samples = num_samples
        self.model = load_model(
            checkpoint, model_config, self.device, strict=True
        ).eval()
        self.scheduler = DDPMScheduler(
            num_train_timesteps=model_config["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def __call__(self, images, closest_node):
        cfg = self.cfg
        obs_images = transform_images(images, cfg["image_size"], center_crop=False)
        obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(self.device)
        start = max(closest_node - self.radius, 0)
        end = min(closest_node + self.radius + 1, self.goal_node)
        goals = torch.concat(
            [
                transform_images(g, cfg["image_size"], center_crop=False).to(
                    self.device
                )
                for g in self.topomap[start : end + 1]
            ],
            dim=0,
        )
        mask = torch.zeros(1).long().to(self.device)
        cond = self.model(
            "vision_encoder",
            obs_img=obs_images.repeat(len(goals), 1, 1, 1),
            goal_img=goals,
            input_goal_mask=mask.repeat(len(goals)),
        )
        distances = to_numpy(self.model("dist_pred_net", obsgoal_cond=cond).flatten())
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite goal distances")
        nearest = int(np.argmin(distances))
        closest_node = nearest + start
        subgoal = min(
            nearest + int(distances[nearest] < self.close_threshold), len(cond) - 1
        )
        obs_cond = cond[subgoal].unsqueeze(0)
        # Preserve the legacy gradient context (including Transformer kernel selection).
        with torch.no_grad():
            obs_cond = (
                obs_cond.repeat(self.num_samples, 1)
                if len(obs_cond.shape) == 2
                else obs_cond.repeat(self.num_samples, 1, 1)
            )
            action = torch.randn(
                (self.num_samples, cfg["len_traj_pred"], 2), device=self.device
            )
            self.scheduler.set_timesteps(cfg["num_diffusion_iters"])
            for timestep in self.scheduler.timesteps:
                noise = self.model(
                    "noise_pred_net",
                    sample=action,
                    timestep=timestep,
                    global_cond=obs_cond,
                )
                action = self.scheduler.step(
                    model_output=noise, timestep=timestep, sample=action
                ).prev_sample
        points = to_numpy(get_action(action))[0]
        return Prediction(points, closest_node, closest_node == self.goal_node)
