from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import BpearlPatternCfg
from isaaclab.sensors import patterns

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from onix_isaac_lab.tasks.locomotion.velocity import mdp

from onix_isaac_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    RewardsCfg,
    ObservationsCfg,
)
from onix_isaac_lab.assets.onix import ONIX_CFG, ONIX_JOINT_NAMES, ONIX_CONTROL_JOINT_NAMES

from isaaclab.envs.mdp.actions import JointVelocityActionCfg

##
# Pre-defined configs
##

SUPPORT_ROLLER_BODY_NAMES = (
    *(f"chassis_l{i:02d}_sprocket" for i in range(1, 9)),
    *(f"chassis_r{i:02d}_sprocket" for i in range(1, 9)),
    *(f"chassis_fl_subroller_{i:02d}" for i in range(1, 9)),
    *(f"chassis_fr_subroller_{i:02d}" for i in range(1, 9)),
    *(f"chassis_rl_subroller_{i:02d}" for i in range(1, 9)),
    *(f"chassis_rr_subroller_{i:02d}" for i in range(1, 9)),
)

FLIPPER_OBS_JOINT_NAMES = (
    ".*rr.*flipper.*",
    ".*rl.*flipper.*",
    ".*fr.*flipper.*",
    ".*fl.*flipper.*",
)
RIGHT_TRACK_OBS_JOINT_NAMES = (
    *(f"chassis_r{i:02d}_sprocket_axle" for i in range(1, 9)),
    *(f"chassis_fr_subroller_{i:02d}_axle" for i in range(1, 9)),
    *(f"chassis_rr_subroller_{i:02d}_axle" for i in range(1, 9)),
)
LEFT_TRACK_OBS_JOINT_NAMES = (
    *(f"chassis_l{i:02d}_sprocket_axle" for i in range(1, 9)),
    *(f"chassis_fl_subroller_{i:02d}_axle" for i in range(1, 9)),
    *(f"chassis_rl_subroller_{i:02d}_axle" for i in range(1, 9)),
)
FLIPPER_MIN_ANGLE = -1.57079632679
DIRECTIONAL_FLIPPER_FOLD_MIN = 3.14159265359
FLIPPER_MAX_ANGLE = 4.71238898038
FLIPPER_FOLD_THRESHOLD = 2.09439510239
FLIPPER_CLEAR_ANGLE = 1.57079632679
FLIPPER_FOLDED_INIT_ANGLE = 3.14159265359
FLIPPER_GROUND_REWARD_ANGLE = 3.14159265359
K_RAIL_MAX_SLOPE_DEG = 8.0 * 2.0 / 3.0
FLIPPER_CONTACT_BODY_NAMES = (
    *(f"chassis_rr_subroller_{i:02d}" for i in range(1, 9)),
    *(f"chassis_rl_subroller_{i:02d}" for i in range(1, 9)),
    *(f"chassis_fr_subroller_{i:02d}" for i in range(1, 9)),
    *(f"chassis_fl_subroller_{i:02d}" for i in range(1, 9)),
)


def _make_observation_joint_cfgs() -> tuple[SceneEntityCfg, SceneEntityCfg, SceneEntityCfg]:
    flipper_cfg = SceneEntityCfg("robot", joint_names=FLIPPER_OBS_JOINT_NAMES, preserve_order=True)
    right_track_cfg = SceneEntityCfg("robot", joint_names=RIGHT_TRACK_OBS_JOINT_NAMES, preserve_order=True)
    left_track_cfg = SceneEntityCfg("robot", joint_names=LEFT_TRACK_OBS_JOINT_NAMES, preserve_order=True)
    return flipper_cfg, right_track_cfg, left_track_cfg


def _make_joint_effort_obs_term() -> ObsTerm:
    flipper_cfg, right_track_cfg, left_track_cfg = _make_observation_joint_cfgs()
    return ObsTerm(
        func=mdp.joint_effort_with_track_sum,
        params={
            "flipper_cfg": flipper_cfg,
            "right_track_cfg": right_track_cfg,
            "left_track_cfg": left_track_cfg,
        },
        scale=0.01,
        clip=(-100.0, 100.0),
    )


def _make_height_scan_obs_term(sensor_name: str) -> ObsTerm:
    return ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg(sensor_name)},
        clip=(-1.0, 1.0),
        scale=1.0,
    )


@configclass
class OnixRewardsCfg(RewardsCfg):
    # ── 1) まずは極端に強い終端ペナルティを一旦弱める（立ち上がり用）
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)

    # ── 2) 前後速度（x）のみ追従。モジュールに x 専用 term がある場合はこちらを推奨
    # 例: mdp.track_lin_vel_x_yaw_frame_exp が存在する場合
    try:
        track_lin_vel_x_exp = RewTerm(
            func=mdp.track_lin_vel_x_yaw_frame_exp,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "std": 1.0,
                "yaw_command_threshold": 0.15,
                "yaw_error_threshold": 0.2,
                "yaw_error_relax_factor": 0.35,
            },
        )
    except AttributeError:
        # フォールバック: 既存の xy 追従を使用。ただし env 側で lin_vel_y を (0,0) に固定する前提
        track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "std": 1.0,
                "yaw_relax_threshold": 0.05,
                "yaw_relax_factor": 0.25,
            },
        )

    # ── 3) 旋回（yaw）追従
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 1.0},
    )

    standstill_reward = RewTerm(
        func=mdp.standstill_reward,
        weight=1.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.1, "std": 0.25},
    )

    # ── 4) 横滑りはペナルティ（横移動を目標にしない代わりに抑制）
    # 利用可能な名前に合わせて（lin_vel_y_l2 / lin_vel_y_abs など）使える方に置き換えてください。
    try:
        lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.5)
    except AttributeError:
        pass  # 無ければ一旦スキップ（後述のコマンド固定だけでも効果は出ます）

    flipper_symmetry_forward_l2 = RewTerm(
        func=mdp.flipper_symmetry_forward_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "forward_threshold": 0.2,
            "yaw_threshold": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*fl.*flipper.*",
                    ".*fr.*flipper.*",
                    ".*rl.*flipper.*",
                    ".*rr.*flipper.*",
                ],
                preserve_order=True,
            ),
        },
    )

    directional_flipper_fold_reward = RewTerm(
        func=mdp.directional_flipper_fold_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "fold_start": FLIPPER_FOLD_THRESHOLD,
            "fold_end": FLIPPER_GROUND_REWARD_ANGLE,
            "min_speed": 0.05,
            "yaw_scale": 0.50,
            "pre_fold_weight": 0.30,
            "yaw_tracking_std": 0.30,
            "lin_tracking_std": 0.50,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=FLIPPER_OBS_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    directional_folded_flipper_contact_reward = RewTerm(
        func=mdp.directional_folded_flipper_contact_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FLIPPER_CONTACT_BODY_NAMES,
                preserve_order=True,
            ),
            "flipper_cfg": SceneEntityCfg(
                "robot",
                joint_names=FLIPPER_OBS_JOINT_NAMES,
                preserve_order=True,
            ),
            "fold_start": FLIPPER_FOLD_THRESHOLD,
            "fold_end": FLIPPER_GROUND_REWARD_ANGLE,
            "min_speed": 0.05,
            "contact_threshold": 1.0,
            "bodies_per_flipper": 8,
            "yaw_tracking_std": 0.30,
            "lin_tracking_std": 0.50,
        },
    )

    directional_unfolded_flipper_contact_reward = RewTerm(
        func=mdp.directional_unfolded_flipper_contact_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FLIPPER_CONTACT_BODY_NAMES,
                preserve_order=True,
            ),
            "flipper_cfg": SceneEntityCfg(
                "robot",
                joint_names=FLIPPER_OBS_JOINT_NAMES,
                preserve_order=True,
            ),
            "fold_start": FLIPPER_FOLD_THRESHOLD,
            "clear_end": FLIPPER_CLEAR_ANGLE,
            "min_speed": 0.05,
            "contact_threshold": 1.0,
            "bodies_per_flipper": 8,
            "yaw_tracking_std": 0.30,
            "lin_tracking_std": 0.50,
        },
    )

    directional_unfolded_flipper_angle_reward = RewTerm(
        func=mdp.directional_unfolded_flipper_angle_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "flipper_cfg": SceneEntityCfg(
                "robot",
                joint_names=FLIPPER_OBS_JOINT_NAMES,
                preserve_order=True,
            ),
            "fold_start": FLIPPER_FOLD_THRESHOLD,
            "clear_end": FLIPPER_CLEAR_ANGLE,
            "min_speed": 0.05,
        },
    )

    flipper_fold_mode_angle_reward = RewTerm(
        func=mdp.flipper_fold_mode_angle_reward,
        weight=0.25,
        params={
            "action_term_name": "flipper_pos",
            "fold_min": DIRECTIONAL_FLIPPER_FOLD_MIN,
            "clear_max": FLIPPER_CLEAR_ANGLE,
            "min_angle": FLIPPER_MIN_ANGLE,
            "max_angle": FLIPPER_MAX_ANGLE,
            "mode_deadband": 0.5,
            "distance_std": 0.5,
        },
    )

    flipper_fold_mode_contact_reward = RewTerm(
        func=mdp.flipper_fold_mode_contact_reward,
        weight=0.4,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FLIPPER_CONTACT_BODY_NAMES,
                preserve_order=True,
            ),
            "flipper_cfg": SceneEntityCfg(
                "robot",
                joint_names=FLIPPER_OBS_JOINT_NAMES,
                preserve_order=True,
            ),
            "fold_start": FLIPPER_FOLD_THRESHOLD,
            "fold_end": FLIPPER_GROUND_REWARD_ANGLE,
            "contact_threshold": 1.0,
            "bodies_per_flipper": 8,
            "mode_deadband": 0.5,
        },
    )

    flipper_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=FLIPPER_OBS_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    roller_contact_count_bonus = RewTerm(
        func=mdp.contact_count_bonus,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=SUPPORT_ROLLER_BODY_NAMES,
                preserve_order=True,
            ),
            "threshold": 1.0,
        },
    )

    support_margin_reward = RewTerm(
        func=mdp.support_margin_quad,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=SUPPORT_ROLLER_BODY_NAMES,
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=SUPPORT_ROLLER_BODY_NAMES,
                preserve_order=True,
            ),
            "threshold": 1.0,
            "margin_scale": 0.12,
            "margin_cap": 0.25,
        },
    )

    # ── 5) 残りの安定化項（多くは RewardsCfg から継承されて有効）
    #   - lin_vel_z_l2, ang_vel_xy_l2, dof_torques_l2, dof_acc_l2,
    #     action_rate_l2, flat_orientation_l2, dof_pos_limits など
    def __post_init__(self):
        super().__post_init__()
        # もし継承側で重みを触れるなら、参考までの値（過去ログと同等）
        # すでに基底で設定済みならこのブロックは省略可
        if hasattr(self, "lin_vel_z_l2"):
            self.lin_vel_z_l2.weight = 0.0 #-2.0e-5 #-2.0
        if hasattr(self, "ang_vel_xy_l2"):
            self.ang_vel_xy_l2.weight = -0.08
        if hasattr(self, "dof_torques_l2"):
            self.dof_torques_l2.weight = 0.0 #-1.0e-9
        if hasattr(self, "dof_acc_l2"):
            self.dof_acc_l2.weight = 0.0 #-2.5e-12
        if hasattr(self, "action_rate_l2"):
            # Let the policy explore large flipper moves into the folded range.
            self.action_rate_l2.weight = -0.01
        if hasattr(self, "flat_orientation_l2"):
            self.flat_orientation_l2.weight = 0.0
        if hasattr(self, "dof_pos_limits"):
            self.dof_pos_limits.weight = 0.0 #-5.0e-2 #-5.0


@configclass
class OnixSceneCfg(MySceneCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # set the robot as onix
        self.robot = ONIX_CFG.replace(prim_path="{ENV_REGEX_NS}/onix_chassis")
        # Keep track actuators so DOF drives exist at spawn. We command velocities via
        # JointVelocityAction; actuators here set drive properties only.
        # self.contact_forces = None
        # self.height_scanner = None

        # terrain parameter settings
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.max_init_terrain_level = 5
        # 列=7（7種類×1列）に設定
        self.terrain.terrain_generator.num_cols = 7
        self.terrain.terrain_generator.sub_terrains["k_rail_field"] = mdp.MeshKRailFieldTerrainCfg(
            proportion=1.0 / 7.0,
            random_sampling=False,
            randomize_slope_phase=False,
            max_slope_deg=K_RAIL_MAX_SLOPE_DEG,
            slope_noise_deg=0.0,
            height_noise_fraction=0.0,
        )
        self.terrain.terrain_generator.sub_terrains["k_rail_step_field"] = mdp.MeshKRailStepFieldTerrainCfg(
            proportion=1.0 / 7.0,
            random_sampling=False,
            max_slope_deg=K_RAIL_MAX_SLOPE_DEG,
        )
        self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (
            0.025,
            0.10,
        )
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (
            0.01,
            0.09,
        )
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.terrain.terrain_generator.sub_terrains[
            "pyramid_stairs"
        ].step_height_range = (0.02, 0.20)
        self.terrain.terrain_generator.sub_terrains[
            "pyramid_stairs_inv"
        ].step_height_range = (0.02, 0.20)

        # 各サブ地形の列の占有率（7種類×1列）
        terrain_proportion = 1.0 / 7.0
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = terrain_proportion
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = terrain_proportion
        self.terrain.terrain_generator.sub_terrains["random_rough"].proportion = terrain_proportion
        self.terrain.terrain_generator.sub_terrains["boxes"].proportion = terrain_proportion
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = terrain_proportion
        self.terrain.terrain_generator.sub_terrains["k_rail_field"].proportion = terrain_proportion
        self.terrain.terrain_generator.sub_terrains["k_rail_step_field"].proportion = terrain_proportion
        # 使わない種類は0に
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        # Actor height scanner. Keep it identical to the critic scanner so both
        # networks consume the same height-map configuration when enabled.
        self.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/onix_chassis/chassis_body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        # Critic height scanner, kept in sync with the actor scanner.
        self.height_scanner_critic = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/onix_chassis/chassis_body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # ① ContactSensor を軽量化
        if hasattr(self, "contact_forces") and self.contact_forces is not None:
            # 履歴メモリと処理を最小化
            self.contact_forces.history_length = 1
            self.contact_forces.track_air_time = False


@configclass
class OnixObservationsCfg(ObservationsCfg):
    @configclass
    class CommonCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()

            # 観測のjoint状態は、位置はフリッパ4軸のみ、
            # 速度は左右クローラ群の平均速度のみを使う。
            flipper_cfg, right_track_cfg, left_track_cfg = _make_observation_joint_cfgs()

            self.joint_vel = ObsTerm(
                func=mdp.track_mean_velocity,
                params={
                    "right_track_cfg": right_track_cfg,
                    "left_track_cfg": left_track_cfg,
                },
            )
            self.joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={
                    "asset_cfg": flipper_cfg,
                },
            )
            self.flipper_fold_mode = ObsTerm(
                func=mdp.flipper_fold_mode,
                scale=1.0,
                clip=(-1.0, 1.0),
            )
            # 姿勢（重力ベクトル）をネットに分かりやすくするため明示スケール/クリップ
            # projected_gravity の各成分は概ね [-1, 1] に収まるため、スケール1〜2で十分。
            # クリップは[-1,1]に設定して外れ値の影響を抑える。
            if hasattr(self, "projected_gravity"):
                self.projected_gravity.scale = 1.0
                self.projected_gravity.clip = (-1.0, 1.0)

            # scale observations
            self.joint_pos.scale = 1.0
            self.joint_vel.scale = 0.05
            self.base_lin_vel.scale = 2.0
            self.base_ang_vel.scale = 0.25

            # clip observations
            self.base_lin_vel.clip = (-100.0, 100.0)
            self.base_ang_vel.clip = (-100.0, 100.0)
            self.velocity_commands.clip = (-100.0, 100.0)
            self.joint_pos.clip = (-100.0, 100.0)
            self.joint_vel.clip = (-100.0, 100.0)

    @configclass
    class PolicyCfg(CommonCfg):
        height_scan = None

        def __post_init__(self):
            super().__post_init__()

            # add noise to the observations
            self.enable_corruption = True
            self.base_lin_vel = None
            self.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
            self.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
            self.joint_vel.noise = Unoise(n_min=-1.5, n_max=1.5)
            self.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)

    @configclass
    class CriticCfg(CommonCfg):
        control_latency = ObsTerm(
            func=mdp.control_latency,
            clip=(0.0, 1.0),
        )
        roller_contact_binary = ObsTerm(
            func=mdp.contact_binary,
            scale=1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        *(f"chassis_l{i:02d}_sprocket" for i in range(1, 9)),
                        *(f"chassis_r{i:02d}_sprocket" for i in range(1, 9)),
                        *(f"chassis_fl_subroller_{i:02d}" for i in range(1, 9)),
                        *(f"chassis_fr_subroller_{i:02d}" for i in range(1, 9)),
                        *(f"chassis_rl_subroller_{i:02d}" for i in range(1, 9)),
                        *(f"chassis_rr_subroller_{i:02d}" for i in range(1, 9)),
                    ],
                    preserve_order=True,
                ),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            super().__post_init__()

            # disable noise for critic observations
            self.enable_corruption = False

            # change offset for base height scan
            # self.height_scan.params["offset"] = 0.3

    policy = PolicyCfg()
    critic = CriticCfg()

    def set_observation_toggles(self, use_last_action: bool, use_joint_effort: bool, use_height_scan: bool):
        # Policy: toggled per run (4-ray scanner).
        self.policy.actions = ObsTerm(func=mdp.last_action) if use_last_action else None
        self.policy.joint_effort = _make_joint_effort_obs_term() if use_joint_effort else None
        self.policy.height_scan = _make_height_scan_obs_term("height_scanner") if use_height_scan else None
        # Critic: always get last_action and dense height scan (ANYmal-like).
        self.critic.actions = ObsTerm(func=mdp.last_action)
        self.critic.joint_effort = _make_joint_effort_obs_term() if use_joint_effort else None
        self.critic.height_scan = _make_height_scan_obs_term("height_scanner_critic")


@configclass
class OnixRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: OnixSceneCfg = OnixSceneCfg(num_envs=256, env_spacing=2.5)
    rewards: OnixRewardsCfg = OnixRewardsCfg()
    observations: OnixObservationsCfg = OnixObservationsCfg()
    use_last_action_obs: bool = False
    use_joint_effort_obs: bool = True
    use_height_scan_obs: bool = False
    latency_randomization_enabled: bool = True
    latency_range_s: tuple[float, float] = (0.015, 0.035)
    latency_delay_actions: bool = True
    latency_delay_policy_observations: bool = True
    viewer: ViewerCfg = ViewerCfg(
        eye=(60.0, -52.0, 34.0),
        lookat=(10.0, 0.0, 3.0),
        origin_type="world",
        env_index=0,
        asset_name="robot",
        resolution=(1920, 1080),
    )

    def __post_init__(self):
        super().__post_init__()
        self.observations.set_observation_toggles(
            use_last_action=self.use_last_action_obs,
            use_joint_effort=self.use_joint_effort_obs,
            use_height_scan=self.use_height_scan_obs,
        )
        if getattr(self.scene, "height_scanner", None) is not None:
            if self.use_height_scan_obs:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            else:
                self.scene.height_scanner.update_period = float("inf")
        if getattr(self.scene, "height_scanner_critic", None) is not None:
            self.scene.height_scanner_critic.update_period = self.decimation * self.sim.dt
        # ① 既存アクションを空にする（どちらか片方だけでOK）
        # A) 置き換え
        # self.actions = ActionsCfg()
        # B) 個別にNone化（こっちのほうが互換性高め）
        for name, val in list(vars(self.actions).items()):
            if not name.startswith("_") and val is not None:
                setattr(self.actions, name, None)
        # =========================
        # 6軸アクション定義（実機I/Fと一致）
        # =========================
        # --- フリッパ（位置）4軸 ---
        flipper_joints = (
            ".*rr.*flipper.*", ".*rl.*flipper.*", ".*fr.*flipper.*", ".*fl.*flipper.*"
        )
        # Keep sub-crawler (flipper) commands in a symmetric non-interference envelope.
        self.actions.flipper_pos = mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            class_type=mdp.RateLimitedJointPositionAction,
            joint_names=list(flipper_joints),
            # 送る値（rad）に掛かる係数。実値そのまま送りたいなら 1.0
            scale=1.0,
            max_velocity=2.0,
            interference_limit=FLIPPER_FOLD_THRESHOLD,
            interference_pairs=((0, 2), (1, 3)),
            # ソフトクリップ（制限したい範囲に調整）
            clip={
                ".*(rr|rl).*flipper.*": (FLIPPER_MIN_ANGLE, FLIPPER_MAX_ANGLE),
                ".*(fr|fl).*flipper.*": (FLIPPER_MIN_ANGLE, FLIPPER_MAX_ANGLE),
            },
            use_directional_clip=True,
            directional_clip_command_name="base_velocity",
            directional_clip_speed_threshold=0.10,
            directional_clip_fold_min=DIRECTIONAL_FLIPPER_FOLD_MIN,
            directional_clip_clear_max=FLIPPER_CLEAR_ANGLE,
            directional_clip_mode_deadband=0.5,
            preserve_order=True,
        )

        # アクション定義部（Broadcast: 1スカラー→複数DOF へ配信）
        right_track = (
            *(f"chassis_r{i:02d}_sprocket_axle" for i in range(1, 9)),
            *(f"chassis_fr_subroller_{i:02d}_axle" for i in range(1, 9)),
            *(f"chassis_rr_subroller_{i:02d}_axle" for i in range(1, 9)),
        )
        left_track = (
            *(f"chassis_l{i:02d}_sprocket_axle" for i in range(1, 9)),
            *(f"chassis_fl_subroller_{i:02d}_axle" for i in range(1, 9)),
            *(f"chassis_rl_subroller_{i:02d}_axle" for i in range(1, 9)),
        )

        # Target ~1.0 m/s upper bound at sprocket radius 0.0703 m -> omega ~= 14.2 rad/s.
        track_omega_limit = 14.2
        self.actions.right_track_vel = mdp.BroadcastJointVelocityActionCfg(
            asset_name="robot",
            class_type=mdp.BroadcastJointVelocityAction,  # 明示: 生成クラス
            joint_names=["chassis_r01_sprocket_axle"],
            broadcast_joint_names=list(right_track),
            scale=track_omega_limit,
            clip={".*": (-track_omega_limit, track_omega_limit)},
            preserve_order=True,
            use_default_offset=False,
        )
        self.actions.left_track_vel = mdp.BroadcastJointVelocityActionCfg(
            asset_name="robot",
            class_type=mdp.BroadcastJointVelocityAction,  # 明示: 生成クラス
            joint_names=["chassis_l01_sprocket_axle"],
            broadcast_joint_names=list(left_track),
            scale=track_omega_limit,
            clip={".*": (-track_omega_limit, track_omega_limit)},
            preserve_order=True,
            use_default_offset=False,
        )

        # --- 旧28軸の位置アクションを無効化 ---
        self.actions.right_track_pos = None
        self.actions.left_track_pos  = None
        self.actions.joint_pos       = None

        # events
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["num_buckets"] = 128
        self.events.physics_material.params["make_consistent"] = True
        self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = ".*body|chassis_top_box"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ".*body|chassis_top_box"
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_with_flipper_pose_mix,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "flipper_joint_names": list(flipper_joints),
                "pose_probabilities": (0.0, 0.50, 0.50),
                "folded_angle": FLIPPER_FOLDED_INIT_ANGLE,
                "front_rest_angle": 0.0,
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.reset_flipper_fold_mode = EventTerm(
            func=mdp.sample_flipper_fold_mode,
            mode="reset",
            params={
                "modes": (0.0, 1.0, -1.0),
                "probabilities": (1.0, 1.0, 1.0),
            },
        )
        # Dense track/subroller contacts need more GPU narrowphase memory than the template defaults.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14159, 3.14159)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
        }

        # Startup: ensure velocity drive parameters on all track DOFs
        track_all = list(right_track) + list(left_track)
        self.events.set_track_velocity_drive = EventTerm(
            func=mdp.set_velocity_drive_for_joints,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_names": track_all,
                "damping": 3000.0,
                "max_force": 3000.0,
                "stiffness": 0.0,
            },
        )
        # Keep self-collision enabled globally, but ignore sprocket-vs-sprocket only.
        self.events.ignore_sprocket_self_collision = EventTerm(
            func=mdp.ignore_self_collision_for_sprockets,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sprocket_body_pattern": r"chassis_[lr][0-9]{2}_sprocket$",
                "track_body_pattern": r"chassis_(left|right|fl|fr|rl|rr)_track$",
                "aux_body_pattern": r"chassis(_(fl|fr|rl|rr))?_body$",
                "allow_multi_env": False,
            },
        )
        # commands
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 3.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.lin_vel_x_bias_power = 1.0
        self.commands.base_velocity.ang_vel_z_bias_power = 2.0
        self.commands.base_velocity.stair_terrain_names = ("pyramid_stairs", "pyramid_stairs_inv")
        self.commands.base_velocity.high_difficulty_level = 4
        self.commands.base_velocity.ang_vel_scale_high_stairs = 1.0

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = [
        #     "body",
        #     ".*_scapula",
        #     ".*_thigh",
        # ]
        # 90°以上の傾きでエピソード終了
        self.terminations.tilt_over = TermTerm(
            func=mdp.tilt_over_threshold,
            params={"angle_deg": 85.0},  # 必要なら 80.0 等に調整
        )

        # ① ContactSensor の更新頻度を低下（物理負荷軽減）
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt


@configclass
class OnixRoughEnvCfg_PLAY(OnixRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = 0
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            # PLAYでも7種類×1列
            self.scene.terrain.terrain_generator.num_cols = 7
            st = self.scene.terrain.terrain_generator.sub_terrains
            if "k_rail_field" not in st:
                st["k_rail_field"] = mdp.MeshKRailFieldTerrainCfg(
                    proportion=1.0 / 7.0,
                    random_sampling=False,
                    randomize_slope_phase=False,
                    max_slope_deg=K_RAIL_MAX_SLOPE_DEG,
                    slope_noise_deg=0.0,
                    height_noise_fraction=0.0,
                )
            else:
                st["k_rail_field"].random_sampling = False
                st["k_rail_field"].randomize_slope_phase = False
                st["k_rail_field"].max_slope_deg = K_RAIL_MAX_SLOPE_DEG
                st["k_rail_field"].slope_noise_deg = 0.0
                st["k_rail_field"].height_noise_fraction = 0.0
            if "k_rail_step_field" not in st:
                st["k_rail_step_field"] = mdp.MeshKRailStepFieldTerrainCfg(
                    proportion=1.0 / 7.0,
                    random_sampling=False,
                    max_slope_deg=K_RAIL_MAX_SLOPE_DEG,
                )
            else:
                st["k_rail_step_field"].random_sampling = False
                st["k_rail_step_field"].max_slope_deg = K_RAIL_MAX_SLOPE_DEG
                st["k_rail_step_field"].slope_noise_deg = 0.0
                st["k_rail_step_field"].height_noise_fraction = 0.0
            # 占有率も通常と同様に7等分（丘を復活）
            terrain_proportion = 1.0 / 7.0
            st["pyramid_stairs"].proportion = terrain_proportion
            st["pyramid_stairs_inv"].proportion = terrain_proportion
            st["random_rough"].proportion = terrain_proportion
            st["boxes"].proportion = terrain_proportion
            st["hf_pyramid_slope"].proportion = terrain_proportion
            st["k_rail_field"].proportion = terrain_proportion
            st["k_rail_step_field"].proportion = terrain_proportion
            st["hf_pyramid_slope_inv"].proportion = 0.0
            # 難易度レンジを上方拡張（最大難易度を約1.5倍）
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.5)
            self.scene.terrain.terrain_generator.curriculum = True #False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # TEST: disable all ActionManager terms so external --direct_vel is not overridden
        for name, val in list(vars(self.actions).items()):
            if not name.startswith("_") and val is not None:
                setattr(self.actions, name, None)
