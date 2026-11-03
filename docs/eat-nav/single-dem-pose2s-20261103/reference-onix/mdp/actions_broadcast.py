# actions_broadcast.py
import re
import torch
from typing import Sequence

from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions import JointVelocityActionCfg
from isaaclab.assets import Articulation


@configclass
class BroadcastJointVelocityActionCfg(JointVelocityActionCfg):
    """1スカラーの速度コマンドを、複数ジョイントへブロードキャストする設定。"""
    # このCFGはデフォルトの JointVelocityAction ではなく、下の
    # BroadcastJointVelocityAction を使うようにする。
    # ActionManager は term_cfg.class_type をインスタンス化するため、
    # ここで明示的に差し替える必要がある。
    class_type = None  # 後方でクラス定義後に代入
    broadcast_joint_names: Sequence[str] = ()


class BroadcastJointVelocityAction(ActionTerm):
    def __init__(self, cfg: BroadcastJointVelocityActionCfg, env):
        super().__init__(cfg, env)
        self.cfg: BroadcastJointVelocityActionCfg = cfg
        self._asset: Articulation = env.scene[self.cfg.asset_name]
        self._env = env

        # 代表ジョイント（1つ）
        assert len(self.cfg.joint_names) == 1, "joint_names must contain exactly one representative name"
        rep_name = self.cfg.joint_names[0]

        # Resolve broadcast destination joints using Articulation's resolver to ensure DOF indices match
        # Preserve order so logging/debug maps correctly to names provided in config
        bcast_ids, bcast_names = self._asset.find_joints(self.cfg.broadcast_joint_names, preserve_order=True)
        # Also ensure representative is included in the list
        rep_ids, rep_names = self._asset.find_joints([rep_name], preserve_order=True)
        if len(rep_ids) != 1:
            raise ValueError(f"Representative joint name not found or ambiguous: {rep_name}")
        rep_id = int(rep_ids[0])
        if rep_id not in bcast_ids:
            bcast_ids = torch.as_tensor(list(bcast_ids) + [rep_id], dtype=torch.long, device=self._asset.device)

        # クリップとスケール
        lo, hi = -float("inf"), float("inf")
        if self.cfg.clip:
            for _, (l, h) in self.cfg.clip.items():
                lo = max(lo, l)
                hi = min(hi, h)
        self._clip_lo, self._clip_hi = lo, hi
        self._scale = float(self.cfg.scale)

        self._raw = None
        self._processed = None

        # ★ 初期化フラグ
        self._drives_initialized = False

        # 推定: 左右どちらのトラックか（キャッシュ統合で使用）
        names_str = " ".join(self.cfg.broadcast_joint_names)
        if (
            any(x in names_str for x in ["l_maintrack", "rl_", "fl_", "/rl_", "/fl_", "rl/", "fl/"])
            or re.search(r"chassis_l[0-9]{2}_sprocket_axle", names_str) is not None
            or re.search(r"chassis_(fl|rl)_subroller_[0-9]{2}_axle", names_str) is not None
        ):
            self._side = "left"
        elif (
            any(x in names_str for x in ["r_maintrack", "rr_", "fr_", "/rr_", "/fr_", "rr/", "fr/"])
            or re.search(r"chassis_r[0-9]{2}_sprocket_axle", names_str) is not None
            or re.search(r"chassis_(fr|rr)_subroller_[0-9]{2}_axle", names_str) is not None
        ):
            self._side = "right"
        else:
            self._side = "unknown"

        # Debug: show resolved joints for this action term
        try:
            print(f"[ONIX] {self._side} track broadcast resolves to ids={bcast_ids.tolist()}")
        except Exception:
            pass

        # Store as CPU list for property writes, and as tensor on device for target writes
        self._broadcast_indices: list[int] = [int(i) for i in (bcast_ids.tolist() if hasattr(bcast_ids, 'tolist') else list(bcast_ids))]

    def _backend_device(self):
        """ArticulationView のバックエンド(PhysX tensors)が期待するデバイスを推定。"""
        try:
            bd = self._asset.root_physx_view._backend.device
            # Omniverse 側は -1=CPU, 0+=CUDA device index のことが多い
            if isinstance(bd, int):
                return torch.device("cpu" if bd < 0 else f"cuda:{bd}")
            if isinstance(bd, torch.device):
                return bd
        except Exception:
            pass
        return torch.device("cpu")  # 安全側に倒す


    # ---- 追加: 速度ドライブの有効化（初回のみ） ----
    def _ensure_velocity_drives(self):
        view = self._asset.root_physx_view
        data = self._asset.data
        E    = int(view.count)
        D    = int(self._asset.num_joints)
        cpu  = torch.device("cpu")

        # 変更対象インデックス（CPU）
        idx = torch.as_tensor(self._broadcast_indices, device=cpu, dtype=torch.long)
        all_idx_cpu = torch.arange(D, device=cpu, dtype=torch.long)

        # --- drive types: read-modify-write full matrix then set
        get_drive_types = getattr(view, "get_dof_drive_types", None)
        set_drive_types = getattr(view, "set_dof_drive_types", None)
        if set_drive_types is not None:
            try:
                if get_drive_types is not None:
                    cur = get_drive_types().to(cpu)
                else:
                    cur = torch.zeros((E, D), device=cpu, dtype=torch.int32)
                JT_VEL = 2  # Velocity drive
                cur = cur.clone()
                cur[:, idx] = JT_VEL
                try:
                    set_drive_types(cur)
                except TypeError:
                    set_drive_types(cur, all_idx_cpu)
            except Exception:
                pass

        # --- dof motions ---
        get_dof_motions = getattr(view, "get_dof_motions", None)
        set_dof_motions = getattr(view, "set_dof_motions", None)
        if set_dof_motions is not None:
            try:
                if get_dof_motions is not None:
                    cur = get_dof_motions().to(cpu)
                else:
                    cur = torch.ones((E, D), device=cpu, dtype=torch.int32)
                cur = cur.clone()
                cur[:, idx] = 1  # free
                try:
                    set_dof_motions(cur)
                except TypeError:
                    set_dof_motions(cur, all_idx_cpu)
            except Exception:
                pass

        # --- stiffness / damping / max_forces ---
        def _full_setter(getter_name, setter_name, value):
            g = getattr(view, getter_name, None)
            s = getattr(view, setter_name, None)
            if s is None:
                return
            try:
                if g is not None:
                    cur = g().to(cpu)
                else:
                    cur = torch.zeros((E, D), device=cpu, dtype=torch.float32)
                cur = cur.clone()
                cur[:, idx] = float(value)
                try:
                    s(cur)
                except TypeError:
                    s(cur, all_idx_cpu)
            except Exception:
                pass

        # Set velocity-drive parameters similar to colleague's working config
        _full_setter("get_dof_stiffnesses", "set_dof_stiffnesses", 0.0)
        _full_setter("get_dof_dampings", "set_dof_dampings", 3000.0)
        _full_setter("get_dof_max_forces", "set_dof_max_forces", 3000.0)

        # デバッグ：設定が反映されたか（取れればでOK）
        try:
            gt = getattr(view, "get_dof_drive_types", None)
            gm = getattr(view, "get_dof_motions", None)
            if gt is not None:
                dt = gt()[0].cpu().numpy().tolist()
                print("[ONIX] init drive_types(env0)[:16] =", dt[:16])
            if gm is not None:
                dm = gm()[0].cpu().numpy().tolist()
                print("[ONIX] init motions(env0)[:16]     =", dm[:16])
        except Exception:
            pass

        self._drives_initialized = True


    # ---- 既存 ----
    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._raw = actions
        tgt = (actions[:, 0] * self._scale).clamp(self._clip_lo, self._clip_hi)
        self._processed = tgt.unsqueeze(1)
        # ステージング: 環境に左右のターゲットを一時保存（同ステップでまとめ書き）
        try:
            if not hasattr(self._env, "_onix_bcast_cache"):
                self._env._onix_bcast_cache = {}
            self._env._onix_bcast_cache[self._side] = {
                "idx": torch.as_tensor(self._broadcast_indices, device=self._asset.device, dtype=torch.long),
                "tgt": tgt.detach(),
            }
        except Exception:
            pass
        return self._processed

    def apply_actions(self) -> None:
        if self._processed is None:
            return

        # ★ 初回だけ、対象12関節を速度ドライブに強制設定
        if not self._drives_initialized:
            self._ensure_velocity_drives()

        # 代表スカラーから対象K関節分の速度配列を作成し、Articulation APIで設定
        processed = self._processed
        latency_fn = getattr(self._env, "apply_action_latency", None)
        if callable(latency_fn):
            term_name = f"{self._side}_track_vel" if self._side in ("left", "right") else "track_vel"
            processed = latency_fn(term_name, processed)
        tgt = processed[:, 0]  # [E]
        dev = self._asset.device
        idx = torch.as_tensor(self._broadcast_indices, device=dev, dtype=torch.long)
        vals = tgt.unsqueeze(1).expand(-1, idx.numel()).contiguous()  # [E, K]
        self._asset.set_joint_velocity_target(vals, joint_ids=idx)



    @property
    def raw_actions(self) -> torch.Tensor | None:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor | None:
        return self._processed

# Bind cfg to its ActionTerm implementation so the ActionManager instantiates
# the broadcast-capable action instead of the default joint-velocity term.
BroadcastJointVelocityActionCfg.class_type = BroadcastJointVelocityAction
