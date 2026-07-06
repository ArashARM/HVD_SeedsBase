from dataclasses import asdict, dataclass
import csv
import importlib
import json
import math
import os
import shutil
import time
from datetime import datetime
from typing import Any

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from Utils.TimelapseRecorder import TimelapseRecorder
from tqdm.auto import tqdm
import numpy as np
from Utils.DifferentiableFilters import (
    smooth_heaviside_projection,
    surface_density_filter_metric_aware,
)

import pyvista as pv
try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass

import matplotlib.pyplot as plt

try:
    from .Loss_Boundary import Loss_Boundary
    from .Loss_FEM import Loss_FEM
    from .Loss_Volume import Loss_Volume
    from .Loss_rep import Loss_rep
except ImportError:
    from Loss_Boundary import Loss_Boundary
    from Loss_FEM import Loss_FEM
    from Loss_Volume import Loss_Volume
    from Loss_rep import Loss_rep


def compute_w_min_from_min_feature_size_3d(
    Xu: torch.Tensor,
    Xv: torch.Tensor,
    min_feature_size_3d: float,
    safety_factor: float = 1.0,
    stat: str = "median",
    eps: float = 1e-8,
) -> float:
    """
    Convert printer minimum printable full-width in 3D units
    into decoder UV half-width w_min.

    VoronoiDecoder.w_min is a UV half-width.
    min_feature_size_3d is a 3D full-width.
    """
    if min_feature_size_3d <= 0:
        raise ValueError("min_feature_size_3d must be > 0")

    Xu_norm = torch.linalg.norm(Xu, dim=-1)
    Xv_norm = torch.linalg.norm(Xv, dim=-1)

    local_scale = torch.minimum(Xu_norm, Xv_norm)
    local_scale = local_scale[torch.isfinite(local_scale)]
    local_scale = local_scale[local_scale > eps]

    if local_scale.numel() == 0:
        raise ValueError("Could not compute valid UV-to-3D scale from Xu/Xv")

    if stat == "median":
        scale = local_scale.median()
    elif stat == "mean":
        scale = local_scale.mean()
    elif stat == "min":
        scale = local_scale.min()
    else:
        raise ValueError("stat must be one of: median, mean, min")

    min_radius_3d = 0.5 * float(min_feature_size_3d) * float(safety_factor)
    w_min_uv = min_radius_3d / scale.clamp_min(eps)


    return float(w_min_uv.detach().cpu())

@dataclass
class TrainingConfig:
    seed_init_fps_seed: int | None = None
    use_balanced_seed_init: bool = True
    seed_number: int = 15
    training_face_index: int = 0
    LoadingCasee: str = "Unspecified loading case"

    target_volfrac: float = 0.5
    seed_repulsion_sigma: float = 0.08
    boundary_margin: float = 0.05

    freeze_w: bool = False
    w_const: float = 0.25
    width_target_frac: float = 0.20
    decoder_raw_temp: float = 1.25
    w_head_bias_init: float | None = None
    w_min: float = 0.005
    w_max_ratio: float = 0.5
    min_feature_size_3d: float | None = None
    auto_update_wmin: bool = False

    beta: float = 0.05
    centerline_beta: float = 0.02
    centerline_softmin_tau: float = 0.01
    tube_curve_samples: int = 64
    tube_lift_tau: float = 0.02
    tube_lift_max_values: int = 4_000_000
    rho_min: float = 0.0
    decoder_eps: float = 1e-8
    decoder_solve_reg: float = 1e-6
    decoder_tau_voronoi: float = 0.01
    decoder_tau_box: float = 0.01
    decoder_tau_trim: float = 0.01
    decoder_use_trim_activity: bool = True
    decoder_return_xyz: bool = True
    decoder_vertex_boundary_margin: float = 0.02
    decoder_edge_trim_samples: int = 32
    decoder_edge_trim_reduction: str = "softmin"
    decoder_edge_trim_reduce_tau: float = 0.05
    decoder_use_edge_trim_gate: bool = True
    decoder_nearest_segment_k: int = 4
    decoder_use_segment_distance: bool = True
    decoder_use_spatial_pruning: bool = True
    decoder_min_tube_spacing: float = 1e-3
    decoder_tube_target_spacing_ratio: float = 0.75
    decoder_use_seed_activation: bool = True
    decoder_duplicate_merge_sigma: float = 1e-4
    decoder_duplicate_effect_temp_ratio: float = 0.25

    use_3d_density_filter: bool = False
    filter_radius_3d: float = 0.03
    filter_self_weight: float = 1.0
    filter_projection_strength: float = 1.0
    filter_projection_beta: float = 8.0
    filter_projection_eta: float = 0.5
    visualize_filtered_density: bool = True
    visualize_raw_density: bool = False
    generate_decoder_density_fiber: bool = True

    lam_fem: float = 1.0
    lam_vol: float = 2.0
    lam_rep: float = 2.0
    lam_bnd: float = 0.5
    lam_vol_effective: float = 0.5
    effective_volume_power: float = 2.0
    lam_curve_length: float = 1.0
    curve_length_target: float | None = None
    curve_length_loss_type: str = "pairwise"
    curve_length_eps: float = 1e-8
    curve_length_tolerance: float = 0.15
    curve_length_outlier_weight: float = 1.0
    lam_cell_edge_uniform: float = 1.0
    lam_cell_angle_uniform: float = 1.0
    lam_cell_radial_uniform: float = 0.5
    cell_edge_uniform_eps: float = 1e-8
    cell_angle_uniform_eps: float = 1e-8

    comp_normalize_by: float | None = 1e10
    normalize_losses: bool = True
    fem_density_floor: float = 0.02
    skip_bad_fem_steps: bool = True

    num_steps: int = 10000
    tau: float = 0.02
    tau_anneal_final: float | None = None
    tau__anneal_final: float | None = None
    tau_anneal_start_frac: float = 0.0
    tau_anneal_ramp_frac: float = 0.5

    seed_anchor_momentum: float = 0.20
    seed_anchor_warmup_frac: float = 0.05
    use_rolling_seed_anchors: bool = True
    guard_seed_anchor_updates: bool = True
    anchor_guard_rep_max: float = 0.30
    anchor_guard_bnd_max: float = 0.80
    anchor_guard_vol_eff_min: float = 0.10
    anchor_guard_width_factor_min: float = 1.20
    anchor_guard_min_seed_dist_factor: float = 2.0

    collapse_min_seed_dist_factor: float = 2.0
    project_seed_spacing_each_step: bool = True
    seed_projection_iters: int = 4
    allow_seed_outside_domain: bool = True
    allow_seed_outside_domain_warmup_frac: float = 0.50
    seed_domain_margin: float = 0.25
    use_seed_domain_mask: bool = True
    seed_domain_mask_threshold: float = 0.5
    seed_domain_temp: float = 0.05
    seed_domain_mask_support_scale: float = 2.5
    seed_domain_mask_max_points: int = 2048
    use_independent_seed_offsets: bool = True
    independent_seed_offset_max: float = 0.05

    lr_seed_refine: float = 1e-1
    lr_independent_seed_offsets: float = 1e-3
    lr_delta_head: float = 2e-4
    lr_mlp: float = 2e-4
    lr_w_head: float = 2e-4

    log_every: int = 50
    early_stop_start: float = 0.30
    patience: int = 300
    min_delta: float = 1e-4
    prune_inactive_on_plateau: bool = False
    prune_patience: int | None = None

    min_active_seeds: int | None = None

    eps: float = 1e-12

    Offset_scale: float = 1.00
    seed_offset_scale_start: float | None = None
    seed_offset_scale_final: float | None = None
    seed_offset_scale_ramp_frac: float = 0.60
    scheduler_milestones: tuple[float, ...] = (80, 160)
    scheduler_gamma: float = 0.5

    save_fem_debug_history: bool = True
    grad_clip_norm: float | None = 1.0
    debug_anomaly_detection: bool = False

    tensorboard_enabled: bool = True
    tensorboard_log_root: str = "runs"
    experiment_name: str | None = None
    tb_flush_secs: int = 10
    tb_log_histograms_every: int = 200

    MakeTimelaps: bool = True
    timelapse_output_folder: str | None = None

    timelapse_frame_step: int = 20
    TM_laps_Thr: float = 0.45
    timelapse_show_3d_tubes: bool = True
    timelapse_tube_radius_scale: float = 1.0
    timelapse_tube_n_sides: int = 12

    def __post_init__(self):
        self.training_face_index = int(self.training_face_index)
        if self.training_face_index < 0:
            raise ValueError(
                f"training_face_index must be >= 0, got {self.training_face_index}"
            )

        if self.tau__anneal_final is not None:
            self.tau_anneal_final = self.tau__anneal_final

        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.tau_anneal_final is not None and self.tau_anneal_final <= 0.0:
            raise ValueError(f"tau_anneal_final must be > 0, got {self.tau_anneal_final}")
        if not (0.0 <= self.filter_projection_strength <= 1.0):
            raise ValueError(
                "filter_projection_strength must be in [0,1], "
                f"got {self.filter_projection_strength}"
            )
        if self.filter_projection_beta <= 0.0:
            raise ValueError(
                f"filter_projection_beta must be > 0, got {self.filter_projection_beta}"
            )
        if self.centerline_beta <= 0.0:
            raise ValueError(f"centerline_beta must be > 0, got {self.centerline_beta}")
        if self.centerline_softmin_tau <= 0.0:
            raise ValueError(
                f"centerline_softmin_tau must be > 0, got {self.centerline_softmin_tau}"
            )
        if self.tube_curve_samples < 2:
            raise ValueError(f"tube_curve_samples must be >= 2, got {self.tube_curve_samples}")
        if self.tube_lift_tau <= 0.0:
            raise ValueError(f"tube_lift_tau must be > 0, got {self.tube_lift_tau}")
        if self.tube_lift_max_values < 1:
            raise ValueError(
                f"tube_lift_max_values must be >= 1, got {self.tube_lift_max_values}"
            )
        if not (0.0 <= self.rho_min < 1.0):
            raise ValueError(f"rho_min must satisfy 0 <= rho_min < 1, got {self.rho_min}")
        if not (0.0 < self.filter_projection_eta < 1.0):
            raise ValueError(
                "filter_projection_eta must be in (0,1), "
                f"got {self.filter_projection_eta}"
            )
        if not (0.0 <= self.seed_anchor_momentum <= 1.0):
            raise ValueError(
                f"seed_anchor_momentum must be in [0,1], got {self.seed_anchor_momentum}"
            )
        if not (0.0 <= self.seed_anchor_warmup_frac <= 1.0):
            raise ValueError(
                f"seed_anchor_warmup_frac must be in [0,1], got {self.seed_anchor_warmup_frac}"
            )
        if self.anchor_guard_width_factor_min < 1.0:
            raise ValueError(
                "anchor_guard_width_factor_min must be >= 1, "
                f"got {self.anchor_guard_width_factor_min}"
            )
        if self.anchor_guard_min_seed_dist_factor < 0.0:
            raise ValueError(
                "anchor_guard_min_seed_dist_factor must be >= 0, "
                f"got {self.anchor_guard_min_seed_dist_factor}"
            )
        if self.collapse_min_seed_dist_factor < 0.0:
            raise ValueError(
                "collapse_min_seed_dist_factor must be >= 0, "
                f"got {self.collapse_min_seed_dist_factor}"
            )
        if self.seed_projection_iters < 0:
            raise ValueError(f"seed_projection_iters must be >= 0, got {self.seed_projection_iters}")
        if not (0.0 <= self.allow_seed_outside_domain_warmup_frac <= 1.0):
            raise ValueError(
                "allow_seed_outside_domain_warmup_frac must be in [0,1], "
                f"got {self.allow_seed_outside_domain_warmup_frac}"
            )
        if self.seed_domain_margin < 0.0:
            raise ValueError(f"seed_domain_margin must be >= 0, got {self.seed_domain_margin}")
        if not (0.0 <= self.seed_domain_mask_threshold <= 1.0):
            raise ValueError(
                "seed_domain_mask_threshold must be in [0,1], "
                f"got {self.seed_domain_mask_threshold}"
            )
        if self.seed_domain_temp <= 0.0:
            raise ValueError(f"seed_domain_temp must be > 0, got {self.seed_domain_temp}")
        if self.seed_domain_mask_support_scale <= 0.0:
            raise ValueError(
                "seed_domain_mask_support_scale must be > 0, "
                f"got {self.seed_domain_mask_support_scale}"
            )
        if self.seed_domain_mask_max_points < 1:
            raise ValueError(
                "seed_domain_mask_max_points must be >= 1, "
                f"got {self.seed_domain_mask_max_points}"
            )
        if self.independent_seed_offset_max < 0.0:
            raise ValueError(
                "independent_seed_offset_max must be >= 0, "
                f"got {self.independent_seed_offset_max}"
            )
        if self.lr_independent_seed_offsets < 0.0:
            raise ValueError(
                "lr_independent_seed_offsets must be >= 0, "
                f"got {self.lr_independent_seed_offsets}"
            )
        if self.seed_offset_scale_start is not None and self.seed_offset_scale_start <= 0.0:
            raise ValueError(
                f"seed_offset_scale_start must be > 0, got {self.seed_offset_scale_start}"
            )
        if self.seed_offset_scale_final is not None and self.seed_offset_scale_final <= 0.0:
            raise ValueError(
                f"seed_offset_scale_final must be > 0, got {self.seed_offset_scale_final}"
            )
        if not (0.0 < self.seed_offset_scale_ramp_frac <= 1.0):
            raise ValueError(
                "seed_offset_scale_ramp_frac must be in (0,1], "
                f"got {self.seed_offset_scale_ramp_frac}"
            )
        if self.min_active_seeds is not None and self.min_active_seeds < 1:
            raise ValueError(f"min_active_seeds must be >= 1, got {self.min_active_seeds}")
        if self.prune_patience is not None and self.prune_patience < 1:
            raise ValueError(f"prune_patience must be >= 1, got {self.prune_patience}")
        if self.auto_update_wmin:
            if self.min_feature_size_3d is None:
                raise ValueError(
                    "min_feature_size_3d must be set when auto_update_wmin=True"
                )
            if self.min_feature_size_3d <= 0.0:
                raise ValueError(
                    f"min_feature_size_3d must be > 0, got {self.min_feature_size_3d}"
                )
        self.use_balanced_seed_init = bool(self.use_balanced_seed_init)


def _cfg_value(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _centerline_radius_raw_from_w(config, w_raw: torch.Tensor | None):
    return None


def _density_postprocess_debug(
    rho_raw: torch.Tensor,
    rho_filtered: torch.Tensor,
    rho_projected: torch.Tensor,
    rho_final: torch.Tensor,
) -> dict[str, float]:
    filter_delta = (rho_filtered - rho_raw).abs()
    projection_delta = (rho_projected - rho_filtered).abs()
    return {
        "filter_delta_mean": float(filter_delta.detach().mean().item()),
        "filter_delta_max": float(filter_delta.detach().max().item()),
        "projection_delta_mean": float(projection_delta.detach().mean().item()),
        "projection_delta_max": float(projection_delta.detach().max().item()),
        "raw_mean": float(rho_raw.detach().mean().item()),
        "filtered_mean": float(rho_filtered.detach().mean().item()),
        "projected_mean": float(rho_projected.detach().mean().item()),
        "final_mean": float(rho_final.detach().mean().item()),
    }


def _fiber_angles_from_3d(fiber3d: torch.Tensor, eps: float = 1e-6):
    fiber3d = fiber3d / torch.linalg.norm(fiber3d, dim=-1, keepdim=True).clamp_min(eps)
    ax, ay, az = fiber3d.unbind(dim=-1)
    phi = torch.atan2(ay, ax)
    theta = torch.acos(az.clamp(-1.0 + eps, 1.0 - eps))
    return fiber3d, phi, theta


def normalize_decoder_density_fiber_output(out: dict, eps: float = 1e-6) -> dict:
    """
    Accept the decoder's current density/fiber aliases and publish the stable
    training contract: rho, density, fiber3d, phi, theta.
    """
    if "rho" not in out:
        if "density" in out:
            out["rho"] = out["density"]
        elif "rho_surface" in out:
            out["rho"] = out["rho_surface"]
        else:
            raise KeyError("Decoder output must contain one of: rho, density, rho_surface")

    out["density"] = out["rho"]

    if "fiber3d" not in out:
        if "fiber" in out:
            out["fiber3d"] = out["fiber"]
        elif "fiber_direction" in out:
            out["fiber3d"] = out["fiber_direction"]
        elif "3d_fiberDir" in out:
            out["fiber3d"] = out["3d_fiberDir"]
        else:
            raise KeyError("Decoder output must contain one of: fiber3d, fiber, fiber_direction, 3d_fiberDir")

    out["fiber3d"], out["phi"], out["theta"] = _fiber_angles_from_3d(out["fiber3d"], eps=eps)
    out["fiber"] = out["fiber3d"]
    return out


def apply_density_postprocess(
    rho,
    face_tensor,
    config,
    return_debug: bool = False,
):
    """
    Canonical decoder-density postprocess.

    The 3D filter is graph-based, so callers must pass a face_tensor whose
    points/faces correspond to the density samples in `rho`.
    """
    rho_raw = rho
    eps = float(_cfg_value(config, "eps", 1e-8))

    use_density_filter = bool(_cfg_value(config, "use_3d_density_filter", False))
    if use_density_filter:
        rho_filtered = surface_density_filter_metric_aware(
            rho=rho_raw,
            points_xyz=face_tensor["points_xyz"],
            faces=face_tensor["faces_ijk"],
            Xu=face_tensor["Xu"],
            Xv=face_tensor["Xv"],
            base_radius=float(_cfg_value(config, "filter_radius_3d", 0.03)),
            self_weight=float(_cfg_value(config, "filter_self_weight", 1.0)),
            eps=eps,
        )
    else:
        rho_filtered = rho_raw

    projection_strength = float(_cfg_value(config, "filter_projection_strength", 0.0))
    if projection_strength > 0.0:
        rho_projected = smooth_heaviside_projection(
            rho_filtered,
            beta=float(_cfg_value(config, "filter_projection_beta", 8.0)),
            eta=float(_cfg_value(config, "filter_projection_eta", 0.5)),
            strength=projection_strength,
            eps=eps,
            debug=False,
        )
    else:
        rho_projected = rho_filtered

    rho_final = rho_projected
    if not return_debug:
        return rho_final
    return rho_final, _density_postprocess_debug(
        rho_raw=rho_raw,
        rho_filtered=rho_filtered,
        rho_projected=rho_projected,
        rho_final=rho_final,
    )


def apply_density_postprocess_to_output(
    out: dict,
    face_tensor,
    config,
    return_debug: bool = False,
):
    out = normalize_decoder_density_fiber_output(
        out,
        eps=float(_cfg_value(config, "eps", 1e-6)),
    )
    rho_raw = out["rho"]
    rho_final, stats = apply_density_postprocess(
        rho_raw,
        face_tensor,
        config,
        return_debug=True,
    )

    out["rho_raw_decoder"] = rho_raw
    out["rho"] = rho_final
    out["density"] = rho_final
    out["rho_postprocessed"] = rho_final
    out["fiber3d"], out["phi"], out["theta"] = _fiber_angles_from_3d(
        out["fiber3d"],
        eps=float(_cfg_value(config, "eps", 1e-6)),
    )
    out["fiber"] = out["fiber3d"]
    if return_debug:
        out["density_postprocess_stats"] = stats
        return out, stats
    return out


class RunningNorm:
    def __init__(self, momentum: float = 0.99, eps: float = 1e-12):
        self.val = None
        self.momentum = momentum
        self.eps = eps

    def update(self, x: float) -> float:
        x = abs(float(x))
        if not math.isfinite(x):
            return max(self.val if self.val is not None else 1.0, 1e-8)

        x = x + self.eps
        if self.val is None:
            self.val = x
        else:
            self.val = self.momentum * self.val + (1.0 - self.momentum) * x
        return max(self.val, 1e-8)


def _cpu_detached_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_detached_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_detached_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_detached_tree(v) for v in value)
    return value


def _tree_to_device(value, device=None, dtype=None):
    if torch.is_tensor(value):
        out = value.to(device=device) if device is not None else value
        if dtype is not None and out.is_floating_point():
            out = out.to(dtype=dtype)
        return out
    if isinstance(value, dict):
        return {k: _tree_to_device(v, device=device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(v, device=device, dtype=dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(v, device=device, dtype=dtype) for v in value)
    return value


def _import_symbol(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class OptimizedShellFunction:
    """
    Reloadable single-face implicit shell field.

    The object evaluates the optimized decoder field on UV points:
        (u, v), Xu, Xv -> density rho and 3D fiber direction.
    """

    package_version = 1

    def __init__(self, package: dict[str, Any], decoder_cls=None, device=None):
        self.package = package
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if decoder_cls is None:
            decoder_info = package.get("decoder_class", {})
            decoder_cls = _import_symbol(
                decoder_info.get("module", "Decoder_CLasses.VoronoiDecorder"),
                decoder_info.get("name", "VoronoiDecoder"),
            )
        self.decoder_cls = decoder_cls

        self.config = package.get("config", {})
        self.decoder_init_kwargs = _tree_to_device(
            package["decoder_init_kwargs"],
            device=self.device,
        )
        self.decoder = self.decoder_cls(**self.decoder_init_kwargs).to(self.device)
        state = package.get("decoder_state_dict", None)
        if state:
            self.decoder.load_state_dict(_tree_to_device(state, device=self.device))
        self.decoder.eval()

        self.best_pred = _tree_to_device(package["best_pred"], device=self.device)
        self.face_metadata = package.get("face_metadata", {})

    @classmethod
    def load(cls, path, decoder_cls=None, device=None):
        try:
            package = torch.load(path, map_location=device or "cpu", weights_only=False)
        except TypeError:
            package = torch.load(path, map_location=device or "cpu")
        return cls(package=package, decoder_cls=decoder_cls, device=device)

    @staticmethod
    def _true_open_boundary_idx(ft, tol=None):
        if ("boundary_idx_ring1" not in ft) or ft["boundary_idx_ring1"] is None:
            return torch.empty(0, dtype=torch.long, device=ft["uv"].device)

        bidx = torch.unique(ft["boundary_idx_ring1"].to(dtype=torch.long))
        if bidx.numel() == 0:
            return bidx

        uv = ft["uv"]
        u = uv[:, 0]
        v = uv[:, 1]
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        if tol is None:
            u_span = (u.max() - u.min()).abs()
            v_span = (v.max() - v.min()).abs()
            base_span = torch.maximum(
                u_span,
                v_span,
            ).clamp_min(torch.as_tensor(1.0, device=uv.device, dtype=uv.dtype))
            tol = 1e-4 * float(base_span.detach().item())

        ub = u[bidx]
        vb = v[bidx]
        keep = torch.ones_like(bidx, dtype=torch.bool)

        if u_periodic:
            umin = u.min()
            umax = u.max()
            is_u_seam = (ub - umin).abs() <= tol
            is_u_seam = is_u_seam | ((ub - umax).abs() <= tol)
            keep = keep & (~is_u_seam)

        if v_periodic:
            vmin = v.min()
            vmax = v.max()
            is_v_seam = (vb - vmin).abs() <= tol
            is_v_seam = is_v_seam | ((vb - vmax).abs() <= tol)
            keep = keep & (~is_v_seam)

        return bidx[keep]

    def _seed_domain_mask_for_face(self, ft):
        if not bool(self.config.get("use_seed_domain_mask", False)):
            return None

        mask_grid = ft.get("seed_domain_mask_grid", None)
        if mask_grid is not None:
            return mask_grid

        uv_face = ft.get("seed_domain_uv_support", ft["uv"])
        if uv_face.numel() == 0:
            return None

        cfg = self.config
        uv_support = uv_face.detach()
        max_points = int(cfg.get("seed_domain_mask_max_points", 2048))
        if uv_support.shape[0] > max_points:
            sample_idx = torch.linspace(
                0,
                uv_support.shape[0] - 1,
                max_points,
                device=uv_support.device,
            ).round().to(torch.long)
            uv_support = uv_support[sample_idx]

        sigma_value = ft.get("seed_domain_sigma", None)
        if sigma_value is None:
            sigma = NN_Trainer._estimate_uv_mask_tol(
                uv_support,
                u_periodic=bool(ft.get("u_periodic", False)),
                v_periodic=bool(ft.get("v_periodic", False)),
                fallback=float(cfg.get("boundary_margin", 0.05)),
                scale=float(cfg.get("seed_domain_mask_support_scale", 2.5)),
            )
        elif torch.is_tensor(sigma_value):
            sigma = float(sigma_value.detach().cpu().item())
        else:
            sigma = float(sigma_value)
        sigma = max(float(sigma), float(cfg.get("eps", 1e-12)))
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        def mask_fn(seeds):
            support = uv_support.to(device=seeds.device, dtype=seeds.dtype)
            diff = seeds.unsqueeze(1) - support.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            dmin = torch.norm(diff, dim=-1).amin(dim=1)
            sigma_t = torch.as_tensor(sigma, device=seeds.device, dtype=seeds.dtype)
            return torch.exp(-0.5 * (dmin / sigma_t.clamp_min(float(cfg.get("eps", 1e-12)))).pow(2))

        return mask_fn

    def evaluate_at_uv(
        self,
        points_uv,
        Xu,
        Xv,
        points_xyz=None,
        face_tensor=None,
        boundary_uv=None,
        hard_seed_mask=True,
    ):
        points_uv = torch.as_tensor(points_uv, device=self.device)
        dtype = points_uv.dtype if points_uv.is_floating_point() else torch.float32
        points_uv = points_uv.to(dtype=dtype)
        Xu = torch.as_tensor(Xu, device=self.device, dtype=dtype)
        Xv = torch.as_tensor(Xv, device=self.device, dtype=dtype)
        points_xyz = (
            None
            if points_xyz is None
            else torch.as_tensor(points_xyz, device=self.device, dtype=dtype)
        )

        ft = None
        if face_tensor is not None:
            ft = _tree_to_device(dict(face_tensor), device=self.device, dtype=dtype)

        points_face_id = torch.zeros(points_uv.shape[0], dtype=torch.long, device=self.device)
        boundary_face_id = None
        if boundary_uv is None and ft is not None:
            bidx = self._true_open_boundary_idx(ft)
            if bidx.numel() > 0:
                boundary_uv = ft["uv"][bidx]
                boundary_face_id = torch.zeros(
                    boundary_uv.shape[0],
                    dtype=torch.long,
                    device=self.device,
                )
        elif boundary_uv is not None:
            boundary_uv = torch.as_tensor(boundary_uv, device=self.device, dtype=dtype)
            boundary_face_id = torch.zeros(
                boundary_uv.shape[0],
                dtype=torch.long,
                device=self.device,
            )

        seed_domain_mask = self._seed_domain_mask_for_face(ft) if ft is not None else None
        pred = _tree_to_device(self.best_pred, device=self.device, dtype=dtype)
        tau = pred.get("tau", None)
        if tau is None:
            tau = float(self.config.get("tau", 0.02))

        with torch.no_grad():
            # Raw arbitrary-point evaluation: graph density postprocess needs a
            # full mesh/face tensor and is applied by evaluate_face().
            use_u_periodic = self.decoder._bool_value(getattr(self.decoder, "face_u_periodic", False))
            use_v_periodic = self.decoder._bool_value(getattr(self.decoder, "face_v_periodic", False))
            return self.decoder.build_swept_tube_fields(
                points_uv=points_uv,
                points_3d=points_xyz,
                seeds_uv=pred["seeds_raw"],
                w_raw=pred["w_raw"],
                Xu=Xu,
                Xv=Xv,
                cad_domain=getattr(self.decoder, "Cad_domain", None),
                u_periodic=use_u_periodic,
                v_periodic=use_v_periodic,
                return_xyz=True,
                generate_density_fiber=bool(self.config.get("generate_decoder_density_fiber", True)),
            )

    def evaluate_face(self, face_tensor, hard_seed_mask=True):
        ft = _tree_to_device(
            dict(face_tensor),
            device=self.device,
            dtype=face_tensor["uv"].dtype if torch.is_tensor(face_tensor["uv"]) else None,
        )
        out = self.evaluate_at_uv(
            points_uv=face_tensor["uv"],
            Xu=face_tensor["Xu"],
            Xv=face_tensor["Xv"],
            points_xyz=face_tensor["points_xyz"],
            face_tensor=ft,
            hard_seed_mask=hard_seed_mask,
        )
        return apply_density_postprocess_to_output(
            out,
            ft,
            self.config,
            return_debug=False,
        )

    def build_fem_fields(self, shell_problem, face_tensor, rho_void=1e-3, hard_seed_mask=True):
        out = self.evaluate_face(face_tensor, hard_seed_mask=hard_seed_mask)
        return shell_problem.build_fem_fields_from_decoder_torch(
            rho_surface=out["rho"],
            fiber_surface=out["fiber3d"],
            rho_void=rho_void,
        )


def evaluate_optimized_shell_function(
    optimized_function,
    face_tensors,
    face_index: int = 0,
    hard_seed_mask: bool = True,
):
    """
    Evaluate a loaded optimized single-face shell function on a face tensor.

    Returns surface density and 3D fiber direction, ready for later
    visualization or export to a custom FEM workflow.
    """
    if isinstance(face_tensors, dict) and "face_tensors" in face_tensors:
        face_tensors = face_tensors["face_tensors"]

    if isinstance(face_tensors, (list, tuple)):
        face_tensor = face_tensors[int(face_index)]
    else:
        face_tensor = face_tensors

    out = optimized_function.evaluate_face(
        face_tensor,
        hard_seed_mask=hard_seed_mask,
    )
    density = out["rho"]
    fiber_2d = out.get("t_uv", out.get("t_uv_raw", None))
    fiber_3d = out["fiber3d"]
    rho_raw_decoder = out.get("rho_raw_decoder", density)
    density_binary = (density >= 0.5).to(dtype=density.dtype)
    return {
        "2d_density": density,
        "2d_fiberDir": fiber_2d,
        "3d_density": density,
        "3d_fiberDir": fiber_3d,
        "density": density,
        "density_binary": density_binary,
        "fiber_direction": fiber_3d,
        "rho": density,
        "rho_raw_decoder": rho_raw_decoder,
        "rho_postprocessed": out.get("rho_postprocessed", density),
        "fiber3d": fiber_3d,
        "t_uv": fiber_2d,
        "decoder_output": out,
        "face_tensor": face_tensor,
    }


def sanity_check_density_postprocess_pipeline(
    optimized_function,
    face_tensor,
    expected_training_rho=None,
    tol: float = 1e-6,
    small_tolerance: float = 1e-8,
):
    out = optimized_function.evaluate_face(face_tensor)
    rho = out["rho"]
    rho_raw = out.get("rho_raw_decoder", rho)
    cfg = optimized_function.config
    postprocess_enabled = bool(cfg.get("use_3d_density_filter", False))
    if postprocess_enabled:
        delta = (rho - rho_raw).abs().mean()
        assert float(delta.detach().item()) > float(small_tolerance), (
            "Postprocess is enabled but evaluate_face returned density too close "
            "to rho_raw_decoder."
        )
    fields = evaluate_optimized_shell_function(optimized_function, face_tensor)
    assert fields["rho"] is out["rho"] or torch.allclose(fields["rho"], out["rho"], atol=tol, rtol=0.0)
    assert fields["density"] is fields["rho"] or torch.allclose(fields["density"], fields["rho"], atol=tol, rtol=0.0)
    assert "rho_raw_decoder" in fields
    if expected_training_rho is not None:
        expected = torch.as_tensor(expected_training_rho, device=rho.device, dtype=rho.dtype)
        assert torch.allclose(rho, expected, atol=tol, rtol=0.0), (
            "Loaded optimized_shell_function.evaluate_face(...) does not match "
            "the expected training-time postprocessed density."
        )
    return {
        "mean_abs_postprocess_delta": float((rho - rho_raw).abs().mean().detach().item()),
        "rho_mean": float(rho.detach().mean().item()),
        "rho_raw_mean": float(rho_raw.detach().mean().item()),
    }


def load_optimized_shell_function(path, decoder_cls=None, device=None):
    return OptimizedShellFunction.load(path, decoder_cls=decoder_cls, device=device)


def _field_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _surface_density_volume_fraction(face_tensor, density_values):
    density = np.asarray(density_values, dtype=np.float64).reshape(-1)
    faces = _field_to_numpy(face_tensor.get("faces_ijk", np.empty((0, 3)))).astype(np.int64)

    if faces.size == 0:
        valid = np.isfinite(density)
        value = float(np.mean(density[valid])) if np.any(valid) else float("nan")
        return value, "point-mean"

    face_areas_raw = face_tensor.get("face_areas", None)
    if face_areas_raw is not None:
        face_areas = _field_to_numpy(face_areas_raw).reshape(-1).astype(np.float64)
    else:
        xyz = _field_to_numpy(face_tensor["points_xyz"]).astype(np.float64)
        tri = xyz[faces]
        face_areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )

    if face_areas.shape[0] != faces.shape[0]:
        valid = np.isfinite(density)
        value = float(np.mean(density[valid])) if np.any(valid) else float("nan")
        return value, "point-mean"

    weights = np.zeros((density.shape[0],), dtype=np.float64)
    local_weight = face_areas / 3.0
    np.add.at(weights, faces[:, 0], local_weight)
    np.add.at(weights, faces[:, 1], local_weight)
    np.add.at(weights, faces[:, 2], local_weight)

    valid = np.isfinite(density) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan"), "area-weighted"
    return float(np.sum(density[valid] * weights[valid]) / np.sum(weights[valid])), "area-weighted"


def visualize_optimized_shell_fields(
    fields,
    show_2d: bool = True,
    show_3d: bool = True,
    density_cmap: str = "viridis",
    fiber_stride: int = 20,
    fiber_min_density: float = 0.05,
    fiber_scale_2d: float = 0.06,
    fiber_scale_3d: float | None = None,
    fiber_vector_style: str = "arrow",
    fiber_color: str = "#1f4fa3",
    show_fiber_surface: bool = True,
    fiber_surface_opacity: float = 0.25,
    show_fiber_background: bool = False,
    show_edges: bool = False,
    window_size: tuple[int, int] = (1500, 700),
):
    """
    Visualize loaded optimized shell fields in UV and on the 3D surface.

    Returns a dictionary with optional:
        uv_fig: matplotlib figure for 2D UV density/fiber
        plotter: pyvista plotter for 3D density/fiber
    """
    face_tensor = fields["face_tensor"]
    uv = _field_to_numpy(face_tensor["uv"]).astype(np.float64)
    xyz = _field_to_numpy(face_tensor["points_xyz"]).astype(np.float64)
    faces = _field_to_numpy(face_tensor["faces_ijk"]).astype(np.int64)

    density_2d = _field_to_numpy(fields["2d_density"]).reshape(-1).astype(np.float64)
    fiber_2d = _field_to_numpy(fields["2d_fiberDir"]).reshape(-1, 2).astype(np.float64)
    density_3d = _field_to_numpy(fields["3d_density"]).reshape(-1).astype(np.float64)
    fiber_3d = _field_to_numpy(fields["3d_fiberDir"]).reshape(-1, 3).astype(np.float64)

    result = {}

    if show_2d:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        ax_density, ax_fiber = axes

        if faces.size > 0:
            density_artist = ax_density.tripcolor(
                uv[:, 0],
                uv[:, 1],
                faces,
                density_2d,
                shading="gouraud",
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
            )
        else:
            density_artist = ax_density.scatter(
                uv[:, 0],
                uv[:, 1],
                c=density_2d,
                s=10,
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
                linewidths=0,
            )
        ax_density.set_title("2D UV Density")
        ax_density.set_xlabel("u")
        ax_density.set_ylabel("v")
        ax_density.set_aspect("equal", adjustable="box")
        fig.colorbar(density_artist, ax=ax_density, label="density")

        if show_fiber_background and faces.size > 0:
            ax_fiber.tripcolor(
                uv[:, 0],
                uv[:, 1],
                faces,
                density_2d,
                shading="gouraud",
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.30,
            )
        elif show_fiber_background:
            ax_fiber.scatter(
                uv[:, 0],
                uv[:, 1],
                c=density_2d,
                s=10,
                cmap=density_cmap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.30,
                linewidths=0,
            )

        fiber_norm_2d = np.linalg.norm(fiber_2d, axis=1)
        mask_2d = np.isfinite(density_2d) & np.isfinite(fiber_2d).all(axis=1)
        mask_2d &= density_2d >= float(fiber_min_density)
        mask_2d &= fiber_norm_2d > 1e-12
        if fiber_stride > 1:
            stride_mask = np.zeros(mask_2d.shape[0], dtype=bool)
            stride_mask[::int(fiber_stride)] = True
            mask_2d &= stride_mask

        if np.any(mask_2d):
            ax_fiber.quiver(
                uv[mask_2d, 0],
                uv[mask_2d, 1],
                fiber_2d[mask_2d, 0],
                fiber_2d[mask_2d, 1],
                density_2d[mask_2d],
                cmap=density_cmap,
                angles="xy",
                scale_units="xy",
                scale=max(float(fiber_scale_2d), 1e-8) ** -1,
                width=0.003,
                pivot="mid",
            )
        ax_fiber.set_title("2D UV Fiber Direction")
        ax_fiber.set_xlabel("u")
        ax_fiber.set_ylabel("v")
        ax_fiber.set_aspect("equal", adjustable="box")
        result["uv_fig"] = fig

    if show_3d:
        volume_fraction, volume_fraction_method = _surface_density_volume_fraction(
            face_tensor,
            density_3d,
        )
        result["density_volume_fraction"] = volume_fraction
        result["density_volume_fraction_method"] = volume_fraction_method
        print(
            "3D density volume fraction "
            f"({volume_fraction_method}): {volume_fraction:.6f}"
        )

        if faces.size > 0:
            pv_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
            pv_faces[:, 0] = 3
            pv_faces[:, 1:] = faces
            mesh = pv.PolyData(xyz, pv_faces.reshape(-1))
        else:
            mesh = pv.PolyData(xyz)
        mesh["density"] = density_3d.astype(np.float32)

        plotter = pv.Plotter(shape=(1, 2), window_size=window_size)

        plotter.subplot(0, 0)
        plotter.add_text("3D Surface Density", font_size=10)
        plotter.add_mesh(
            mesh,
            scalars="density",
            cmap=density_cmap,
            clim=[0.0, 1.0],
            show_edges=show_edges,
        )
        plotter.show_axes()

        plotter.subplot(0, 1)
        plotter.add_text("3D Surface Fiber Direction", font_size=10)
        if show_fiber_background:
            plotter.add_mesh(
                mesh.copy(),
                scalars="density",
                cmap=density_cmap,
                clim=[0.0, 1.0],
                opacity=0.30,
                show_edges=show_edges,
            )
        elif show_fiber_surface:
            plotter.add_mesh(
                mesh.copy(),
                color="white",
                opacity=float(fiber_surface_opacity),
                show_edges=False,
                smooth_shading=True,
            )

        fiber_norm_3d = np.linalg.norm(fiber_3d, axis=1)
        mask_3d = np.isfinite(density_3d) & np.isfinite(fiber_3d).all(axis=1)
        mask_3d &= density_3d >= float(fiber_min_density)
        mask_3d &= fiber_norm_3d > 1e-12
        if fiber_stride > 1:
            stride_mask = np.zeros(mask_3d.shape[0], dtype=bool)
            stride_mask[::int(fiber_stride)] = True
            mask_3d &= stride_mask

        if np.any(mask_3d):
            diag = float(np.linalg.norm(np.ptp(xyz, axis=0)))
            glyph_scale = 0.04 * max(diag, 1e-6) if fiber_scale_3d is None else float(fiber_scale_3d)
            cloud = pv.PolyData(xyz[mask_3d])
            cloud["vectors"] = fiber_3d[mask_3d].astype(np.float32)
            cloud["density"] = density_3d[mask_3d].astype(np.float32)
            style = str(fiber_vector_style).lower()
            if style == "arrow":
                glyph_geom = pv.Arrow(
                    start=(0.0, 0.0, 0.0),
                    direction=(1.0, 0.0, 0.0),
                    tip_length=0.30,
                    tip_radius=0.045,
                    shaft_radius=0.014,
                    shaft_resolution=8,
                    tip_resolution=12,
                )
            elif style == "line":
                glyph_geom = pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))
            else:
                raise ValueError("fiber_vector_style must be 'arrow' or 'line'")
            glyphs = cloud.glyph(
                orient="vectors",
                scale=False,
                factor=glyph_scale,
                geom=glyph_geom,
            )
            plotter.add_mesh(glyphs, color=fiber_color, line_width=2)

        plotter.show_axes()
        plotter.link_views()
        result["plotter"] = plotter

    return result


def visualize_optimized_shell_fields_2d(fields, **kwargs):
    return visualize_optimized_shell_fields(
        fields,
        show_2d=True,
        show_3d=False,
        **kwargs,
    )["uv_fig"]


def visualize_optimized_shell_fields_3d(fields, **kwargs):
    result = visualize_optimized_shell_fields(
        fields,
        show_2d=False,
        show_3d=True,
        **kwargs,
    )
    return result["plotter"], result["density_volume_fraction"]


def binarize_optimized_shell_fields(
    fields,
    density_threshold: float = 0.5,
    solid_density: float = 1.0,
    void_density: float = 1e-3,
    mask_void_fibers: bool = True,
):
    """
    Convert optimized continuous surface density to solid/void density.

    Fiber directions are directions, so they are not thresholded into binary
    values. They are normalized and optionally zeroed in void regions.
    """
    out = dict(fields)
    density = fields["3d_density"]
    fiber_2d = fields.get("2d_fiberDir", None)
    fiber_3d = fields["3d_fiberDir"]

    solid_mask = density >= float(density_threshold)
    binary_density = torch.where(
        solid_mask,
        torch.as_tensor(solid_density, dtype=density.dtype, device=density.device),
        torch.as_tensor(void_density, dtype=density.dtype, device=density.device),
    )

    def normalize_and_mask(fiber):
        if fiber is None:
            return None
        norm = torch.linalg.norm(fiber, dim=1, keepdim=True).clamp_min(1e-12)
        fiber_out = fiber / norm
        if mask_void_fibers:
            fiber_out = torch.where(solid_mask[:, None], fiber_out, torch.zeros_like(fiber_out))
        return fiber_out

    binary_fiber_2d = normalize_and_mask(fiber_2d)
    binary_fiber_3d = normalize_and_mask(fiber_3d)

    out["2d_density_continuous"] = fields["2d_density"]
    out["3d_density_continuous"] = fields["3d_density"]
    out["2d_fiberDir_continuous"] = fields.get("2d_fiberDir", None)
    out["3d_fiberDir_continuous"] = fields["3d_fiberDir"]

    out["solid_mask"] = solid_mask
    out["2d_density"] = binary_density
    out["3d_density"] = binary_density
    out["density"] = binary_density
    out["rho"] = binary_density

    if binary_fiber_2d is not None:
        out["2d_fiberDir"] = binary_fiber_2d
        out["t_uv"] = binary_fiber_2d
    out["3d_fiberDir"] = binary_fiber_3d
    out["fiber_direction"] = binary_fiber_3d
    out["fiber3d"] = binary_fiber_3d

    return out


class Load_Model:
    @staticmethod
    def load(path, decoder_cls=None, device=None):
        return load_optimized_shell_function(
            path=path,
            decoder_cls=decoder_cls,
            device=device,
        )

    @staticmethod
    def evaluate(
        optimized_function,
        face_tensors,
        face_index: int = 0,
        hard_seed_mask: bool = True,
    ):
        return evaluate_optimized_shell_function(
            optimized_function=optimized_function,
            face_tensors=face_tensors,
            face_index=face_index,
            hard_seed_mask=hard_seed_mask,
        )

    @staticmethod
    def visualize(
        fields,
        show_2d: bool = True,
        show_3d: bool = True,
        **kwargs,
    ):
        return visualize_optimized_shell_fields(
            fields,
            show_2d=show_2d,
            show_3d=show_3d,
            **kwargs,
        )

    @staticmethod
    def visualize_2d(fields, **kwargs):
        return visualize_optimized_shell_fields_2d(fields, **kwargs)

    @staticmethod
    def visualize_3d(fields, **kwargs):
        return visualize_optimized_shell_fields_3d(fields, **kwargs)

    @staticmethod
    def binarize(
        fields,
        density_threshold: float = 0.5,
        solid_density: float = 1.0,
        void_density: float = 1e-3,
        mask_void_fibers: bool = True,
    ):
        return binarize_optimized_shell_fields(
            fields,
            density_threshold=density_threshold,
            solid_density=solid_density,
            void_density=void_density,
            mask_void_fibers=mask_void_fibers,
        )

class NN_Trainer:
    def __init__(
        self,
        generator,
        viz,
        decoder_cls,
        ppnet_cls,
        fem,
        shell_problem,
        config: TrainingConfig,
        loading_img=None,
        Cad_domain=None,
        cad_domain=None,
        face_mesh=None,
    ):
        self.generator = generator
        self.viz = viz
        self.decoder_cls = decoder_cls
        self.ppnet_cls = ppnet_cls
        self.fem = fem
        self.shell_problem = shell_problem
        self.cfg = config
        self.Cad_domain = Cad_domain if Cad_domain is not None else cad_domain
        if self.Cad_domain is None:
            self.Cad_domain = generator
        self.face_mesh = face_mesh

        self.last_fem_debug = {}
        self.fem_debug_history = []
        self.loss_volume = Loss_Volume()
        self.loss_fem = Loss_FEM(self)
        self.loss_boundary = Loss_Boundary()
        self.loss_rep = Loss_rep()
        self.timelapse_loading_img = (
            None if loading_img is None else self._composite_to_white(np.asarray(loading_img))
        )

        self.writer = None
        self.tensorboard_log_dir = None
        self._init_tensorboard()

    def curve_3d_edge_lengths(self, decoder_out):
        curves = decoder_out.get("edge_curves_xyz", None)
        if curves is None:
            for value in decoder_out.values():
                if torch.is_tensor(value):
                    return value.new_empty((0,))
            try:
                device = next(self.ppnet_cls.parameters()).device
            except Exception:
                device = getattr(self, "device", torch.device("cpu"))
            return torch.empty((0,), device=device)

        if curves.ndim != 3 or curves.shape[0] == 0 or curves.shape[1] < 2:
            return curves.new_empty((0,))

        seg = curves[:, 1:, :] - curves[:, :-1, :]
        edge_len = torch.linalg.norm(seg, dim=-1).sum(dim=-1)
        return edge_len[torch.isfinite(edge_len)]

    def curve_length_similarity_loss(self, decoder_out):
        """
        Differentiable loss that encourages all generated 3D edge curves
        to have similar lengths.

        Uses decoder_out["edge_curves_xyz"] with shape [E, K, 3].
        """
        cfg = self.cfg
        edge_len = self.curve_3d_edge_lengths(decoder_out)
        if edge_len.numel() <= 1:
            return edge_len.new_zeros(())

        eps = float(getattr(cfg, "curve_length_eps", 1e-8))
        mean_len = edge_len.mean().clamp_min(eps)
        loss_type = str(getattr(cfg, "curve_length_loss_type", "cv")).lower()

        if loss_type == "target" and getattr(cfg, "curve_length_target", None) is not None:
            target = torch.as_tensor(
                float(cfg.curve_length_target),
                dtype=edge_len.dtype,
                device=edge_len.device,
            )
            return ((edge_len - target) / target.clamp_min(eps)).pow(2).mean()

        if loss_type == "var":
            return ((edge_len - mean_len) / mean_len).pow(2).mean()
        if loss_type == "pairwise":
            diff = edge_len[:, None] - edge_len[None, :]
            denom = mean_len.pow(2).clamp_min(eps)
            return diff.pow(2).mean() / denom
        
        if loss_type == "pairwise_smooth_l1":
            diff = edge_len[:, None] - edge_len[None, :]
            return torch.sqrt(diff.pow(2) + eps).mean() / mean_len
        
        if loss_type == "range_cv":
            cv = edge_len.var(unbiased=False) / mean_len.pow(2).clamp_min(eps)
            rel_range = (edge_len.max() - edge_len.min()) / mean_len
            return cv + 5.0*rel_range.pow(2)
        
        if loss_type == "log_tolerance":
            ratio = edge_len / mean_len
            log_ratio = torch.log(ratio.clamp_min(eps))
            tol = float(getattr(cfg, "curve_length_tolerance", 0.15))
            excess = torch.relu(log_ratio.abs() - tol)
            return excess.pow(2).mean()

        if loss_type == "log_tolerance_max":
            ratio = edge_len / mean_len
            log_ratio = torch.log(ratio.clamp_min(eps))
            tol = float(getattr(cfg, "curve_length_tolerance", 0.15))
            excess = torch.relu(log_ratio.abs() - tol)
            outlier_weight = float(getattr(cfg, "curve_length_outlier_weight", 1.0))
            return excess.pow(2).mean() + outlier_weight * excess.max().pow(2)

        return edge_len.var(unbiased=False) / mean_len.pow(2)
    

    
    

    def cell_edge_uniformity_loss(self, decoder_out):
        """
        Penalize per-cell irregularity without forcing all cells to one size.

        Uses decoder_out["edge_curves_xyz"] with shape [E, K, 3] and
        decoder_out["graph"]["edge_seed_pair"] with shape [E, 2]. Each graph
        edge contributes to both cells in its seed pair.
        """
        cfg = self.cfg
        curves = decoder_out.get("edge_curves_xyz", None)
        graph = decoder_out.get("graph", None)
        if curves is None or not isinstance(graph, dict):
            for value in decoder_out.values():
                if torch.is_tensor(value):
                    return value.new_zeros(())
            try:
                device = next(self.ppnet_cls.parameters()).device
            except Exception:
                device = getattr(self, "device", torch.device("cpu"))
            return torch.zeros((), device=device)

        pairs = graph.get("edge_seed_pair", None)
        if pairs is None:
            return curves.new_zeros(())
        if curves.ndim != 3 or curves.shape[0] == 0 or curves.shape[1] < 2:
            return curves.new_zeros(())
        if pairs.ndim != 2 or pairs.shape != (curves.shape[0], 2):
            return curves.new_zeros(())

        seg = curves[:, 1:, :] - curves[:, :-1, :]
        edge_len = torch.linalg.norm(seg, dim=-1).sum(dim=-1)
        pairs = pairs.to(device=edge_len.device)
        finite_edges = torch.isfinite(edge_len)
        valid_pairs = pairs >= 0
        valid_edge = finite_edges & valid_pairs.any(dim=1)
        if not bool(valid_edge.any().detach().cpu().item()):
            return curves.new_zeros(())

        edge_alpha = graph.get("edge_alpha", None)
        if torch.is_tensor(edge_alpha) and edge_alpha.shape == edge_len.shape:
            edge_weight = edge_alpha.to(dtype=edge_len.dtype, device=edge_len.device).clamp_min(0.0)
        else:
            edge_weight = torch.ones_like(edge_len)

        eps = float(getattr(cfg, "cell_edge_uniform_eps", 1e-8))
        angle_eps = float(getattr(cfg, "cell_angle_uniform_eps", 1e-8))
        lam_angle = float(getattr(cfg, "lam_cell_angle_uniform", 1.0))
        lam_radial = float(getattr(cfg, "lam_cell_radial_uniform", 0.5))
        edge_start = curves[:, 0, :]
        edge_end = curves[:, -1, :]
        cell_ids = torch.unique(pairs[valid_edge & valid_pairs.any(dim=1)])
        cell_ids = cell_ids[cell_ids >= 0]
        losses = []

        def unique_endpoint_indices(points: torch.Tensor) -> torch.Tensor:
            rounded = torch.round(points.detach().cpu() / max(eps, 1e-12)).to(torch.long)
            seen = {}
            keep = []
            for idx, key_t in enumerate(rounded):
                key = tuple(int(v) for v in key_t.tolist())
                if key not in seen:
                    seen[key] = idx
                    keep.append(idx)
            return torch.as_tensor(keep, dtype=torch.long, device=points.device)

        def polygon_vertices_for_cell(edge_ids: torch.Tensor) -> torch.Tensor | None:
            endpoints = torch.cat((edge_start[edge_ids], edge_end[edge_ids]), dim=0)
            finite_points = torch.isfinite(endpoints).all(dim=1)
            endpoints = endpoints[finite_points]
            if endpoints.shape[0] < 3:
                return None

            keep = unique_endpoint_indices(endpoints)
            vertices = endpoints.index_select(0, keep)
            if vertices.shape[0] < 3:
                return None
            return vertices

        def ordered_vertices_for_cell(vertices: torch.Tensor) -> torch.Tensor | None:
            if vertices is None or vertices.shape[0] < 3:
                return None

            center_det = vertices.detach().mean(dim=0)
            rel_det = vertices.detach() - center_det
            if not bool((torch.linalg.vector_norm(rel_det, dim=1) > eps).all().detach().cpu().item()):
                return None

            try:
                _u, _s, vh = torch.linalg.svd(rel_det, full_matrices=False)
                basis_x = vh[0]
                basis_y = vh[1] if vh.shape[0] > 1 else rel_det.new_tensor([0.0, 1.0, 0.0])
            except Exception:
                basis_x = rel_det[0] / torch.linalg.vector_norm(rel_det[0]).clamp_min(eps)
                trial = rel_det[1]
                basis_y = trial - (trial * basis_x).sum() * basis_x
                basis_y = basis_y / torch.linalg.vector_norm(basis_y).clamp_min(eps)

            x = rel_det @ basis_x
            y = rel_det @ basis_y
            if not bool((torch.isfinite(x).all() & torch.isfinite(y).all()).detach().cpu().item()):
                return None
            order = torch.argsort(torch.atan2(y, x))
            return vertices.index_select(0, order.to(device=vertices.device))

        def polygon_angle_loss(vertices: torch.Tensor) -> torch.Tensor | None:
            if vertices is None or vertices.shape[0] < 3:
                return None
            prev_v = torch.roll(vertices, shifts=1, dims=0)
            next_v = torch.roll(vertices, shifts=-1, dims=0)
            a = prev_v - vertices
            b = next_v - vertices
            a_norm = torch.linalg.vector_norm(a, dim=1)
            b_norm = torch.linalg.vector_norm(b, dim=1)
            valid = (a_norm > angle_eps) & (b_norm > angle_eps)
            if int(valid.detach().sum().cpu().item()) < 3:
                return None
            cos_angle = (a[valid] * b[valid]).sum(dim=1) / (
                a_norm[valid] * b_norm[valid]
            ).clamp_min(angle_eps)
            angles = torch.acos(cos_angle.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            mean_angle = angles.mean().clamp_min(angle_eps)
            return angles.var(unbiased=False) / mean_angle.pow(2)

        def polygon_radial_loss(vertices: torch.Tensor) -> torch.Tensor | None:
            if vertices is None or vertices.shape[0] < 3:
                return None
            center = vertices.mean(dim=0)
            radial = torch.linalg.vector_norm(vertices - center, dim=1)
            finite_radial = torch.isfinite(radial) & (radial > eps)
            if int(finite_radial.detach().sum().cpu().item()) < 3:
                return None
            radial = radial[finite_radial]
            mean_radial = radial.mean().clamp_min(eps)
            return radial.var(unbiased=False) / mean_radial.pow(2)

        for cell_id in cell_ids.detach().cpu().tolist():
            belongs = valid_edge & (pairs == int(cell_id)).any(dim=1)
            if int(belongs.sum().detach().cpu().item()) <= 1:
                continue
            edge_ids = torch.nonzero(belongs, as_tuple=False).flatten()

            lengths = edge_len[belongs]
            weights = edge_weight[belongs]
            finite = torch.isfinite(lengths) & torch.isfinite(weights) & (weights > 0.0)
            if int(finite.sum().detach().cpu().item()) <= 1:
                continue

            lengths = lengths[finite]
            weights = weights[finite]
            weight_sum = weights.sum().clamp_min(eps)
            mean_len = (weights * lengths).sum() / weight_sum
            mean_len = mean_len.clamp_min(eps)
            var_len = (weights * (lengths - mean_len).pow(2)).sum() / weight_sum
            cell_loss = var_len / mean_len.pow(2)

            cell_vertices = polygon_vertices_for_cell(edge_ids)
            ordered_vertices = ordered_vertices_for_cell(cell_vertices)
            angle_loss = polygon_angle_loss(ordered_vertices)
            if angle_loss is not None:
                cell_loss = cell_loss + lam_angle * angle_loss
            elif lam_radial != 0.0:
                radial_loss = polygon_radial_loss(cell_vertices)
                if radial_loss is not None:
                    cell_loss = cell_loss + lam_radial * radial_loss

            losses.append(cell_loss)

        if not losses:
            return curves.new_zeros(())
        return torch.stack(losses).mean()

    def neutral_density_fiber_fields(self, uv: torch.Tensor, Xu: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        rho = torch.zeros((uv.shape[0],), dtype=uv.dtype, device=uv.device)
        if Xu is not None:
            fiber = Xu.to(device=uv.device, dtype=uv.dtype)
            if fiber.ndim != 2 or fiber.shape != (uv.shape[0], 3):
                fiber = None
        else:
            fiber = None
        if fiber is None:
            fiber = uv.new_tensor([1.0, 0.0, 0.0]).expand(uv.shape[0], 3)
        fiber = fiber / torch.linalg.norm(fiber, dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        stats = {
            "filter_delta_mean": 0.0,
            "filter_delta_max": 0.0,
            "projection_delta_mean": 0.0,
            "projection_delta_max": 0.0,
            "raw_mean": 0.0,
            "filtered_mean": 0.0,
            "projected_mean": 0.0,
            "final_mean": 0.0,
        }
        return rho, fiber, stats

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------

    def _init_tensorboard(self):
        if not self.cfg.tensorboard_enabled:
            return

        exp_name = self.cfg.experiment_name
        if exp_name is None or str(exp_name).strip() == "":
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(self.cfg.tensorboard_log_root, exp_name)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=self.cfg.tb_flush_secs,
        )
        self.tensorboard_log_dir = log_dir

        cfg_lines = [f"{k}: {v}" for k, v in vars(self.cfg).items()]
        self.writer.add_text("config", "\n".join(cfg_lines), global_step=0)

        print(f"TensorBoard log dir: {self.tensorboard_log_dir}")

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _true_open_boundary_idx(self, ft, tol=None):
        if ("boundary_idx_ring1" not in ft) or ft["boundary_idx_ring1"] is None:
            return torch.empty(0, dtype=torch.long, device=ft["uv"].device)

        bidx = torch.unique(ft["boundary_idx_ring1"].to(dtype=torch.long))
        if bidx.numel() == 0:
            return bidx

        uv = ft["uv"]
        u = uv[:, 0]
        v = uv[:, 1]

        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        if tol is None:
            u_span = (u.max() - u.min()).abs()
            v_span = (v.max() - v.min()).abs()
            base_span = torch.maximum(
                u_span,
                v_span,
            ).clamp_min(torch.as_tensor(1.0, device=uv.device, dtype=uv.dtype))
            tol = 1e-4 * float(base_span.detach().item())

        ub = u[bidx]
        vb = v[bidx]
        keep = torch.ones_like(bidx, dtype=torch.bool)

        if u_periodic:
            umin = u.min()
            umax = u.max()
            is_u_seam = (ub - umin).abs() <= tol
            is_u_seam = is_u_seam | ((ub - umax).abs() <= tol)
            keep = keep & (~is_u_seam)

        if v_periodic:
            vmin = v.min()
            vmax = v.max()
            is_v_seam = (vb - vmin).abs() <= tol
            is_v_seam = is_v_seam | ((vb - vmax).abs() <= tol)
            keep = keep & (~is_v_seam)

        return bidx[keep]

    def _ordered_true_open_boundary(self, ft):
        bidx = self._true_open_boundary_idx(ft)
        if bidx.numel() == 0 or ft.get("faces_ijk", None) is None:
            return bidx, None

        device = bidx.device
        boundary_set = set(int(i) for i in bidx.detach().cpu().tolist())
        if len(boundary_set) < 2:
            return bidx, None

        faces = ft["faces_ijk"].detach().cpu().to(torch.long)
        edge_count = {}
        for a, b, c in faces.tolist():
            for i, j in ((a, b), (b, c), (c, a)):
                key = (i, j) if i < j else (j, i)
                edge_count[key] = edge_count.get(key, 0) + 1

        adj = {i: [] for i in boundary_set}
        for (i, j), count in edge_count.items():
            if count == 1 and i in boundary_set and j in boundary_set:
                adj[i].append(j)
                adj[j].append(i)

        if not any(adj.values()):
            return bidx, None

        ordered = []
        loop_ids = []
        visited_edges = set()

        def edge_key(i, j):
            return (i, j) if i < j else (j, i)

        starts = [i for i, nbrs in adj.items() if len(nbrs) == 1]
        starts.extend(i for i in adj.keys() if i not in starts)

        loop_id = 0
        for start in starts:
            has_unused = any(edge_key(start, nb) not in visited_edges for nb in adj[start])
            if not has_unused:
                continue

            chain = [start]
            prev = None
            cur = start
            while True:
                next_nodes = [
                    nb for nb in adj[cur]
                    if nb != prev and edge_key(cur, nb) not in visited_edges
                ]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                visited_edges.add(edge_key(cur, nxt))
                if nxt == start:
                    break
                chain.append(nxt)
                prev, cur = cur, nxt

            if len(chain) >= 2:
                ordered.extend(chain)
                loop_ids.extend([loop_id] * len(chain))
                loop_id += 1

        if not ordered:
            return bidx, None

        ordered_idx = torch.tensor(ordered, dtype=torch.long, device=device)
        loop_id_t = torch.tensor(loop_ids, dtype=torch.long, device=device)
        return ordered_idx, loop_id_t

    @staticmethod
    def _to_float_if_finite(x):
        if isinstance(x, torch.Tensor):
            x = x.reshape(())
            if torch.isfinite(x).item():
                return float(x.detach().item())
            return None
        try:
            x = float(x)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    def _tb_add_scalar(self, tag: str, value, step: int):
        if self.writer is None:
            return
        v = self._to_float_if_finite(value)
        if v is not None:
            self.writer.add_scalar(tag, v, step)

    def _tb_add_histogram(self, tag: str, value: torch.Tensor, step: int):
        if self.writer is None or value is None:
            return
        try:
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                finite_mask = torch.isfinite(value)
                if finite_mask.any():
                    self.writer.add_histogram(tag, value[finite_mask].detach().cpu(), step)
        except Exception:
            pass


    def _tb_log_step(
        self,
        step: int,
        row: dict,
        rho: torch.Tensor,
        fiber_surface: torch.Tensor,
        seeds_list: list[torch.Tensor],
        pred_list: list[dict],
    ):
        if self.writer is None:
            return

        self._tb_add_scalar("Loss/Total", row["L_total"], step)
        self._tb_add_scalar("Loss/Volume", row["loss_vol"], step)
        self._tb_add_scalar("Loss/Repulsion", row["loss_rep"], step)
        self._tb_add_scalar("Loss/Boundary", row["loss_bnd"], step)
        self._tb_add_scalar("Loss/CurveLength", row["loss_curve_length"], step)
        self._tb_add_scalar("Loss/CellEdgeUniform", row["loss_cell_edge_uniform"], step)
        self._tb_add_scalar("Loss/FEM", row["loss_fem"], step)
        self._tb_add_scalar("Loss/Compliance", row["loss_comp"], step)

        self._tb_add_scalar("Physics/ComplianceRaw", row["comp"], step)
        self._tb_add_scalar("Physics/VolumeFraction", row["vol_frac"], step)
        self._tb_add_scalar("Physics/VF_total", row["VF_total"], step)
        self._tb_add_scalar("Physics/VF_eff_total", row["VF_eff_total"], step)
        self._tb_add_scalar("Physics/VolumeFractionEffective", row["vol_frac_eff"], step)
        self._tb_add_scalar("Physics/VolumeDeviation", row["vol_dev"], step)
        self._tb_add_scalar("Physics/VolumeDeviationEffective", row["vol_dev_eff"], step)
        self._tb_add_scalar("Physics/WGeoMean", row["w_geo_mean"], step)

        self._tb_add_scalar("Density/Min", row["rho_min"], step)
        self._tb_add_scalar("Density/Mean", row["rho_mean"], step)
        self._tb_add_scalar("Density/Max", row["rho_max"], step)

        self._tb_add_scalar("Train/DeltaRho", row["drho"], step)
        self._tb_add_scalar("Train/DeltaSeed", row["dseed"], step)
        self._tb_add_scalar("Train/GradMean", row["grad_mean"], step)
        self._tb_add_scalar("Train/BestScore", row["best_score"], step)
        self._tb_add_scalar("Train/BestStep", row["best_step"], step)
        self._tb_add_scalar("Train/FEMValid", 1.0 if row["fem_valid"] else 0.0, step)
        self._tb_add_scalar(
            "Train/OptimizerStepSkipped",
            1.0 if row["optimizer_step_skipped"] else 0.0,
            step,
        )
        self._tb_add_scalar("Geometry/HMean", row["h_mean"], step)
        self._tb_add_scalar("Geometry/CenterlineRadius", row["centerline_radius_mean"], step)
        self._tb_add_scalar("Train/Tau", row["tau"], step)
        self._tb_add_scalar("Train/BestHardScore", row["best_hard_score"], step)
        self._tb_add_scalar("Train/BestHardStep", row["best_hard_step"], step)

        fiber_norm = torch.linalg.norm(fiber_surface, dim=1)
        if fiber_norm.numel() > 0:
            self._tb_add_scalar("Fiber/NormMean", fiber_norm.mean(), step)
            self._tb_add_scalar("Fiber/NormMin", fiber_norm.min(), step)
            self._tb_add_scalar("Fiber/NormMax", fiber_norm.max(), step)

        if step % self.cfg.tb_log_histograms_every == 0 or step == self.cfg.num_steps - 1:
            self._tb_add_histogram("Density/Rho", rho, step)
            self._tb_add_histogram("Fiber/Norm", fiber_norm, step)

            if len(seeds_list) > 0:
                all_seeds = torch.cat(seeds_list, dim=0)
                self._tb_add_histogram("Seeds/All", all_seeds, step)
                if all_seeds.shape[1] >= 1:
                    self._tb_add_histogram("Seeds/U", all_seeds[:, 0], step)
                if all_seeds.shape[1] >= 2:
                    self._tb_add_histogram("Seeds/V", all_seeds[:, 1], step)

            w_geo_vals = []
            for p in pred_list:
                if "w_geo" in p and p["w_geo"] is not None:
                    w_geo_vals.append(self._pair_upper_values(p["w_geo"]))
            if len(w_geo_vals) > 0:
                self._tb_add_histogram("Geometry/WGeo", torch.cat(w_geo_vals, dim=0), step)
            h_vals = []
            centerline_radius_vals = []

            for p in pred_list:
                if "h" in p and p["h"] is not None:
                    h_vals.append(p["h"].reshape(-1))
                if "centerline_radius" in p and p["centerline_radius"] is not None:
                    centerline_radius_vals.append(p["centerline_radius"].reshape(-1))

            if h_vals: self._tb_add_histogram("Geometry/HHist", torch.cat(h_vals, dim=0), step)
            if centerline_radius_vals: self._tb_add_histogram("Geometry/CenterlineRadiusHist", torch.cat(centerline_radius_vals, dim=0), step)

            tau_vals = []
            for p in pred_list:
                if p.get("tau") is not None:
                    tau_value = p["tau"]
                    if isinstance(tau_value, torch.Tensor):
                        tau_vals.append(tau_value.reshape(-1))
                    else:
                        tau_vals.append(torch.as_tensor([float(tau_value)]))

            if tau_vals:
                self._tb_add_histogram("Train/TauHist", torch.cat(tau_vals, dim=0), step)

        if self.last_fem_debug:
            dbg = self.last_fem_debug
            for key in [
                "density_raw_min",
                "density_raw_mean",
                "density_raw_max",
                "density_min",
                "density_mean",
                "density_max",
                "fiber_norm_min",
                "fiber_norm_mean",
                "fiber_norm_max",
                "void_fraction_lt_1e_2_raw",
                "void_fraction_lt_5e_2_raw",
                "void_fraction_lt_floor_raw",
            ]:
                if key in dbg:
                    self._tb_add_scalar(f"FEMDebug/{key}", dbg[key], step)

            if "fem_valid" in dbg:
                self._tb_add_scalar("FEMDebug/Valid", 1.0 if dbg["fem_valid"] else 0.0, step)

            if dbg.get("failure_reason"):
                self.writer.add_text("FEMDebug/FailureReason", str(dbg["failure_reason"]), step)

    # ------------------------------------------------------------------
    # Losses / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def volume_loss_constant_height(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return Loss_Volume.constant_height(
            rho=rho,
            A_v=A_v,
            target_volfrac=target_volfrac,
            eps=eps,
        )

    @staticmethod
    def powered_volume_fraction(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return Loss_Volume.powered_fraction(rho=rho, A_v=A_v, power=power, eps=eps)

    @classmethod
    def volume_loss_powered(
        cls,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return Loss_Volume().powered(
            rho=rho,
            A_v=A_v,
            target_volfrac=target_volfrac,
            power=power,
            eps=eps,
        )


    @staticmethod
    def ramp_weight(step: int, total_steps: int, start_frac: float, ramp_frac: float) -> float:
        if total_steps <= 0:
            return 0.0
        start_step = max(int(start_frac * total_steps), 0)
        ramp_steps = max(int(ramp_frac * total_steps), 1)
        if step <= start_step:
            return 0.0
        if step >= start_step + ramp_steps:
            return 1.0
        return float(step - start_step) / float(ramp_steps)

    def seed_offset_scale_for_step(self, step: int) -> float:
        cfg = self.cfg
        start = cfg.Offset_scale if cfg.seed_offset_scale_start is None else cfg.seed_offset_scale_start
        final = start if cfg.seed_offset_scale_final is None else cfg.seed_offset_scale_final
        if cfg.num_steps <= 0:
            return float(final)

        t = min(max(float(step) / max(float(cfg.seed_offset_scale_ramp_frac) * float(cfg.num_steps), 1.0), 0.0), 1.0)
        # Smooth decay: exploration changes gently instead of snapping at a milestone.
        t = t * t * (3.0 - 2.0 * t)
        return float((1.0 - t) * float(start) + t * float(final))

    def allow_seed_outside_domain_for_step(self, step: int) -> bool:
        cfg = self.cfg
        if not bool(cfg.allow_seed_outside_domain):
            return False
        warmup_step = int(round(float(cfg.allow_seed_outside_domain_warmup_frac) * float(cfg.num_steps)))
        return int(step) >= warmup_step

    def early_stop_start_step(self) -> int:
        value = float(self.cfg.early_stop_start)
        if 0.0 <= value <= 1.0:
            return int(round(value * float(self.cfg.num_steps)))
        return int(round(value))

    @staticmethod
    def min_pairwise_seed_distance(seeds_list: list[torch.Tensor]) -> float:
        min_seed_dist = float("inf")
        for seeds_i in seeds_list:
            if seeds_i.shape[0] < 2:
                continue
            d_seed = torch.cdist(seeds_i, seeds_i)
            eye = torch.eye(
                seeds_i.shape[0],
                dtype=torch.bool,
                device=seeds_i.device,
            )
            d_seed = d_seed.masked_fill(eye, float("inf"))
            min_seed_dist = min(min_seed_dist, float(d_seed.min().detach().item()))
        if not math.isfinite(min_seed_dist):
            min_seed_dist = 0.0
        return min_seed_dist

    @staticmethod
    def project_seed_spacing(
        seeds_list: list[torch.Tensor],
        min_dist: float,
        iters: int = 8,
        eps_uv: float = 1e-4,
        detach: bool = True,
        clamp_to_domain: bool = True,
    ) -> list[torch.Tensor]:
        repaired = [(s.detach() if detach else s).clone() for s in seeds_list]
        if min_dist <= 0.0 or iters <= 0:
            return repaired

        for _ in range(int(iters)):
            for seeds in repaired:
                s = int(seeds.shape[0])
                if s < 2:
                    continue
                for i in range(s - 1):
                    for j in range(i + 1, s):
                        diff = seeds[i] - seeds[j]
                        dist = torch.linalg.norm(diff)
                        shortfall = float(min_dist) - float(dist.detach().item())
                        if shortfall <= 0.0:
                            continue
                        if float(dist.detach().item()) > 1e-8:
                            direction = diff / dist.clamp_min(1e-8)
                        else:
                            # Deterministic fallback direction for exact overlaps.
                            angle = torch.as_tensor(
                                2.399963229728653 * float(i + 1) + 1.61803398875 * float(j + 1),
                                dtype=seeds.dtype,
                                device=seeds.device,
                            )
                            direction = torch.stack((torch.cos(angle), torch.sin(angle)))
                        step = 0.5 * shortfall * direction
                        seeds[i] = seeds[i] + step
                        seeds[j] = seeds[j] - step
                if clamp_to_domain:
                    seeds.clamp_(float(eps_uv), 1.0 - float(eps_uv))
        return repaired

    def _tau_for_step(self, step: int) -> float:
        cfg = self.cfg
        tau_start = float(cfg.tau)
        tau_end = tau_start if cfg.tau_anneal_final is None else float(cfg.tau_anneal_final)

        anneal = self.ramp_weight(
            step=step,
            total_steps=cfg.num_steps,
            start_frac=cfg.tau_anneal_start_frac,
            ramp_frac=cfg.tau_anneal_ramp_frac,
        )
        return (1.0 - anneal) * tau_start + anneal * tau_end

    def _fallback_tau_value(self) -> float:
        return float(self.cfg.tau)

    @staticmethod
    def _format_elapsed_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        total = int(round(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d} h {minutes:02d} min {secs:02d} sec"
        return f"{minutes:d} min {secs:02d} sec"

    @staticmethod
    def _volume_metric_definitions() -> dict[str, str]:
        return {
            "VF_total": "Total volume fraction. Area-weighted mean density of the full shell field, including boundary attachment.",
            "VF_eff_total": "Efficient total volume fraction. Area-weighted mean of the powered full shell density; lower density material contributes less.",
            "VF_int": "Interior (Voronoi edges only) volume fraction. Area-weighted mean density of the interior Voronoi-edge field without boundary attachment.",
            "VF_eff_int": "Efficient Interior (Voronoi edges only) volume fraction. Area-weighted mean of the powered interior Voronoi-edge density.",
        }

    def _save_optimization_logs(
        self,
        output_folder: str | None,
        history: list[dict],
        best_row: dict | None,
        best_score: float,
        best_step: int,
        computation_time_sec: float,
        returned_best_source: str,
    ) -> str | None:
        if not output_folder:
            return None

        log_dir = os.path.join(os.path.normpath(str(output_folder)), "OptimizationLogs")
        os.makedirs(log_dir, exist_ok=True)

        config_path = os.path.join(log_dir, "training_parameters.txt")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("Training Parameters\n")
            f.write("===================\n")
            for key, value in sorted(asdict(self.cfg).items()):
                f.write(f"{key}: {value}\n")

        definitions_path = os.path.join(log_dir, "volume_metric_definitions.txt")
        with open(definitions_path, "w", encoding="utf-8") as f:
            f.write("Volume Metric Definitions\n")
            f.write("=========================\n")
            f.write("Legacy names in older logs:\n")
            f.write("Tot_VolFrac = VF_total\n")
            f.write("HVD_OFRAC / HVD_VolFrac = VF_int\n")
            f.write("EFF_volfrac / Eff_VolFrac = VF_eff_int\n\n")
            for key, description in self._volume_metric_definitions().items():
                f.write(f"{key}: {description}\n")

        summary = {
            "best_score": best_score,
            "best_step": best_step,
            "returned_best_source": returned_best_source,
            "computation_time": self._format_elapsed_time(computation_time_sec),
            "computation_time_seconds": computation_time_sec,
            "volume_metrics": self._volume_metric_definitions(),
            "best_row": best_row or {},
        }
        summary_path = os.path.join(log_dir, "optimization_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        history_path = os.path.join(log_dir, "optimization_history.csv")
        if history:
            fieldnames = []
            for row in history:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with open(history_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(history)
        else:
            with open(history_path, "w", encoding="utf-8", newline="") as f:
                f.write("")

        return log_dir

    def _timelapse_geometry_summary(self, face_tensors) -> str:


        surface_pts = int(sum(int(ft["points_xyz"].shape[0]) for ft in face_tensors))

        if self.shell_problem is not None and getattr(self.shell_problem, "brep_bbox", None) is not None:
            bbox = self.shell_problem.brep_bbox
            bbox_dims = (
                float(bbox["xmax"] - bbox["xmin"]),
                float(bbox["ymax"] - bbox["ymin"]),
                float(bbox["zmax"] - bbox["zmin"]),
            )
        else:
            xyz_all = torch.cat([ft["points_xyz"].detach() for ft in face_tensors], dim=0)
            bbox_t = xyz_all.amax(dim=0) - xyz_all.amin(dim=0)
            bbox_dims = tuple(float(v) for v in bbox_t.detach().cpu().tolist())

        load_value = (
            float(getattr(self.shell_problem, "Load_magnitude", 0.0))
            if self.shell_problem is not None
            else 0.0
        )

        bbox_text = " x ".join(f"{dim:.4g}" for dim in bbox_dims)
        return (
            f"BBox: {bbox_text}, "
            f"SurfacePts={surface_pts} "
        )

    def _timelapse_optimized_parameter_summary(self) -> str:
        cfg = self.cfg
        params = [
            f"seed positions ({int(cfg.seed_number)})",
            "global strut width" if not cfg.freeze_w else f"width fixed={float(cfg.w_const):.6g}",
            f"tau={self._fallback_tau_value():.6g}",
        ]
        return "Optimized: " + ", ".join(params)

    @staticmethod
    def _clone_pred_list(pred_list: list[dict]) -> list[dict]:
        def _clone_value(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            return value

        return [
            {
                "face_id": p["face_id"],
                "seeds_raw": p["seeds_raw"].detach().clone(),
                "w_raw": p["w_raw"].detach().clone(),
                "h_raw": _clone_value(p.get("h_raw")),
                "tau": _clone_value(p.get("tau")),
                "w_geo": _clone_value(p.get("w_geo")),
                "h": _clone_value(p.get("h")),
                "centerline_radius": _clone_value(p.get("centerline_radius")),
                "seeds_uv": _clone_value(p.get("seeds_uv")),
                "seeds_xyz": _clone_value(p.get("seeds_xyz")),
                "edge_curves_uv": _clone_value(p.get("edge_curves_uv")),
                "edge_curves_xyz": _clone_value(p.get("edge_curves_xyz")),
            }
            for p in pred_list
        ]
    @staticmethod
    def _scalar_tensor_is_finite(x: torch.Tensor | float | int) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).reshape(()).detach().item())
        return math.isfinite(float(x))

    @staticmethod
    def _require_decoder_keys(decoder_out: dict, required_keys: list[str]):
        missing = [k for k in required_keys if k not in decoder_out]
        if missing:
            raise ValueError(
                f"Decoder output missing required keys: {missing}. "
                f"Available keys: {list(decoder_out.keys())}"
            )

    def _record_invalid_fem_debug(
        self,
        debug: dict,
        reason: str,
        save_debug_history: bool,
    ):
        debug = dict(debug)
        debug["fem_valid"] = False
        debug["failure_reason"] = reason
        self.last_fem_debug = debug
        if save_debug_history:
            self.fem_debug_history.append(debug.copy())

    # ------------------------------------------------------------------
    # Model / optimizer builders
        # ------------------------------------------------------------------
    def _build_single_face_models(
        self,
        device,
        seed_number,
        u_periodic,
        v_periodic,
        boundary_solid_idx=None,
        face_tensor=None,
    ):
        decoder = self.decoder_cls(
            **self._decoder_init_kwargs(
                device=device,
                seed_number=seed_number,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
                boundary_solid_idx=boundary_solid_idx,
                face_tensor=face_tensor,
            )
        ).to(device)

        ppnet = self.ppnet_cls(
            n_seeds=seed_number,
            freeze_w=self.cfg.freeze_w,
            w_const=self.cfg.w_const,   
            w_head_bias_init=(
                float(self.cfg.decoder_raw_temp)
                * math.atanh(
                    2.0 * max(min(float(self.cfg.width_target_frac), 1.0 - 1e-4), 1e-4)
                    - 1.0
                )
                if self.cfg.w_head_bias_init is None
                else float(self.cfg.w_head_bias_init)
            ),
            allow_seed_outside_domain=(
                bool(self.cfg.allow_seed_outside_domain)
                and float(self.cfg.allow_seed_outside_domain_warmup_frac) <= 0.0
            ),
            seed_domain_margin=self.cfg.seed_domain_margin,
            use_independent_seed_offsets=self.cfg.use_independent_seed_offsets,
            independent_seed_offset_max=self.cfg.independent_seed_offset_max,
        ).to(device)

        return decoder, ppnet

    def _decoder_face_mesh_for_face(self, face_tensor):
        if self.face_mesh is None:
            return face_tensor
        if isinstance(self.face_mesh, dict) and "face_tensors" in self.face_mesh:
            tensors = self.face_mesh["face_tensors"]
            if isinstance(tensors, (list, tuple)):
                return tensors[int(getattr(self.cfg, "training_face_index", 0))]
            return tensors
        if isinstance(self.face_mesh, (list, tuple)):
            return self.face_mesh[int(getattr(self.cfg, "training_face_index", 0))]
        return self.face_mesh

    def _decoder_init_kwargs(self, device, seed_number, u_periodic, v_periodic, boundary_solid_idx=None, face_tensor=None):
        return {
            "Cad_domain": self.Cad_domain,
            "face_mesh": self._decoder_face_mesh_for_face(face_tensor),
            "return_xyz": True,
            "tube_curve_samples": 64,
            "edge_trim_samples": 32,
            "tube_density_tau": 0.002,
            "tube_fiber_tau": 0.002,
            "face_u_periodic": bool(u_periodic),
            "face_v_periodic": bool(v_periodic),
            "duplicate_merge_sigma": self.cfg.decoder_duplicate_merge_sigma,
        }

    def _build_face_model(self, face_tensor, device):
        return self._build_single_face_models(
            device=device,
            seed_number=self.cfg.seed_number,
            u_periodic=face_tensor.get("u_periodic", False),
            v_periodic=face_tensor.get("v_periodic", False),
            boundary_solid_idx=self._true_open_boundary_idx(face_tensor),
            face_tensor=face_tensor,
        )

    def _save_optimized_shell_function(
        self,
        save_dir,
        decoder,
        ppnet,
        face_tensor,
        best_pred,
        best_score,
        best_step,
        returned_best_source,
        final_shape_density=None,
        final_shape_fiber_direction=None,
    ):
        if save_dir is None:
            return None
        if best_pred is None:
            return None

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "optimized_shell_function.pt")
        device = face_tensor["uv"].device

        package = {
            "package_type": "OptimizedShellFunction",
            "package_version": OptimizedShellFunction.package_version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(self.cfg),
            "decoder_class": {
                "module": decoder.__class__.__module__,
                "name": decoder.__class__.__name__,
            },
            "ppnet_class": {
                "module": ppnet.__class__.__module__,
                "name": ppnet.__class__.__name__,
            },
            "decoder_init_kwargs": _cpu_detached_tree(
                self._decoder_init_kwargs(
                    device=device,
                    seed_number=int(getattr(decoder, "n_seeds", self.cfg.seed_number)),
                    u_periodic=face_tensor.get("u_periodic", False),
                    v_periodic=face_tensor.get("v_periodic", False),
                    face_tensor=face_tensor,
                )
            ),
            "decoder_state_dict": _cpu_detached_tree(decoder.state_dict()),
            "ppnet_state_dict": _cpu_detached_tree(ppnet.state_dict()),
            "best_pred": _cpu_detached_tree(best_pred),
            "best_score": float(best_score),
            "best_step": int(best_step),
            "returned_best_source": returned_best_source,
            "face_metadata": {
                "face_id": _cpu_detached_tree(face_tensor.get("face_id", 0)),
                "u_periodic": bool(face_tensor.get("u_periodic", False)),
                "v_periodic": bool(face_tensor.get("v_periodic", False)),
                "num_surface_points": int(face_tensor["uv"].shape[0]),
            },
            "final_shape_density": _cpu_detached_tree(final_shape_density),
            "final_shape_fiber_direction": _cpu_detached_tree(final_shape_fiber_direction),
        }
        torch.save(package, path)
        return path

    @staticmethod
    def load_optimized_shell_function(path, decoder_cls=None, device=None):
        return OptimizedShellFunction.load(path, decoder_cls=decoder_cls, device=device)

    def _build_optimizer(self, ppnet, decoder):
        cfg = self.cfg
        param_groups = []

        seed_refine_params = list(ppnet.seed_refine.parameters())
        if getattr(ppnet, "seed_id_embed", None) is not None:
            seed_refine_params.extend(ppnet.seed_id_embed.parameters())

        param_groups.extend([
            {"params": seed_refine_params, "lr": cfg.lr_seed_refine},
            {"params": ppnet.delta_head.parameters(), "lr": cfg.lr_delta_head},
            {"params": [ppnet.global_latent], "lr": cfg.lr_mlp},
        ])

        independent_seed_offsets = getattr(ppnet, "independent_seed_offsets", None)
        if independent_seed_offsets is not None and independent_seed_offsets.requires_grad:
            param_groups.append(
                {
                    "params": [independent_seed_offsets],
                    "lr": cfg.lr_independent_seed_offsets,
                }
            )

        w_head = getattr(ppnet, "w_head", None)
        if w_head is not None:
            param_groups.append({"params": w_head.parameters(), "lr": cfg.lr_w_head})

        return torch.optim.Adam(param_groups)

    def _build_scheduler(self, opt, milestones):
        cfg = self.cfg
        if not getattr(cfg, "scheduler_milestones", None):
            return None
        return torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=list(milestones),
            gamma=cfg.scheduler_gamma,
        )

    @staticmethod
    def _copy_optimizer_lrs(src_opt, dst_opt):
        for src_group, dst_group in zip(src_opt.param_groups, dst_opt.param_groups):
            dst_group["lr"] = src_group.get("lr", dst_group["lr"])

    @staticmethod
    def _clone_module_state_dict(module):
        return {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

    def _prune_inactive_seeds(
        self,
        ppnet,
        decoder,
        uv_anchor: torch.Tensor,
        pred_i: dict,
    ) -> tuple[bool, torch.Tensor, int, int]:
        active_mask = pred_i.get("seed_active_mask", None)
        if active_mask is None:
            return False, uv_anchor, int(getattr(ppnet, "n_seeds", uv_anchor.shape[0])), 0

        active_mask = active_mask.detach().to(device=uv_anchor.device, dtype=torch.bool).reshape(-1)
        old_count = int(active_mask.numel())
        active_idx = torch.nonzero(active_mask, as_tuple=False).flatten()
        new_count = int(active_idx.numel())

        min_keep = int(self.cfg.min_active_seeds or 1)
        if new_count <= 0 or new_count >= old_count or new_count < min_keep:
            return False, uv_anchor, old_count, old_count - new_count

        with torch.no_grad():
            uv_anchor_pruned = uv_anchor.index_select(0, active_idx).detach().clone()

            ppnet.n_seeds = new_count
            seed_identity = getattr(ppnet, "seed_identity", None)
            embedding = getattr(seed_identity, "embedding", None)
            if embedding is not None:
                old_embedding = embedding
                new_embedding = torch.nn.Embedding(new_count, old_embedding.embedding_dim).to(
                    device=old_embedding.weight.device,
                    dtype=old_embedding.weight.dtype,
                )
                new_embedding.weight.copy_(old_embedding.weight.index_select(0, active_idx.to(old_embedding.weight.device)))
                seed_identity.embedding = new_embedding

            independent_seed_offsets = getattr(ppnet, "independent_seed_offsets", None)
            if independent_seed_offsets is not None:
                active_idx_offsets = active_idx.to(independent_seed_offsets.device)
                ppnet.seed_free_offset_raw = torch.nn.Parameter(
                    independent_seed_offsets.index_select(0, active_idx_offsets).detach().clone()
                )

            decoder.n_seeds = new_count
            decoder.seed_face_id = decoder.seed_face_id.index_select(
                0,
                active_idx.to(decoder.seed_face_id.device),
            ).detach().clone()

        return True, uv_anchor_pruned, old_count, old_count - new_count

    @staticmethod
    def _decoder_seed_state_for_pred(decoder, pred_i: dict, device) -> tuple[int | None, torch.Tensor | None]:
        old_n_seeds_raw = getattr(decoder, "n_seeds", None)
        old_n_seeds = None if old_n_seeds_raw is None else int(old_n_seeds_raw)
        old_seed_face_id = (
            decoder.seed_face_id.detach().clone()
            if hasattr(decoder, "seed_face_id")
            else None
        )
        pred_seed_count = int(pred_i["seeds_raw"].shape[0])
        if old_n_seeds is None or pred_seed_count != old_n_seeds:
            decoder.n_seeds = pred_seed_count
            if old_seed_face_id is not None:
                decoder.seed_face_id = torch.zeros(
                    pred_seed_count,
                    dtype=torch.long,
                    device=device,
                )
        return old_n_seeds, old_seed_face_id

    @staticmethod
    def _restore_decoder_seed_state(decoder, state: tuple[int | None, torch.Tensor | None]):
        old_n_seeds, old_seed_face_id = state
        decoder.n_seeds = None if old_n_seeds is None else int(old_n_seeds)
        if old_seed_face_id is not None:
            decoder.seed_face_id = old_seed_face_id

    @staticmethod
    def _pair_upper_values(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError("Expected tensor for pair reduction")
        if t.ndim < 2:
            return t.reshape(-1)

        mask = torch.triu(
            torch.ones(t.shape[-2], t.shape[-1], device=t.device, dtype=torch.bool),
            diagonal=1,
        )
        vals = t[..., mask]
        if vals.numel() == 0:
            return t.reshape(-1)
        return vals.reshape(-1)

    @staticmethod
    def _face_id_key(face_id) -> int:
        if isinstance(face_id, torch.Tensor):
            if face_id.numel() == 0:
                return 0
            return int(face_id.detach().reshape(-1)[0].item())
        return int(face_id)
    
    def _init_face_seed(self, face_tensor):
        cfg = self.cfg
        boundary = self._true_open_boundary_idx(face_tensor)
        if not cfg.use_balanced_seed_init:
            seed_idx = self._random_seed_indices(
                n_points=int(face_tensor["uv"].shape[0]),
                n_samples=int(cfg.seed_number),
                exclude_idx=boundary,
                seed=cfg.seed_init_fps_seed,
                device=face_tensor["uv"].device,
            )
            return face_tensor["uv"][seed_idx].clone()

        seed_idx = self.generator.fps_3d(
            face_tensor["points_xyz"],
            cfg.seed_number,
            exclude_idx=boundary,
            seed = cfg.seed_init_fps_seed,
        )
        return face_tensor["uv"][seed_idx].clone()

    @staticmethod
    def _random_seed_indices(
        n_points: int,
        n_samples: int,
        exclude_idx=None,
        seed: int | None = None,
        device=None,
    ) -> torch.Tensor:
        device = torch.device("cpu") if device is None else torch.device(device)
        n_points = int(n_points)
        n_samples = min(int(n_samples), n_points)
        if n_samples <= 0 or n_points <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        candidate_mask = torch.ones((n_points,), dtype=torch.bool, device=device)
        if exclude_idx is not None:
            exclude_idx = torch.as_tensor(exclude_idx, dtype=torch.long, device=device)
            exclude_idx = exclude_idx[(exclude_idx >= 0) & (exclude_idx < n_points)]
            if exclude_idx.numel() > 0:
                candidate_mask[exclude_idx] = False

        candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(n_points, dtype=torch.long, device=device)
        n_samples = min(n_samples, int(candidates.numel()))

        if seed is None:
            order = torch.randperm(candidates.numel(), device=device)
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            order_cpu = torch.randperm(candidates.numel(), generator=gen)
            order = order_cpu.to(device=device)
        return candidates[order[:n_samples]].to(dtype=torch.long)

    def _seed_points_xyz(self, seeds, face_tensor):
        return self.generator.seeds_uv_to_xyz_nearest(
            seeds,
            face_tensor["uv"],
            face_tensor["points_xyz"],
        )

    def _finite_or_default(self, x: torch.Tensor | float | int, default: float = float("nan")) -> float:
        if self._scalar_tensor_is_finite(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().item())
            return float(x)
        return default

    @staticmethod
    def _named_trainable_params(modules):
        for mi, module in enumerate(modules):
            for pn, p in module.named_parameters():
                if p.requires_grad:
                    yield mi, pn, p

    @classmethod
    def _trainable_zero(cls, modules, dtype, device):
        zero = torch.zeros((), dtype=dtype, device=device)
        for _mi, _pn, p in cls._named_trainable_params(modules):
            return p.reshape(-1)[0] * 0.0
        return zero

    @classmethod
    def _nonfinite_grad_info(cls, modules):
        bad = []
        for mi, pn, p in cls._named_trainable_params(modules):
            g = p.grad
            if g is not None and not torch.isfinite(g).all():
                bad.append((mi, pn))
        return bad

    @classmethod
    def _nonfinite_grad_cause_summary(
        cls,
        modules,
        bad_grad_info,
        loss_terms=None,
        fem_is_valid=True,
        fem_failure_reason=None,
    ) -> str:
        reasons = []

        if loss_terms:
            bad_losses = []
            finite_losses = []
            for name, value in loss_terms:
                if value is None:
                    continue
                if cls._scalar_tensor_is_finite(value):
                    raw = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
                    finite_losses.append((name, raw))
                else:
                    bad_losses.append(name)

            if bad_losses:
                reasons.append("non-finite loss term(s): " + ", ".join(bad_losses[:5]))
            elif finite_losses:
                largest_name, largest_value = max(finite_losses, key=lambda item: abs(item[1]))
                reasons.append(f"all tracked losses finite; largest={largest_name}={largest_value:.3e}")

        if not fem_is_valid:
            if fem_failure_reason:
                reasons.append(f"FEM invalid: {fem_failure_reason}")
            else:
                reasons.append("FEM invalid")

        bad_set = set(bad_grad_info)
        for mi, pn, p in cls._named_trainable_params(modules):
            if (mi, pn) not in bad_set or p.grad is None:
                continue
            g = p.grad.detach()
            nan_count = int(torch.isnan(g).sum().item())
            posinf_count = int(torch.isposinf(g).sum().item())
            neginf_count = int(torch.isneginf(g).sum().item())
            reasons.append(f"bad grad at face={mi}:{pn} (nan={nan_count}, +inf={posinf_count}, -inf={neginf_count})")
            break

        if not reasons:
            reasons.append("likely backward overflow or unstable derivative")
        elif loss_terms and not any(reason.startswith("non-finite loss") for reason in reasons):
            reasons.append("likely backward overflow or unstable derivative")

        return "Cause: " + "; ".join(reasons)

    @classmethod
    def _nonfinite_param_info(cls, modules):
        bad = []
        for mi, pn, p in cls._named_trainable_params(modules):
            if not torch.isfinite(p).all():
                bad.append((mi, pn))
        return bad

    @staticmethod
    def _restore_param_snapshot(snapshot):
        for p, saved in snapshot.items():
            p.data.copy_(saved)

    @staticmethod
    def _clear_optimizer_state_for_params(opt, params):
        for p in params:
            if p in opt.state:
                opt.state.pop(p, None)

    def _print_fem_failure(self, step: int):
        print(f"\n=== FEM FAILURE AT STEP {step} ===")
        for k, v in self.last_fem_debug.items():
            print(f"{k}: {v}")
        print("Skipping FEM term for this step.\n")

    def _auto_update_w_min_from_face_scale(self, face_tensor):
        cfg = self.cfg
        if not bool(getattr(cfg, "auto_update_wmin", False)):
            return

        cfg.w_min = compute_w_min_from_min_feature_size_3d(
            Xu=face_tensor["Xu"],
            Xv=face_tensor["Xv"],
            min_feature_size_3d=float(cfg.min_feature_size_3d),
        )
        tqdm.write(
            "Auto-updated w_min from min_feature_size_3d="
            f"{float(cfg.min_feature_size_3d):.6g}: w_min={float(cfg.w_min):.6g}"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _validate_face_tensors(self, face_tensors):
        required_keys = [
            "face_id",
            "uv",
            "Xu",
            "Xv",
            "points_xyz",
            "faces_ijk",
            "face_areas",
            "global_vertex_idx",
        ]

        if not isinstance(face_tensors, (list, tuple)) or len(face_tensors) == 0:
            raise ValueError("face_tensors must be a non-empty list.")

        ref_uv = face_tensors[0]["uv"]
        ref_device = ref_uv.device
        ref_dtype = ref_uv.dtype

        for i, ft in enumerate(face_tensors):
            missing = [k for k in required_keys if k not in ft]
            if missing:
                raise ValueError(f"face_tensors[{i}] is missing required keys: {missing}")

            uv = ft["uv"]
            Xu = ft["Xu"]
            Xv = ft["Xv"]
            points_xyz = ft["points_xyz"]
            faces_ijk = ft["faces_ijk"]
            face_areas = ft["face_areas"]
            gidx = ft["global_vertex_idx"]

            if uv.device != ref_device:
                raise ValueError(f"face_tensors[{i}]['uv'] device mismatch: {uv.device} != {ref_device}")
            if uv.dtype != ref_dtype:
                raise ValueError(f"face_tensors[{i}]['uv'] dtype mismatch: {uv.dtype} != {ref_dtype}")

            n_local = uv.shape[0]
            if Xu.shape[0] != n_local or Xv.shape[0] != n_local or points_xyz.shape[0] != n_local:
                raise ValueError(f"face_tensors[{i}] local tensor lengths do not match uv.shape[0]={n_local}")

            if gidx.shape[0] != n_local:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] length mismatch with local vertex count")

            if gidx.dtype != torch.long:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] must be torch.long")

            if gidx.numel() > 0 and int(gidx.min().item()) < 0:
                raise ValueError(f"face_tensors[{i}]['global_vertex_idx'] contains negative indices")

            if faces_ijk.numel() > 0:
                if faces_ijk.dtype != torch.long:
                    raise ValueError(f"face_tensors[{i}]['faces_ijk'] must be torch.long")
                fmin = int(faces_ijk.min().item())
                fmax = int(faces_ijk.max().item())
                if fmin < 0 or fmax >= n_local:
                    raise ValueError(
                        f"face_tensors[{i}]['faces_ijk'] contains invalid local indices "
                        f"(min={fmin}, max={fmax}, n_local={n_local})"
                    )

            if face_areas.ndim != 1:
                raise ValueError(f"face_tensors[{i}]['face_areas'] must be 1D")

            if face_areas.shape[0] != faces_ijk.shape[0]:
                raise ValueError(
                    f"face_tensors[{i}]['face_areas'] length must match number of faces "
                    f"({face_areas.shape[0]} != {faces_ijk.shape[0]})"
                )

    def _select_single_training_face(self, face_tensors):
        if not isinstance(face_tensors, (list, tuple)) or len(face_tensors) == 0:
            raise ValueError("face_tensors must be a non-empty list.")

        face_index = int(getattr(self.cfg, "training_face_index", 0))
        if face_index < 0 or face_index >= len(face_tensors):
            raise IndexError(
                f"training_face_index={face_index} is out of range for "
                f"{len(face_tensors)} face tensor(s)"
            )

        selected = dict(face_tensors[face_index])
        selected["global_vertex_idx"] = torch.arange(
            selected["uv"].shape[0],
            dtype=torch.long,
            device=selected["uv"].device,
        )

        if len(face_tensors) > 1:
            tqdm.write(
                f"Using only face_tensors[{face_index}] for single-face training "
                f"(received {len(face_tensors)} faces)."
            )
        return [selected]

    @staticmethod
    def _build_face_uv_grid(ft, grid_res_u, grid_res_v):
        uv_face = ft["uv"]
        device = uv_face.device
        dtype = uv_face.dtype
        u = uv_face[:, 0]
        v = uv_face[:, 1]

        if bool(ft.get("u_periodic", False)):
            u_lin = torch.linspace(0.0, 1.0, grid_res_u + 1, device=device, dtype=dtype)[:-1]
        else:
            u_lin = torch.linspace(u.min(), u.max(), grid_res_u, device=device, dtype=dtype)

        if bool(ft.get("v_periodic", False)):
            v_lin = torch.linspace(0.0, 1.0, grid_res_v + 1, device=device, dtype=dtype)[:-1]
        else:
            v_lin = torch.linspace(v.min(), v.max(), grid_res_v, device=device, dtype=dtype)

        UU, VV = torch.meshgrid(u_lin, v_lin, indexing="ij")
        uv_grid = torch.stack([UU.reshape(-1), VV.reshape(-1)], dim=1)
        return uv_grid, u_lin, v_lin

    @staticmethod
    def _periodic_uv_min_dist(uv_query, uv_face, u_periodic=False, v_periodic=False, chunk_size=4096):
        if uv_query.numel() == 0 or uv_face.numel() == 0:
            return torch.empty((uv_query.shape[0],), device=uv_query.device, dtype=uv_query.dtype)

        mins = []
        for start in range(0, uv_query.shape[0], chunk_size):
            q = uv_query[start:start + chunk_size]
            diff = q.unsqueeze(1) - uv_face.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            mins.append(torch.norm(diff, dim=-1).min(dim=1).values)
        return torch.cat(mins, dim=0)

    @staticmethod
    def _estimate_uv_mask_tol(
        uv_face: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
        fallback: float = 0.05,
        scale: float = 2.5,
        max_points: int = 2048,
        chunk_size: int = 512,
    ) -> float:
        if uv_face.shape[0] < 2:
            return float(fallback)

        uv_cpu = uv_face.detach().to(device="cpu")
        n = uv_cpu.shape[0]
        if n > max_points:
            sample_idx = torch.linspace(0, n - 1, max_points).round().long()
            uv_cpu = uv_cpu[sample_idx]
            n = uv_cpu.shape[0]

        min_vals = []
        for start in range(0, n, chunk_size):
            q = uv_cpu[start:start + chunk_size]
            diff = q.unsqueeze(1) - uv_cpu.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)

            dist = torch.norm(diff, dim=-1)
            rows = q.shape[0]
            dist[torch.arange(rows), start:start + rows] = float("inf")
            min_vals.append(dist.min(dim=1).values)

        spacing = torch.cat(min_vals, dim=0).median()
        if not torch.isfinite(spacing):
            return float(fallback)
        return float(max(scale * float(spacing.item()), 1e-6))

    def _seed_domain_mask_for_face(self, ft):
        cfg = self.cfg
        if not bool(cfg.use_seed_domain_mask):
            return None
        cached = ft.get("_seed_domain_mask_callable", None)
        if cached is not None:
            return cached
        mask_grid = ft.get("seed_domain_mask_grid", None)
        if mask_grid is not None:
            return mask_grid

        uv_face = ft.get("seed_domain_uv_support", ft["uv"])
        if uv_face.numel() == 0:
            return None

        uv_support = uv_face.detach()
        max_points = int(cfg.seed_domain_mask_max_points)
        if uv_support.shape[0] > max_points:
            sample_idx = torch.linspace(
                0,
                uv_support.shape[0] - 1,
                max_points,
                device=uv_support.device,
            ).round().to(torch.long)
            uv_support = uv_support[sample_idx]

        sigma_value = ft.get("seed_domain_sigma", None)
        if sigma_value is None:
            sigma = self._estimate_uv_mask_tol(
                uv_support,
                u_periodic=bool(ft.get("u_periodic", False)),
                v_periodic=bool(ft.get("v_periodic", False)),
                fallback=float(cfg.boundary_margin),
                scale=float(cfg.seed_domain_mask_support_scale),
            )
        elif torch.is_tensor(sigma_value):
            sigma = float(sigma_value.detach().cpu().item())
        else:
            sigma = float(sigma_value)
        sigma = max(float(sigma), float(cfg.eps))
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        def mask_fn(seeds):
            support = uv_support.to(device=seeds.device, dtype=seeds.dtype)
            diff = seeds.unsqueeze(1) - support.unsqueeze(0)
            if u_periodic:
                du = diff[..., 0]
                diff[..., 0] = du - torch.round(du)
            if v_periodic:
                dv = diff[..., 1]
                diff[..., 1] = dv - torch.round(dv)
            dmin = torch.norm(diff, dim=-1).amin(dim=1)
            sigma_t = torch.as_tensor(sigma, device=seeds.device, dtype=seeds.dtype)
            return torch.exp(-0.5 * (dmin / sigma_t.clamp_min(cfg.eps)).pow(2))

        ft["_seed_domain_mask_callable"] = mask_fn
        return mask_fn

    def build_timelapse_render_cache(
        self,
        face_tensors,
    ):
        cache = []

        for ft in face_tensors:
            device = ft["uv"].device
            uv_dense = ft["uv"]
            xyz_dense = ft["points_xyz"]
            Xu_dense = ft["Xu"]
            Xv_dense = ft["Xv"]

            local_face_id = torch.zeros(
                uv_dense.shape[0], dtype=torch.long, device=device
            )

            boundary_uv_i = None
            boundary_face_id_i = None
            boundary_loop_id_i = None
            true_bidx_i, boundary_loop_id_i = self._ordered_true_open_boundary(ft)
            if true_bidx_i.numel() > 0:
                boundary_uv_i = ft["uv"][true_bidx_i]
                boundary_face_id_i = torch.zeros(
                    boundary_uv_i.shape[0], dtype=torch.long, device=device
                )

            cache.append({
                "face_id": self._face_id_key(ft.get("face_id", 0)),
                "uv_dense": uv_dense,
                "xyz_dense": xyz_dense,
                "points_xyz": xyz_dense,
                "Xu_dense": Xu_dense,
                "Xv_dense": Xv_dense,
                "Xu": Xu_dense,
                "Xv": Xv_dense,
                "local_face_id": local_face_id,
                "boundary_uv": boundary_uv_i,
                "boundary_face_id": boundary_face_id_i,
                "boundary_loop_id": boundary_loop_id_i,
                "seed_domain_mask": self._seed_domain_mask_for_face(ft),
                "faces_ijk": ft["faces_ijk"],
            })

        return cache

    def evaluate_cached_face_fields(self, render_cache, decoder, pred):
        decoder_out = decoder(
            seeds_uv=pred["seeds_raw"],
            w_raw=pred["w_raw"],
            generate_density_fiber=getattr(self.cfg, "generate_decoder_density_fiber", True),
        )

        if getattr(self.cfg, "generate_decoder_density_fiber", True):
            decoder_out = apply_density_postprocess_to_output(
                decoder_out,
                render_cache,
                self.cfg,
                return_debug=False,
            )
            rho_dense = decoder_out["rho"]
            rho_raw_decoder_dense = decoder_out["rho_raw_decoder"]
            rho_postprocessed_dense = decoder_out["rho_postprocessed"]
            fiber3d_dense = decoder_out["fiber3d"]
        else:
            rho_dense, fiber3d_dense, _ = self.neutral_density_fiber_fields(
                render_cache["uv_dense"],
                render_cache.get("Xu_dense", None),
            )
            rho_raw_decoder_dense = rho_dense
            rho_postprocessed_dense = rho_dense

        return {
            "xyz_dense": render_cache["xyz_dense"],
            "rho_dense": rho_dense,
            "rho_raw_decoder_dense": rho_raw_decoder_dense,
            "rho_postprocessed_dense": rho_postprocessed_dense,
            "fiber3d_dense": fiber3d_dense,
            "seeds_uv": decoder_out.get("seeds_uv", decoder_out.get("seeds", None)),
            "topology_seeds_uv": decoder_out.get("topology_seeds_uv", decoder_out.get("seeds_uv", decoder_out.get("seeds", None))),
            "seeds_xyz": decoder_out.get("seeds_xyz", None),
            "edge_curves_uv": decoder_out.get("edge_curves_uv", None),
            "edge_curves_xyz": decoder_out.get("edge_curves_xyz", None),
            "graph": decoder_out.get("graph", None),
            "faces_ijk": render_cache["faces_ijk"],
        }

    @staticmethod
    def _concat_polydata(meshes, scalar_name=None):
        if len(meshes) == 0:
            return None

        pts_parts = []
        face_parts = []
        scalar_parts = []
        offset = 0

        for mesh in meshes:
            pts = np.asarray(mesh.points, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4).copy()
            faces[:, 1:] += offset
            pts_parts.append(pts)
            face_parts.append(faces.reshape(-1))
            if scalar_name is not None:
                scalar_parts.append(np.asarray(mesh[scalar_name], dtype=np.float32))
            offset += pts.shape[0]

        out = pv.PolyData(
            np.concatenate(pts_parts, axis=0),
            np.concatenate(face_parts, axis=0),
        )
        if scalar_name is not None and len(scalar_parts) > 0:
            out[scalar_name] = np.concatenate(scalar_parts, axis=0)
        return out

    @staticmethod
    def _curves_xyz_to_polydata(curves_xyz):
        if curves_xyz is None:
            return None
        if torch.is_tensor(curves_xyz):
            curves_xyz = curves_xyz.detach().cpu().numpy()
        curves_xyz = np.asarray(curves_xyz, dtype=np.float32)
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3 or curves_xyz.shape[0] == 0 or curves_xyz.shape[1] < 2:
            return None
        points = curves_xyz.reshape(-1, 3)
        lines = []
        samples = curves_xyz.shape[1]
        for edge_id in range(curves_xyz.shape[0]):
            base = edge_id * samples
            for j in range(samples - 1):
                lines.extend([2, base + j, base + j + 1])
        return pv.PolyData(points, lines=np.asarray(lines, dtype=np.int64))

    @staticmethod
    def _concat_line_polydata(meshes):
        if len(meshes) == 0:
            return None
        point_parts = []
        line_parts = []
        offset = 0
        for mesh in meshes:
            points = np.asarray(mesh.points, dtype=np.float32)
            lines = np.asarray(mesh.lines, dtype=np.int64).copy()
            cursor = 0
            while cursor < lines.size:
                count = int(lines[cursor])
                lines[cursor + 1:cursor + 1 + count] += offset
                cursor += count + 1
            point_parts.append(points)
            line_parts.append(lines)
            offset += points.shape[0]
        return pv.PolyData(
            np.concatenate(point_parts, axis=0),
            lines=np.concatenate(line_parts, axis=0),
        )

    @staticmethod
    def _composite_to_white(img):
        if img.ndim != 3:
            return img
        if img.shape[2] == 3:
            return img
        if img.shape[2] != 4:
            return img[..., :3]

        rgb = img[..., :3].astype(np.float32)
        alpha = (img[..., 3:4].astype(np.float32) / 255.0)
        white = np.full_like(rgb, 255.0)
        out = rgb * alpha + white * (1.0 - alpha)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _render_offscreen_plotter(plotter, view_name):
        tight_view = None
        if view_name == "xy":
            plotter.enable_parallel_projection()
            plotter.view_xy()
            tight_view = "xy"
        elif view_name == "xz":
            plotter.enable_parallel_projection()
            plotter.view_xz()
            tight_view = "xz"
        elif view_name == "yz":
            plotter.enable_parallel_projection()
            plotter.view_yz()
            tight_view = "yz"
        else:
            plotter.disable_parallel_projection()
            plotter.view_isometric()
            tight_view = None

        plotter.reset_camera()
        if tight_view is not None:
            try:
                plotter.camera.tight(view=tight_view, adjust_render_window=False)
            except Exception:
                pass
            try:
                plotter.camera.zoom(0.90)
            except Exception:
                pass
        else:
            try:
                plotter.camera.zoom(0.94)
            except Exception:
                pass
        img = plotter.screenshot(return_img=True, transparent_background=False)
        return NN_Trainer._composite_to_white(img)

    @staticmethod
    def _add_image_title(img, title, pad=10, band_height=42):
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        title_band = np.full((band_height, img.shape[1], 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.72
        thickness = 2
        text_size, baseline = cv2.getTextSize(title, font, font_scale, thickness)
        x = max(pad, (img.shape[1] - text_size[0]) // 2)
        y = max(pad + text_size[1], (band_height + text_size[1]) // 2 - baseline)
        cv2.putText(
            title_band,
            title,
            (x, y),
            font,
            font_scale,
            (32, 32, 32),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return np.vstack([title_band, img])

    @staticmethod
    def _add_panel_border(img, pad=10, border=2, bg_color=(255, 255, 255), border_color=(180, 186, 195)):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        inner = cv2.copyMakeBorder(
            img,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color,
        )
        return cv2.copyMakeBorder(
            inner,
            border,
            border,
            border,
            border,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )

    @staticmethod
    def _pad_to_size(img, target_h=None, target_w=None, bg_color=(255, 255, 255)):
        h, w = img.shape[:2]
        if target_h is None:
            target_h = h
        if target_w is None:
            target_w = w
        if h == target_h and w == target_w:
            return img

        top = 0
        bottom = max(0, target_h - h)
        left = max(0, (target_w - w) // 2)
        right = max(0, target_w - w - left)
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color,
        )

    @staticmethod
    def _resize_to_width(img, target_w):
        h, w = img.shape[:2]
        if w == target_w:
            return img
        target_h = max(1, int(round(h * (target_w / w))))
        return cv2.resize(img, (target_w, target_h))

    @staticmethod
    def _equalize_row_heights(images):
        target_h = min(img.shape[0] for img in images)
        out = []
        for img in images:
            h, w = img.shape[:2]
            if h == target_h:
                out.append(img)
            else:
                target_w = max(1, int(round(w * (target_h / h))))
                out.append(cv2.resize(img, (target_w, target_h)))
        return out

    @staticmethod
    def _stack_row_with_gaps(images, gap=18, bg_color=(255, 255, 255)):
        images = NN_Trainer._equalize_row_heights(images)
        if len(images) == 1:
            return images[0]
        gap_tile = np.full((images[0].shape[0], gap, 3), bg_color, dtype=np.uint8)
        parts = []
        for i, img in enumerate(images):
            parts.append(img)
            if i != len(images) - 1:
                parts.append(gap_tile)
        return np.hstack(parts)

    @staticmethod
    def _center_row_to_width(images, target_w, gap=18, bg_color=(255, 255, 255)):
        row = NN_Trainer._stack_row_with_gaps(images, gap=gap, bg_color=bg_color)
        if row.shape[1] > target_w:
            row = NN_Trainer._resize_to_width(row, target_w)
        return NN_Trainer._pad_to_size(row, target_w=target_w, bg_color=bg_color)

    @staticmethod
    def _clip_segment_to_uv_box_np(p0, p1, tol=1e-12):
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        if p0.shape != (2,) or p1.shape != (2,) or not np.isfinite([*p0, *p1]).all():
            return None
        delta = p1 - p0
        t_enter, t_exit = 0.0, 1.0
        for p, q in (
            (-delta[0], p0[0]),
            (delta[0], 1.0 - p0[0]),
            (-delta[1], p0[1]),
            (delta[1], 1.0 - p0[1]),
        ):
            if abs(float(p)) <= tol:
                if float(q) < -tol:
                    return None
                continue
            ratio = float(q / p)
            if p < 0.0:
                t_enter = max(t_enter, ratio)
            else:
                t_exit = min(t_exit, ratio)
            if t_enter > t_exit + tol:
                return None
        return (
            np.clip(p0 + t_enter * delta, 0.0, 1.0),
            np.clip(p0 + t_exit * delta, 0.0, 1.0),
        )

    def _render_current_cad_frame_cached(
        self,
        seeds_list,
        decoders,
        pred_list,
        render_cache,
        thr=0.5,
        loading_img=None,
    ):


        pred_by_face_id = {self._face_id_key(p["face_id"]): p for p in pred_list}
        dec_by_face_id = {self._face_id_key(ft.get("face_id", 0)): dec for ft, dec in zip(self.current_face_tensors, decoders)}
        density_meshes = []
        solid_meshes = []
        curve_meshes = []
        fiber_xyz_parts = []
        fiber_vec_parts = []
        fiber_rho_parts = []

        for cache_i in render_cache:
            face_id = cache_i["face_id"]
            pred = pred_by_face_id[face_id]
            decoder = dec_by_face_id[face_id]

            out = self.evaluate_cached_face_fields(cache_i, decoder, pred)

            xyz = out["xyz_dense"].detach().cpu().numpy()
            rho_dense = out["rho_dense"].detach().cpu().numpy()
            fiber_dense = out["fiber3d_dense"].detach().cpu().numpy()
            faces_local = out["faces_ijk"].detach().cpu().numpy().astype(np.int64)
            curve_mesh = self._curves_xyz_to_polydata(out.get("edge_curves_xyz", None))
            if curve_mesh is not None:
                curve_meshes.append(curve_mesh)
            if faces_local.size > 0:
                pv_faces_all = np.empty((faces_local.shape[0], 4), dtype=np.int64)
                pv_faces_all[:, 0] = 3
                pv_faces_all[:, 1:] = faces_local
                mesh_all = pv.PolyData(xyz, pv_faces_all.reshape(-1))
                mesh_all["rho"] = rho_dense.astype(np.float32)
                density_meshes.append(mesh_all)

            if faces_local.size > 0:
                solid_keep = np.mean(rho_dense[faces_local], axis=1) >= float(thr)
                faces_solid_local = faces_local[solid_keep]
                if faces_solid_local.size > 0:
                    pv_faces_solid = np.empty((faces_solid_local.shape[0], 4), dtype=np.int64)
                    pv_faces_solid[:, 0] = 3
                    pv_faces_solid[:, 1:] = faces_solid_local
                    solid_meshes.append(pv.PolyData(xyz, pv_faces_solid.reshape(-1)))

            valid_fiber = np.isfinite(rho_dense)
            valid_fiber &= np.isfinite(fiber_dense).all(axis=1)
            valid_fiber &= (rho_dense >= float(thr))
            valid_fiber &= (np.linalg.norm(fiber_dense, axis=1) > 1e-10)
            if np.any(valid_fiber):
                fiber_xyz_parts.append(xyz[valid_fiber])
                fiber_vec_parts.append(fiber_dense[valid_fiber])
                fiber_rho_parts.append(rho_dense[valid_fiber])

        if density_meshes:
            rho_all = np.concatenate([m["rho"] for m in density_meshes], axis=0)
            rho_clim = [0.0, max(1.0, float(np.quantile(rho_all, 0.995)))]
        else:
            rho_clim = [0.0, 1.0]

        density_mesh_merged = self._concat_polydata(density_meshes, scalar_name="rho")
        solid_mesh_merged = self._concat_polydata(solid_meshes, scalar_name=None)
        curve_mesh_merged = self._concat_line_polydata(curve_meshes) if curve_meshes else None

        all_points = []
        for mesh in density_meshes:
            all_points.append(np.asarray(mesh.points))
        if all_points:
            all_points = np.concatenate(all_points, axis=0)
            diag = float(np.linalg.norm(np.ptp(all_points, axis=0)))
        else:
            diag = 1.0
        arrow_scale = 0.04 * max(diag, 1e-6)

        fiber_points = None
        fiber_vectors = None
        fiber_rho = None
        if fiber_xyz_parts:
            fiber_points = np.concatenate(fiber_xyz_parts, axis=0).astype(np.float32)
            fiber_vectors = np.concatenate(fiber_vec_parts, axis=0).astype(np.float32)
            fiber_rho = np.concatenate(fiber_rho_parts, axis=0).astype(np.float32)
            max_arrows = 600
            if fiber_points.shape[0] > max_arrows:
                stride = int(np.ceil(fiber_points.shape[0] / max_arrows))
                fiber_points = fiber_points[::stride]
                fiber_vectors = fiber_vectors[::stride]
                fiber_rho = fiber_rho[::stride]

        seed_vis = self._seed_points_xyz_and_activity_all_faces(
            seeds_list=seeds_list,
            pred_list=pred_list,
            face_tensors=self.current_face_tensors,
        )
        active_seed_points = seed_vis["xyz_active"]
        inactive_seed_points = seed_vis["xyz_inactive"]
        seed_point_size = max(6.0, 0.006 * max(diag, 1.0) * 100.0)
        show_seed_points = True
        show_axes_widget = True

        first_face_voronoi_img = None
        first_face_graph_img = None
        first_face_core_curves_img = None
        if render_cache:
            first_cache = render_cache[0]
            first_face_id = first_cache["face_id"]
            first_out = self.evaluate_cached_face_fields(
                first_cache,
                dec_by_face_id[first_face_id],
                pred_by_face_id[first_face_id],
            )
            first_seed_idx = 0
            for idx, ft in enumerate(self.current_face_tensors):
                if self._face_id_key(ft.get("face_id", 0)) == first_face_id:
                    first_seed_idx = idx
                    break
            first_face_voronoi_img = self._render_first_face_density_2d(
                cache_i=first_cache,
                out_i=first_out,
                seeds_i=first_out.get("seeds_uv", seeds_list[first_seed_idx]),
                pred_i=pred_by_face_id[first_face_id],
                window_size=(1050, 1050),
                show_scipy_voronoi=True,
                show_core_curves=False,
            )
            first_face_graph_img = self._render_first_face_generated_graph_2d(
                decoder=dec_by_face_id[first_face_id],
                out_i=first_out,
                seeds_i=first_out.get("seeds_uv", seeds_list[first_seed_idx]),
                window_size=(1050, 1050),
                show_node_ids=False,
                show_edge_ids=False,
            )
            first_face_core_curves_img = self._render_first_face_density_2d(
                cache_i=first_cache,
                out_i=first_out,
                seeds_i=first_out.get("seeds_uv", seeds_list[first_seed_idx]),
                pred_i=pred_by_face_id[first_face_id],
                window_size=(1050, 1050),
                show_scipy_voronoi=False,
                show_core_curves=True,
            )

        def make_plotter(title, mode, window_size):
            pl = pv.Plotter(off_screen=True, window_size=window_size)
            pl.set_background("white")
            try:
                pl.disable_anti_aliasing()
            except Exception:
                pass
            try:
                pl.ren_win.SetMultiSamples(0)
            except Exception:
                pass
            pl.remove_all_lights()

            if mode == "density":
                if density_mesh_merged is not None:
                    pl.add_mesh(
                        density_mesh_merged,
                        scalars="rho",
                        cmap="viridis",
                        clim=rho_clim,
                        show_edges=False,
                        lighting=False,
                        smooth_shading=False,
                        nan_color="white",
                        interpolate_before_map=False,
                        scalar_bar_args={
                            "title": "rho",
                            "position_x": 0.28,
                            "position_y": 0.02,
                            "width": 0.64,
                            "height": 0.05,
                            "title_font_size": 12,
                            "label_font_size": 10,
                            "color": "#4b5563",
                            "fmt": "%.2f",
                            "n_labels": 5,
                        },
                    )
                if curve_mesh_merged is not None:
                    pl.add_mesh(curve_mesh_merged, color="black", line_width=3, render_lines_as_tubes=True)
            elif mode == "solid":
                if solid_mesh_merged is not None:
                    pl.add_mesh(
                        solid_mesh_merged,
                        color="#8ecae6",
                        smooth_shading=False,
                        specular=0.0,
                        show_edges=False,
                        lighting=False,
                    )
            elif mode == "fiber":
                if solid_mesh_merged is not None:
                    pl.add_mesh(
                        solid_mesh_merged,
                        color="#dbeafe",
                        opacity=1.0,
                        smooth_shading=False,
                        show_edges=False,
                        lighting=False,
                    )
                if fiber_points is not None and fiber_points.shape[0] > 0:
                    cloud = pv.PolyData(fiber_points)
                    cloud["vectors"] = fiber_vectors
                    cloud["rho"] = fiber_rho
                    glyphs = cloud.glyph(
                        orient="vectors",
                        scale=False,
                        factor=arrow_scale,
                        geom=pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0)),
                    )
                    pl.add_mesh(glyphs, color="#1d4ed8", line_width=2)
                if curve_mesh_merged is not None:
                    pl.add_mesh(curve_mesh_merged, color="black", line_width=3, render_lines_as_tubes=True)

            if show_seed_points and active_seed_points is not None and len(active_seed_points) > 0:
                pl.add_mesh(
                    pv.PolyData(active_seed_points.astype(np.float32)),
                    color="red",
                    render_points_as_spheres=True,
                    point_size=seed_point_size,
                )
            if show_seed_points and inactive_seed_points is not None and len(inactive_seed_points) > 0:
                pl.add_mesh(
                    pv.PolyData(inactive_seed_points.astype(np.float32)),
                    color="gray",
                    opacity=0.35,
                    render_points_as_spheres=True,
                    point_size=max(5.0, 0.8 * seed_point_size),
                )
            if show_axes_widget:
                pl.show_axes()
            return pl

        top_specs = [
            ("3D Heaviside Material | Front View", "solid", "xz"),
            ("3D Heaviside Material | Side View", "solid", "yz"),
            ("3D Heaviside Material | Top View", "solid", "xy"),
        ]
        perspective_spec = ("3D Heaviside Material | Perspective View", "solid", "iso")
        top_window_size = (560, 430)
        bottom_window_size = (1050, 1050)

        top_imgs = []
        if loading_img is not None:
            loading_panel_img = cv2.resize(
                loading_img,
                top_window_size,
                interpolation=cv2.INTER_AREA if loading_img.shape[1] > top_window_size[0] else cv2.INTER_CUBIC,
            )
            top_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        loading_panel_img,
                        "Voxel Loading And Boundary Conditions",
                    )
                )
            )
        for title, mode, view in top_specs:
            pl = make_plotter(title, mode, window_size=top_window_size)
            img = self._render_offscreen_plotter(pl, view)
            top_imgs.append(self._add_panel_border(self._add_image_title(img, title)))
            pl.close()

        bottom_imgs = []
        if first_face_voronoi_img is not None:
            bottom_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        first_face_voronoi_img,
                        "Exact SciPy Voronoi"
                        )
                )
            )
        if first_face_graph_img is not None:
            bottom_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        first_face_graph_img,
                        "Connectivity Graph"
                    )
                )
            )
        if first_face_core_curves_img is not None:
            bottom_imgs.append(
                self._add_panel_border(
                    self._add_image_title(
                        first_face_core_curves_img,
                        "Core Curves UV"
                        )
                )
            )
        title, mode, view = perspective_spec
        pl = make_plotter(title, mode, window_size=bottom_window_size)
        img = self._render_offscreen_plotter(pl, view)
        bottom_imgs.append(self._add_panel_border(self._add_image_title(img, title)))
        pl.close()

        col_gap = 22
        row_gap = 28
        top_row = self._stack_row_with_gaps(top_imgs, gap=col_gap)
        bottom_row = self._center_row_to_width(bottom_imgs, target_w=top_row.shape[1], gap=col_gap)
        gap_tile = np.full((row_gap, top_row.shape[1], 3), 255, dtype=np.uint8)
        cad_panel = np.vstack([top_row, gap_tile, bottom_row])
        cad_panel = cv2.copyMakeBorder(
            cad_panel,
            16,
            16,
            16,
            16,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        return cad_panel

    def _render_current_3d_tube_frame_cached(
        self,
        seeds_list,
        decoders,
        pred_list,
        render_cache,
        loading_img=None,
    ):
        import numpy as np
        import pyvista as pv

        tube_meshes = []
        seed_meshes = []
        first_face_voronoi_img = None
        first_face_graph_img = None
        first_face_core_curves_img = None

        for face_i, (decoder, pred, cache) in enumerate(zip(decoders, pred_list, render_cache)):
            fields = self.evaluate_cached_face_fields(cache, decoder, pred)

            if first_face_voronoi_img is None:
                first_face_voronoi_img = self._render_first_face_density_2d(
                    cache_i=cache,
                    out_i=fields,
                    seeds_i=fields.get("seeds_uv", seeds_list[face_i]),
                    pred_i=pred,
                    window_size=(650, 650),
                    show_scipy_voronoi=True,
                    show_core_curves=False,
                )
                first_face_graph_img = self._render_first_face_generated_graph_2d(
                    decoder=decoder,
                    out_i=fields,
                    seeds_i=fields.get("seeds_uv", seeds_list[face_i]),
                    window_size=(650, 650),
                    show_node_ids=False,
                    show_edge_ids=False,
                )
                first_face_core_curves_img = self._render_first_face_density_2d(
                    cache_i=cache,
                    out_i=fields,
                    seeds_i=fields.get("seeds_uv", seeds_list[face_i]),
                    pred_i=pred,
                    window_size=(650, 650),
                    show_scipy_voronoi=False,
                    show_core_curves=True,
                )

            curves = fields.get("edge_curves_xyz", None)
            if curves is None:
                continue

            if torch.is_tensor(curves):
                curves_np = curves.detach().cpu().numpy()
            else:
                curves_np = np.asarray(curves)

            edge_colors = {
                0: "black",
                1: "orange",
                2: "gray",
                3: "orange",
                4: "cyan",
            }
            edge_types_np = None
            graph = fields.get("graph", None)
            if isinstance(graph, dict):
                edge_types = graph.get("edge_type", None)
                if torch.is_tensor(edge_types):
                    edge_types_np = edge_types.detach().cpu().numpy()
                elif edge_types is not None:
                    edge_types_np = np.asarray(edge_types)

            radius = pred.get("centerline_radius", None)
            if radius is None:
                radius = 0.01
            elif torch.is_tensor(radius):
                radius = float(radius.detach().mean().cpu().item())
            else:
                radius = float(radius)

            radius = max(
                radius * float(getattr(self.cfg, "timelapse_tube_radius_scale", 1.0)),
                1e-4,
            )

            for edge_id, points in enumerate(curves_np):
                if points.shape[0] < 2 or not np.isfinite(points).all():
                    continue

                polyline = pv.PolyData(points.astype(np.float32))
                polyline.lines = np.concatenate(
                    ([len(points)], np.arange(len(points)))
                ).astype(np.int64)

                tube = polyline.tube(
                    radius=radius,
                    n_sides=int(getattr(self.cfg, "timelapse_tube_n_sides", 12)),
                )
                if tube.n_points > 0:
                    if edge_types_np is not None and edge_id < len(edge_types_np):
                        color = edge_colors.get(int(edge_types_np[edge_id]), "gray")
                    else:
                        color = "orange"
                    tube_meshes.append((tube, color))

            seeds_xyz = fields.get("seeds_xyz", None)
            if seeds_xyz is not None:
                if torch.is_tensor(seeds_xyz):
                    seeds_xyz = seeds_xyz.detach().cpu().numpy()
                seeds_xyz = np.asarray(seeds_xyz, dtype=np.float32)
                if seeds_xyz.ndim == 2 and seeds_xyz.shape[0] > 0 and seeds_xyz.shape[1] == 3:
                    finite_seed = np.isfinite(seeds_xyz).all(axis=1)
                    if np.any(finite_seed):
                        seed_mesh = pv.PolyData(seeds_xyz[finite_seed])
                        if seed_mesh.n_points > 0:
                            seed_meshes.append(seed_mesh)

        plotter = pv.Plotter(off_screen=True, window_size=(900, 650))
        plotter.set_background("white")

        for mesh, color in tube_meshes:
            if mesh.n_points > 0:
                plotter.add_mesh(mesh, color=color, smooth_shading=True)

        for sm in seed_meshes:
            if sm.n_points > 0:
                plotter.add_mesh(
                    sm,
                    color="red",
                    point_size=8,
                    render_points_as_spheres=True,
                )

        if not tube_meshes and not seed_meshes:
            plotter.add_text("No 3D tube curves", color="black", font_size=14)

        plotter.view_isometric()
        plotter.reset_camera()
        img = plotter.screenshot(return_img=True)
        plotter.close()

        tube_img = self._add_panel_border(
            self._add_image_title(
                self._composite_to_white(img),
                "3D Voronoi Tube Curves",
            )
        )

        if first_face_voronoi_img is None or first_face_graph_img is None or first_face_core_curves_img is None:
            return tube_img

        voronoi_img = self._add_panel_border(
            self._add_image_title(
                first_face_voronoi_img,
                "Exact SciPy Voronoi",
            )
        )
        graph_img = self._add_panel_border(
            self._add_image_title(
                first_face_graph_img,
                "Connectivity Graph",
            )
        )
        core_curves_img = self._add_panel_border(
            self._add_image_title(
                first_face_core_curves_img,
                "Core Curves UV",
            )
        )
        frame = self._stack_row_with_gaps([voronoi_img, graph_img, core_curves_img, tube_img], gap=22)
        return cv2.copyMakeBorder(
            frame,
            16,
            16,
            16,
            16,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def _render_first_face_density_2d(
        self,
        cache_i,
        out_i,
        seeds_i,
        pred_i,
        window_size=(820, 820),
        show_scipy_voronoi=True,
        show_core_curves=True,
    ):
        width, height = int(window_size[0]), int(window_size[1])
        uv = cache_i["uv_dense"].detach().cpu().numpy().astype(np.float64)
        faces = cache_i["faces_ijk"].detach().cpu().numpy().astype(np.int64)
        seeds = seeds_i.detach().cpu().numpy().astype(np.float64)
        topology_seeds_t = out_i.get("topology_seeds_uv", None)
        topology_seeds = None
        if isinstance(topology_seeds_t, torch.Tensor):
            topology_seeds = topology_seeds_t.detach().cpu().numpy().astype(np.float64)
        elif topology_seeds_t is not None:
            topology_seeds = np.asarray(topology_seeds_t, dtype=np.float64)
        curves_uv_t = out_i.get("edge_curves_uv", None)
        curves_uv = None
        if isinstance(curves_uv_t, torch.Tensor):
            curves_uv = curves_uv_t.detach().cpu().numpy().astype(np.float64)

        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="white")
        ax = fig.add_axes([0.08, 0.08, 0.88, 0.84])

        if faces.size > 0:
            ax.triplot(
                uv[:, 0],
                uv[:, 1],
                faces,
                color="#d1d5db",
                linewidth=0.35,
                alpha=0.5,
            )
        else:
            ax.scatter(
                uv[:, 0],
                uv[:, 1],
                c="#d1d5db",
                s=5,
                alpha=0.45,
                linewidths=0,
            )

        def plot_clipped_segment(p0, p1, color, linewidth, alpha, zorder):
            clipped = self._clip_segment_to_uv_box_np(p0, p1)
            if clipped is None:
                return
            q0, q1 = clipped
            ax.plot(
                [q0[0], q1[0]],
                [q0[1], q1[1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

        if show_scipy_voronoi and topology_seeds is not None and topology_seeds.ndim == 2 and topology_seeds.shape[0] >= 3:
            finite_topology_seeds = topology_seeds[np.isfinite(topology_seeds).all(axis=1)]
            if finite_topology_seeds.shape[0] >= 3:
                try:
                    from scipy.spatial import Voronoi, voronoi_plot_2d
                    raw_voronoi = Voronoi(finite_topology_seeds)
                    voronoi_plot_2d(
                        raw_voronoi,
                        ax=ax,
                        show_vertices=False,
                        show_points=False,
                        line_colors="#2563eb",
                        line_width=1.05,
                        line_alpha=0.58,
                        point_size=0,
                    )
                    if raw_voronoi.vertices.size > 0:
                        ax.scatter(
                            raw_voronoi.vertices[:, 0],
                            raw_voronoi.vertices[:, 1],
                            marker="x",
                            c="#374151",
                            s=38,
                            linewidths=1.0,
                            alpha=0.82,
                            zorder=4,
                        )
                except Exception:
                    pass

        active_values = pred_i.get("seed_active_mask", None)
        if active_values is not None:
            active = active_values.detach().cpu().numpy().reshape(-1).astype(bool)
        else:
            active = np.ones((seeds.shape[0],), dtype=bool)
        weight_values = pred_i.get("seed_active_weights", None)
        if weight_values is not None:
            weights = weight_values.detach().cpu().numpy().reshape(-1)
            active = active & (weights >= 0.5)

        if seeds.shape[0] > 0:
            if np.any(~active):
                ax.scatter(
                    seeds[~active, 0],
                    seeds[~active, 1],
                    s=72,
                    c="#6b7280",
                    edgecolors="white",
                    linewidths=1.4,
                    alpha=0.55,
                    zorder=5,
                )
            if np.any(active):
                ax.scatter(
                    seeds[active, 0],
                    seeds[active, 1],
                    s=92,
                    c="#ef4444",
                    edgecolors="white",
                    linewidths=1.6,
                    zorder=6,
                )

        if show_core_curves and curves_uv is not None and curves_uv.ndim == 3:
            for curve in curves_uv:
                if curve.shape[0] >= 2:
                    ax.plot(
                        curve[:, 0],
                        curve[:, 1],
                        color="black",
                        linewidth=1.8,
                        alpha=0.95,
                        zorder=4,
                    )

        ax.set_xlim(float(np.nanmin(uv[:, 0])), float(np.nanmax(uv[:, 0])))
        ax.set_ylim(float(np.nanmin(uv[:, 1])), float(np.nanmax(uv[:, 1])))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("u", fontsize=11)
        ax.set_ylabel("v", fontsize=11)
        ax.tick_params(labelsize=9, colors="#374151")
        ax.grid(color="#e5e7eb", linewidth=0.7, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3].copy()
        plt.close(fig)
        return img

    def _render_first_face_generated_graph_2d(
        self,
        decoder,
        out_i,
        seeds_i,
        window_size=(820, 820),
        show_node_ids=False,
        show_edge_ids=False,
    ):
        width, height = int(window_size[0]), int(window_size[1])
        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="white")
        ax = fig.add_axes([0.08, 0.08, 0.88, 0.84])
        topology_seeds = out_i.get("topology_seeds_uv", seeds_i)
        try:
            decoder._draw_generated_graph(
                ax,
                topology_seeds,
                out_i,
                show_node_ids=show_node_ids,
                show_edge_ids=show_edge_ids,
                node_id_fontsize=7,
                show_pruned_nodes=False,
                color_by_edge_type=True,
            )
            ax.set_title("Generated Connectivity Graph", fontsize=12)
        except Exception as error:
            ax.text(
                0.5,
                0.5,
                f"Generated graph unavailable\n{error}",
                ha="center",
                va="center",
                fontsize=10,
                color="#111827",
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                fontsize=6,
                loc="upper right",
                framealpha=0.78,
            )
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3].copy()
        plt.close(fig)
        return img

    def _seed_points_xyz_and_activity_all_faces(self, seeds_list, pred_list, face_tensors):
        xyz_active = []
        xyz_inactive = []
        active_weight = []
        inactive_weight = []

        for seeds, pred, ft in zip(seeds_list, pred_list, face_tensors):
            if isinstance(pred.get("seeds_xyz"), torch.Tensor):
                xyz_i = pred["seeds_xyz"].detach().cpu().numpy()
            else:
                xyz_i = self.generator.seeds_uv_to_xyz_nearest(
                    seeds,
                    ft["uv"],
                    ft["points_xyz"],
                )

            active_mask_i = pred.get("seed_active_mask", None)
            active_weights_i = pred.get("seed_active_weights", None)

            if active_mask_i is None:
                xyz_active.append(xyz_i)
                continue

            active_mask = active_mask_i.detach().cpu().numpy().astype(bool)
            weights = (
                active_weights_i.detach().cpu().numpy()
                if active_weights_i is not None
                else active_mask.astype(float)
            )
            participating_mask = active_mask & (weights >= 0.5)
            inactive_mask = ~participating_mask

            xyz_i_active = xyz_i[participating_mask]
            xyz_i_inactive = xyz_i[inactive_mask]

            if len(xyz_i_active) > 0:
                xyz_active.append(xyz_i_active)
                active_weight.append(weights[participating_mask])

            if len(xyz_i_inactive) > 0:
                xyz_inactive.append(xyz_i_inactive)
                inactive_weight.append(weights[inactive_mask])

        import numpy as np

        xyz_active = np.concatenate(xyz_active, axis=0) if len(xyz_active) > 0 else None
        xyz_inactive = np.concatenate(xyz_inactive, axis=0) if len(xyz_inactive) > 0 else None
        active_weight = np.concatenate(active_weight, axis=0) if len(active_weight) > 0 else None
        inactive_weight = np.concatenate(inactive_weight, axis=0) if len(inactive_weight) > 0 else None

        return {
            "xyz_active": xyz_active,
            "xyz_inactive": xyz_inactive,
            "active_weight": active_weight,
            "inactive_weight": inactive_weight,
        }
    
    def visualize_best_seed_activity(self, result, points_xyz=None, faces_ijk=None):
        best_seeds = result["best_seeds"]
        best_pred = result["best_pred"]
        face_tensors = result["face_tensors"]

        seed_vis = self._seed_points_xyz_and_activity_all_faces(
            seeds_list=best_seeds,
            pred_list=best_pred,
            face_tensors=face_tensors,
        )

        plotter = pv.Plotter()

        if points_xyz is not None and faces_ijk is not None:
            pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)
            mesh = pv.PolyData(points_xyz.detach().cpu().numpy(), pv_faces_fixed)
            plotter.add_mesh(mesh, color="white", opacity=0.25, show_edges=False)

        if seed_vis["xyz_active"] is not None and len(seed_vis["xyz_active"]) > 0:
            active_cloud = pv.PolyData(seed_vis["xyz_active"])
            plotter.add_mesh(
                active_cloud,
                color="red",
                render_points_as_spheres=True,
                point_size=14,
                label="Active seeds",
            )

        if seed_vis["xyz_inactive"] is not None and len(seed_vis["xyz_inactive"]) > 0:
            inactive_cloud = pv.PolyData(seed_vis["xyz_inactive"])
            plotter.add_mesh(
                inactive_cloud,
                color="gray",
                render_points_as_spheres=True,
                point_size=10,
                opacity=0.4,
                label="Inactive seeds",
            )

        plotter.add_legend()
        plotter.show()
    
    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _get_result_final_density(self, result):
        density = result.get("Final_shape_density", None)
        if density is None:
            density = result.get("best_rho", None)
        if density is None:
            available = ", ".join(sorted(str(k) for k in result.keys()))
            raise KeyError(
                "Could not find final density in result. Expected "
                "'Final_shape_density' or legacy fallback 'best_rho'. "
                f"Available keys: {available}"
            )
        return density

    def visualize_result_stepwise(self, result, points_xyz, faces_ijk):
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        density_init = result["Initial_shape_density"].detach().cpu().numpy()
        density_mid = result["Mid_shape_density"].detach().cpu().numpy()
        density_final = self._get_result_final_density(result).detach().cpu().numpy()

        self.viz.plot_density_and_seedpoints_3stage_2(
            mesh_points=points_xyz.detach().cpu().numpy(),
            pv_faces=pv_faces_fixed,
            density_init=density_init,
            density_mid=density_mid,
            density_final=density_final,
            seed_points_init=result["seed_points_init"],
            seed_points_mid=result["seed_points_mid"],
            seed_points_final=result["seed_points_final"],
        )

    def visualize_result_final(
        self,
        result,
        points_xyz,
        faces_ijk,
        thr=0.5,
        show_solid=True,
        show_rho_overlay=False,
    ):
        points_np = (
            points_xyz.detach().cpu().numpy()
            if isinstance(points_xyz, torch.Tensor)
            else np.asarray(points_xyz)
        )
        faces_np = (
            faces_ijk.detach().cpu().numpy()
            if isinstance(faces_ijk, torch.Tensor)
            else np.asarray(faces_ijk)
        ).astype(np.int64)
        rho_np = self._get_result_final_density(result).detach().cpu().numpy()

        pv_faces = np.empty((faces_np.shape[0], 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = faces_np
        mesh = pv.PolyData(points_np, pv_faces.reshape(-1))
        mesh["rho"] = rho_np.astype(np.float32)

        face_rho = np.mean(rho_np[faces_np], axis=1) if faces_np.size > 0 else np.empty((0,))
        solid_faces = faces_np[face_rho >= float(thr)]
        if solid_faces.size > 0:
            pv_solid_faces = np.empty((solid_faces.shape[0], 4), dtype=np.int64)
            pv_solid_faces[:, 0] = 3
            pv_solid_faces[:, 1:] = solid_faces
            solid = pv.PolyData(points_np, pv_solid_faces.reshape(-1))
        else:
            solid = pv.PolyData()

        if show_solid:
            plotter = pv.Plotter()
            plotter.set_background("white")
            if solid.n_cells > 0:
                plotter.add_mesh(solid, color="#8ecae6", show_edges=False, lighting=False)
            seed_xyz_parts = []
            for pred in result.get("best_pred", []):
                seeds_xyz_i = pred.get("seeds_xyz", None) if isinstance(pred, dict) else None
                if isinstance(seeds_xyz_i, torch.Tensor):
                    seed_xyz_parts.append(seeds_xyz_i.detach().cpu().numpy())
            if seed_xyz_parts:
                seed_xyz = np.concatenate(seed_xyz_parts, axis=0).astype(np.float32)
                plotter.add_mesh(
                    pv.PolyData(seed_xyz),
                    color="red",
                    render_points_as_spheres=True,
                    point_size=12,
                )
            if show_rho_overlay:
                plotter.add_mesh(
                    mesh,
                    scalars="rho",
                    cmap="viridis",
                    opacity=0.20,
                    show_edges=False,
                    lighting=False,
                )
            plotter.show_axes()
            plotter.show()

        return solid, float(thr)
    def sample_face_field_for_visualization(
        self,
        ft: dict,
        decoder,
        pred: dict,
        shape_or_path,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        use_boundary_attachment: bool = True,
        trim_tol: float = 1e-7,
    ):
        """
        Dense CAD-native field sampling on one face for smooth visualization.

        This version:
        - builds a dense UV grid in normalized face UV
        - optionally prefilters points by proximity to sampled UV cloud
        - evaluates xyz, Xu, Xv on the actual CAD face
        - keeps only trim-valid points
        - evaluates decoder on those dense query points

        Returns:
            {
                "uv_dense": (Nd,2),
                "uv_raw_dense": (Nd,2),
                "xyz_dense": (Nd,3),
                "Xu_dense": (Nd,3),
                "Xv_dense": (Nd,3),
                "rho_dense": (Nd,),
                "rho_v_dense": (Nd,),
                "rho_b_dense": (Nd,),
                "fiber3d_dense": (Nd,3),
                "edge_field_dense": (Nd,),
                "mask_dense_prefilter": (Nu*Nv,),
                "grid_shape": (Nu, Nv),
            }
        """
        device = ft["uv"].device
        dtype = ft["uv"].dtype

        uv_face = ft["uv"]
        u_periodic = bool(ft.get("u_periodic", False))
        v_periodic = bool(ft.get("v_periodic", False))

        # ------------------------------------------------------------
        # 1) Dense UV grid in normalized face UV coordinates
        # ------------------------------------------------------------
        uv_grid, _u_lin, _v_lin = self._build_face_uv_grid(ft, grid_res_u, grid_res_v)

        # ------------------------------------------------------------
        # 2) Optional UV-cloud prefilter
        #    Helps avoid querying huge empty regions on trimmed faces.
        # ------------------------------------------------------------
        if uv_mask_tol is None:
            uv_mask_tol = self._estimate_uv_mask_tol(
                uv_face=uv_face,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )

        dmin = self._periodic_uv_min_dist(
            uv_grid,
            uv_face,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        mask_dense_prefilter = dmin <= uv_mask_tol
        uv_query = uv_grid[mask_dense_prefilter]

        if uv_query.numel() == 0:
            raise ValueError(
                f"No dense UV query points survived prefilter on face {ft.get('face_id', 'unknown')}. "
                f"Try increasing uv_mask_tol."
            )

        # ------------------------------------------------------------
        # 3) CAD-native geometry evaluation
        # ------------------------------------------------------------
        geom = self.generator.eval_face_uv_from_face_tensor(
            shape_or_path=shape_or_path,
            face_tensor=ft,
            uv_norm=uv_query,
            metric_tol=getattr(self.generator, "metric_tol", 1e-9),
            trim_tol=trim_tol,
            as_torch=True,
        )

        valid_mask = geom["valid_mask"]
        if valid_mask.numel() == 0 or not bool(valid_mask.any().item()):
            raise ValueError(
                f"No valid CAD-evaluable dense points on face {ft.get('face_id', 'unknown')}."
            )

        uv_dense = geom["uv_norm"][valid_mask]
        uv_raw_dense = geom["uv_raw"][valid_mask]
        xyz_dense = geom["points_xyz"][valid_mask]
        Xu_dense = geom["Xu"][valid_mask]
        Xv_dense = geom["Xv"][valid_mask]
        mask_dense_valid = torch.zeros_like(mask_dense_prefilter, dtype=torch.bool)
        mask_dense_valid[mask_dense_prefilter] = valid_mask

        # ------------------------------------------------------------
        # 4) Boundary data for decoder
        # ------------------------------------------------------------
        local_face_id = torch.zeros(
            uv_dense.shape[0],
            dtype=torch.long,
            device=device,
        )

        boundary_uv_i = None
        boundary_face_id_i = None
        boundary_loop_id_i = None

        if use_boundary_attachment:
            true_bidx_i, boundary_loop_id_i = self._ordered_true_open_boundary(ft)
            if true_bidx_i.numel() > 0:
                boundary_uv_i = uv_face[true_bidx_i]
                boundary_face_id_i = torch.zeros(
                    boundary_uv_i.shape[0],
                    dtype=torch.long,
                    device=device,
                )

        # ------------------------------------------------------------
        # 5) Recover trained parameters
        # ------------------------------------------------------------
        seeds_raw = pred["seeds_raw"]
        w_raw = pred["w_raw"]
        h_raw = pred.get("h_raw", None)

        theta = pred.get("theta", None)
        a_raw = pred.get("a_raw", None)

        boundary_width_raw = pred.get("boundary_width_raw", None)
        boundary_alpha_raw = pred.get("boundary_alpha_raw", None)
        boundary_beta_raw = pred.get("boundary_beta_raw", None)

        # ------------------------------------------------------------
        # 6) Evaluate decoder on CAD-native dense query points
        # ------------------------------------------------------------
        decoder_out = decoder.build_swept_tube_fields(
            points_uv=uv_dense,
            points_3d=xyz_dense,
            seeds_uv=seeds_raw,
            w_raw=w_raw,
            Xu=Xu_dense,
            Xv=Xv_dense,
            cad_domain=self.Cad_domain,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
            return_xyz=True,
            generate_density_fiber=getattr(self.cfg, "generate_decoder_density_fiber", True),
        )

        self._require_decoder_keys(
            decoder_out,
            ["rho", "fiber3d"],
        )

        full_indices = -torch.ones(
            mask_dense_valid.shape[0],
            dtype=torch.long,
            device=device,
        )
        full_indices[mask_dense_valid] = torch.arange(
            int(mask_dense_valid.sum().item()),
            dtype=torch.long,
            device=device,
        )
        faces_dense = []
        for i in range(grid_res_u - 1):
            for j in range(grid_res_v - 1):
                k00 = i * grid_res_v + j
                k01 = i * grid_res_v + j + 1
                k10 = (i + 1) * grid_res_v + j
                k11 = (i + 1) * grid_res_v + j + 1
                ids = full_indices[torch.tensor([k00, k01, k10, k11], device=device)]
                if bool((ids >= 0).all().item()):
                    faces_dense.append(ids[[0, 1, 2]])
                    faces_dense.append(ids[[2, 1, 3]])
        if faces_dense:
            faces_dense = torch.stack(faces_dense, dim=0)
        else:
            faces_dense = torch.empty((0, 3), dtype=torch.long, device=device)

        dense_face_tensor = {
            "points_xyz": xyz_dense,
            "faces_ijk": faces_dense,
            "Xu": Xu_dense,
            "Xv": Xv_dense,
        }
        decoder_out = apply_density_postprocess_to_output(
            decoder_out,
            dense_face_tensor,
            self.cfg,
            return_debug=False,
        )

        return {
            "face_id": self._face_id_key(ft.get("face_id", 0)),
            "uv_dense": uv_dense,
            "uv_raw_dense": uv_raw_dense,
            "xyz_dense": xyz_dense,
            "Xu_dense": Xu_dense,
            "Xv_dense": Xv_dense,
            "rho_dense": decoder_out["rho"],
            "rho_raw_decoder_dense": decoder_out["rho_raw_decoder"],
            "rho_postprocessed_dense": decoder_out["rho_postprocessed"],
            "fiber3d_dense": decoder_out["fiber3d"],
            "seeds_uv": decoder_out.get("seeds_uv", decoder_out.get("seeds", None)),
            "seeds_xyz": decoder_out.get("seeds_xyz", None),
            "edge_curves_uv": decoder_out.get("edge_curves_uv", None),
            "edge_curves_xyz": decoder_out.get("edge_curves_xyz", None),
            "mask_dense_prefilter": mask_dense_prefilter,
            "mask_dense_valid": mask_dense_valid,
            "grid_shape": (grid_res_u, grid_res_v),
        }
   
    def sample_result_field_dense_for_visualization(
        self,
        result: dict,
        shape_or_path=None,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        use_best_pred: bool = True,
    ):
        """
        Dense CAD-native field sampling over all faces for smooth visualization.
        """
        face_tensors = result["face_tensors"]
        decoders = result["decoders"]

        if use_best_pred:
            pred_list = result["best_pred"]
        else:
            raise ValueError("Only use_best_pred=True is currently supported.")

        if shape_or_path is None:
            shape_or_path = result.get("shape_path", None)

        if shape_or_path is None:
            raise ValueError(
                "shape_or_path is required for CAD-native dense sampling. "
                "Pass it explicitly or store 'shape_path' in result."
            )

        pred_by_face_id = {self._face_id_key(p["face_id"]): p for p in pred_list}

        xyz_parts = []
        rho_parts = []
        fiber_parts = []
        face_ranges = []
        per_face = []

        start = 0
        for ft, decoder in zip(face_tensors, decoders):
            face_id = self._face_id_key(ft.get("face_id", 0))
            if face_id not in pred_by_face_id:
                raise KeyError(f"Missing best_pred for face_id={face_id}")

            pred = pred_by_face_id[face_id]

            sampled = self.sample_face_field_for_visualization(
                ft=ft,
                decoder=decoder,
                pred=pred,
                shape_or_path=shape_or_path,
                grid_res_u=grid_res_u,
                grid_res_v=grid_res_v,
                uv_mask_tol=uv_mask_tol,
            )

            n = sampled["xyz_dense"].shape[0]
            end = start + n

            xyz_parts.append(sampled["xyz_dense"])
            rho_parts.append(sampled["rho_dense"])
            fiber_parts.append(sampled["fiber3d_dense"])

            face_ranges.append((start, end, face_id))
            per_face.append(sampled)
            start = end

        return {
            "points_xyz": torch.cat(xyz_parts, dim=0),
            "rho": torch.cat(rho_parts, dim=0),
            "fiber3d": torch.cat(fiber_parts, dim=0),
            "face_ranges": face_ranges,
            "per_face": per_face,
        }

    @staticmethod
    def _resolve_visualization_grid_resolution(
        grid_res_u: int,
        grid_res_v: int,
        dense_factor: float = 1.0,
        min_res: int = 8,
        max_res: int = 1024,
    ) -> tuple[int, int]:
        dense_factor = float(max(dense_factor, 1e-3))
        res_u = int(round(float(grid_res_u) * dense_factor))
        res_v = int(round(float(grid_res_v) * dense_factor))
        res_u = max(int(min_res), min(int(max_res), res_u))
        res_v = max(int(min_res), min(int(max_res), res_v))
        return res_u, res_v

    @staticmethod
    def _dense_face_triangles(mask_dense_valid, grid_shape):
        mask = np.asarray(mask_dense_valid, dtype=bool).reshape(-1)
        Nu, Nv = (int(grid_shape[0]), int(grid_shape[1]))
        full_indices = -np.ones(mask.shape[0], dtype=np.int64)
        full_indices[mask] = np.arange(np.count_nonzero(mask), dtype=np.int64)
        triangles = []

        def idx(i, j):
            return i * Nv + j

        for i in range(Nu - 1):
            for j in range(Nv - 1):
                ids = [idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)]
                mapped = [full_indices[k] for k in ids]
                if any(m < 0 for m in mapped):
                    continue
                i0, i1, i2, i3 = mapped
                triangles.append([i0, i1, i2])
                triangles.append([i2, i1, i3])

        return np.asarray(triangles, dtype=np.int64)

    def visualize_result_final_edge_field(
        self,
        result,
        shape_or_path=None,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
        cmap: str = "viridis",
        show_seeds: bool = True,
        show_uv: bool = True,
        show_3d: bool = True,
    ):
        """Plot the decoder's geometric Voronoi edge field in UV and on the CAD surface."""
        if not show_uv and not show_3d:
            raise ValueError("At least one of show_uv or show_3d must be True.")

        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )
        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        pred_by_face_id = {self._face_id_key(p["face_id"]): p for p in result["best_pred"]}
        face_plots = []
        for face_data in dense["per_face"]:
            mask = face_data["mask_dense_valid"].detach().cpu().numpy()
            triangles = self._dense_face_triangles(mask, face_data["grid_shape"])
            face_plots.append(
                {
                    "face_id": face_data["face_id"],
                    "uv": face_data["uv_dense"].detach().cpu().numpy().astype(np.float32),
                    "xyz": face_data["xyz_dense"].detach().cpu().numpy().astype(np.float32),
                    "edge_field": face_data["edge_field_dense"].detach().cpu().numpy().astype(np.float32),
                    "triangles": triangles,
                }
            )

        uv_fig = None
        if show_uv:
            n_faces = len(face_plots)
            ncols = min(3, max(1, n_faces))
            nrows = int(np.ceil(float(n_faces) / float(ncols)))
            uv_fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5.6 * ncols, 5.0 * nrows),
                squeeze=False,
                constrained_layout=True,
            )
            color_artist = None
            for ax, face_plot in zip(axes.ravel(), face_plots):
                uv = face_plot["uv"]
                triangles = face_plot["triangles"]
                edge_field = face_plot["edge_field"]
                if triangles.size > 0:
                    color_artist = ax.tripcolor(
                        uv[:, 0],
                        uv[:, 1],
                        triangles,
                        edge_field,
                        shading="gouraud",
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                    )
                else:
                    color_artist = ax.scatter(
                        uv[:, 0],
                        uv[:, 1],
                        c=edge_field,
                        s=6,
                        linewidths=0,
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                    )

                if show_seeds:
                    pred = pred_by_face_id.get(face_plot["face_id"])
                    if pred is not None:
                        seeds_uv = pred["seeds_raw"].detach().cpu().numpy()
                        ax.scatter(
                            seeds_uv[:, 0],
                            seeds_uv[:, 1],
                            s=38,
                            c="#e04b3f",
                            edgecolors="white",
                            linewidths=1.0,
                            zorder=3,
                        )
                ax.set_title(f"Face {face_plot['face_id']} | Edge Field")
                ax.set_xlabel("u")
                ax.set_ylabel("v")
                ax.set_aspect("equal", adjustable="box")

            for ax in axes.ravel()[len(face_plots):]:
                ax.axis("off")
            if color_artist is not None:
                uv_fig.colorbar(color_artist, ax=axes.ravel().tolist(), label="edge_field")
            uv_fig.suptitle("Geometric Voronoi Edge Field in UV", y=1.02)
            plt.show()

        plotter = None
        if show_3d:
            plotter = pv.Plotter()
            for face_plot in face_plots:
                triangles = face_plot["triangles"]
                if triangles.size == 0:
                    continue
                pv_faces = np.empty((triangles.shape[0], 4), dtype=np.int64)
                pv_faces[:, 0] = 3
                pv_faces[:, 1:] = triangles
                mesh = pv.PolyData(face_plot["xyz"], pv_faces.reshape(-1))
                mesh["edge_field"] = face_plot["edge_field"]
                plotter.add_mesh(
                    mesh,
                    scalars="edge_field",
                    cmap=cmap,
                    clim=[0.0, 1.0],
                    show_edges=False,
                    scalar_bar_args={"title": "edge_field"},
                )

            seed_points_final = result.get("seed_points_final")
            if show_seeds and seed_points_final is not None:
                plotter.add_mesh(
                    seed_points_final,
                    render_points_as_spheres=True,
                    point_size=6,
                    color="#e04b3f",
                )
            plotter.add_text("Geometric Voronoi Edge Field", font_size=11)
            plotter.show_axes()
            plotter.show()

        return {
            "uv_fig": uv_fig,
            "plotter": plotter,
            "dense": dense,
            "per_face": face_plots,
            "grid_shape": (grid_res_u, grid_res_v),
        }

    def visualize_result_final_smooth_points(
        self,
        result,
        shape_or_path=None,
        thr: float = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
    ):
        """
        Smooth point-cloud style threshold visualization from dense CAD-native decoder sampling.

        `dense_factor` scales the internal UV sampling density used for visualization.
        Larger values produce a denser point cloud and finer visual detail.
        """
        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        points_xyz = dense["points_xyz"].detach().cpu().numpy()
        rho = dense["rho"].detach().cpu().numpy()

        keep = rho >= thr
        solid_points = points_xyz[keep]

        print(
            f"Smooth CAD-native visualization: kept {keep.sum()} / {keep.shape[0]} dense points "
            f"with threshold {thr:.3f} on grid ({grid_res_u} x {grid_res_v})"
        )


        cloud = pv.PolyData(solid_points)

        plotter = pv.Plotter()
        plotter.add_points(
            cloud,
            render_points_as_spheres=True,
            point_size=6,
        )

        plotter.show()

        return {
            "solid_points": solid_points,
            "points_xyz": points_xyz,
            "rho": rho,
            "keep_mask": keep,
            "dense": dense,
        }

    def Visualize_fresult_final_fiber_Direction(
        self,
        result,
        points_xyz,
        faces_ijk,
        thr: float = 0.5,
    ):
        import numpy as np
        import pyvista as pv

        if result.get("Final_shape_fiber_direction", None) is None:
            raise ValueError(
                "result['Final_shape_fiber_direction'] is missing. "
                "Run training with the updated trainer result output."
            )

        density = self._get_result_final_density(result).detach().cpu()
        fiber = result["Final_shape_fiber_direction"].detach().cpu()
        points_xyz_cpu = points_xyz.detach().cpu()
        pv_faces_fixed = self.generator.faces_ijk_to_pv_faces(faces_ijk)

        keep = torch.isfinite(density)
        keep = keep & torch.isfinite(fiber).all(dim=1)
        keep = keep & (density >= float(thr))
        keep = keep & (torch.linalg.norm(fiber, dim=1) > 1e-10)

        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if keep_idx.numel() == 0:
            print(f"No fiber arrows to display for threshold {thr:.3f}.")
            return {
                "points_xyz": points_xyz_cpu.numpy(),
                "rho": density.numpy(),
                "fiber3d": fiber.numpy(),
                "keep_mask": keep.numpy(),
            }

        max_arrows = 2000
        if keep_idx.numel() > max_arrows:
            step = int(np.ceil(float(keep_idx.numel()) / float(max_arrows)))
            keep_idx = keep_idx[::step]

        pts_np = points_xyz_cpu[keep_idx].numpy().astype(np.float32)
        fiber_np = fiber[keep_idx].numpy().astype(np.float32)
        rho_np = density[keep_idx].numpy().astype(np.float32)

        bbox = points_xyz_cpu.amax(dim=0) - points_xyz_cpu.amin(dim=0)
        diag = float(torch.linalg.norm(bbox).item())
        arrow_scale_used = 0.03 * diag

        plotter = pv.Plotter()

        surface = pv.PolyData(
            points_xyz_cpu.numpy().astype(np.float32),
            pv_faces_fixed,
        )
        surface["rho"] = density.numpy().astype(np.float32)
        plotter.add_mesh(
            surface,
            scalars="rho",
            cmap="Greys",
            opacity=0.20,
            show_edges=False,
        )

        arrow_cloud = pv.PolyData(pts_np)
        arrow_cloud["vectors"] = fiber_np
        arrow_cloud["rho"] = rho_np

        glyphs = arrow_cloud.glyph(
            orient="vectors",
            scale=False,
            factor=arrow_scale_used,
            geom=pv.Arrow(),
        )
        plotter.add_mesh(glyphs, color="royalblue")
        plotter.show_axes()
        plotter.show()

        print(
            f"Fiber-direction visualization: showing {pts_np.shape[0]} arrows "
            f"with threshold {thr:.3f}"
        )

        return {
            "arrow_points": pts_np,
            "arrow_vectors": fiber_np,
            "arrow_rho": rho_np,
            "points_xyz": points_xyz_cpu.numpy(),
            "rho": density.numpy(),
            "fiber3d": fiber.numpy(),
            "keep_mask": keep.numpy(),
            "arrow_scale_used": float(arrow_scale_used),
        }

    def visualize_result_final_fiber_direction(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def Visualize_fresult_final_fiber_Direction_3D(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def visualize_result_final_fiber_direction_3d(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction(*args, **kwargs)

    def Visualize_fresult_final_fiber_Direction_2D(
        self,
        result,
        points_xyz=None,
        faces_ijk=None,
        shape_or_path=None,
        thr: float = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        dense_factor: float = 1.0,
        max_arrows_per_face: int = 1200,
        arrow_scale: float = 28.0,
        arrow_width: float = 0.0025,
        cmap: str = "viridis",
        show_boundary: bool = True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        if result.get("Final_shape_fiber_direction", None) is None:
            raise ValueError(
                "result['Final_shape_fiber_direction'] is missing. "
                "Run training with the updated trainer result output."
            )

        density_global = self._get_result_final_density(result).detach().cpu()
        fiber_global = result["Final_shape_fiber_direction"].detach().cpu()
        face_tensors = result["face_tensors"]

        n_faces = len(face_tensors)
        ncols = min(3, max(1, n_faces))
        nrows = int(np.ceil(n_faces / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.5 * ncols, 5.0 * nrows),
            squeeze=False,
        )

        plotted_faces = []

        for ax, ft in zip(axes.ravel(), face_tensors):
            face_id = self._face_id_key(ft.get("face_id", 0))
            gidx = ft["global_vertex_idx"].detach().cpu()

            uv_face = ft["uv"].detach().cpu().numpy().astype(np.float32)
            Xu_face = ft["Xu"].detach().cpu().numpy().astype(np.float32)
            Xv_face = ft["Xv"].detach().cpu().numpy().astype(np.float32)
            rho_face = density_global[gidx].numpy().astype(np.float32)
            fiber_face = fiber_global[gidx].numpy().astype(np.float32)

            t_uv_face = self._fiber3d_to_uv_direction(
                Xu_np=Xu_face,
                Xv_np=Xv_face,
                fiber_np=fiber_face,
            )

            keep = (
                np.isfinite(rho_face)
                & np.isfinite(t_uv_face).all(axis=1)
                & (rho_face >= float(thr))
                & (np.linalg.norm(t_uv_face, axis=1) > 1e-10)
            )

            arrow_points = 0
            if np.count_nonzero(keep) > 0:
                uv_keep = uv_face[keep]
                rho_keep = rho_face[keep]
                t_uv_keep = t_uv_face[keep]

                if uv_keep.shape[0] > int(max_arrows_per_face):
                    pick = np.linspace(
                        0,
                        uv_keep.shape[0] - 1,
                        num=int(max_arrows_per_face),
                    ).round().astype(np.int64)
                    uv_keep = uv_keep[pick]
                    rho_keep = rho_keep[pick]
                    t_uv_keep = t_uv_keep[pick]

                ax.quiver(
                    uv_keep[:, 0],
                    uv_keep[:, 1],
                    t_uv_keep[:, 0],
                    t_uv_keep[:, 1],
                    rho_keep,
                    cmap=cmap,
                    clim=(0.0, 1.0),
                    angles="xy",
                    scale_units="xy",
                    scale=float(arrow_scale),
                    width=float(arrow_width),
                )
                arrow_points = int(uv_keep.shape[0])

            ax.set_title(f"Face {face_id} | arrows={arrow_points}")
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_aspect("equal")
            plotted_faces.append(face_id)

        for ax in axes.ravel()[len(face_tensors):]:
            ax.axis("off")

        fig.suptitle(
            f"Final fiber direction in UV domain on training points | thr={float(thr):.3f}",
            y=0.98,
        )
        fig.tight_layout()
        plt.show()

        print(
            f"2D fiber-direction visualization: plotted {len(plotted_faces)} faces "
            f"with threshold {thr:.3f}"
        )

        return {
            "figure": fig,
            "face_ids": plotted_faces,
            "thr_used": float(thr),
            "uv_by_face": {
                self._face_id_key(ft.get("face_id", 0)): ft["uv"].detach().cpu().numpy().astype(np.float32)
                for ft in face_tensors
            },
        }

    def visualize_result_final_fiber_direction_2d(self, *args, **kwargs):
        return self.Visualize_fresult_final_fiber_Direction_2D(*args, **kwargs)

    @staticmethod
    def _fiber3d_to_uv_direction(Xu_np, Xv_np, fiber_np, eps=1e-12):
        a11 = np.sum(Xu_np * Xu_np, axis=1)
        a12 = np.sum(Xu_np * Xv_np, axis=1)
        a22 = np.sum(Xv_np * Xv_np, axis=1)
        b1 = np.sum(Xu_np * fiber_np, axis=1)
        b2 = np.sum(Xv_np * fiber_np, axis=1)
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < eps, np.nan, det)

        du = (a22 * b1 - a12 * b2) / det
        dv = (-a12 * b1 + a11 * b2) / det
        tuv = np.stack([du, dv], axis=1)
        nrm = np.linalg.norm(tuv, axis=1, keepdims=True)
        ok = np.isfinite(tuv).all(axis=1, keepdims=True) & (nrm > eps)
        tuv = np.where(ok, tuv / np.clip(nrm, eps, None), 0.0)
        return tuv.astype(np.float32)

    def visualize_result_final_smooth_surface_pyvista(
        self,
        result,
        points_xyz=None,
        faces_ijk=None,
        shape_or_path=None,
        thr: float | str | None = 0.5,
        grid_res_u: int = 120,
        grid_res_v: int = 120,
        uv_mask_tol: float | None = None,
        show_density: bool = True,
        auto_target_volfrac: float | None = None,
        dense_factor: float = 1.0,
    ):
        import pyvista as pv
        import numpy as np

        grid_res_u, grid_res_v = self._resolve_visualization_grid_resolution(
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            dense_factor=dense_factor,
        )

        if shape_or_path is None:
            shape_or_path = result.get("shape_path", None)
        if shape_or_path is None:
            raise ValueError(
                "shape_or_path is required for smooth CAD-native visualization. "
                "Pass it explicitly or store it in result['shape_path']."
            )

        dense = self.sample_result_field_dense_for_visualization(
            result=result,
            shape_or_path=shape_or_path,
            grid_res_u=grid_res_u,
            grid_res_v=grid_res_v,
            uv_mask_tol=uv_mask_tol,
            use_best_pred=True,
        )

        rho_all = []
        area_w_all = []
        for face_data in dense["per_face"]:
            rho_i = face_data["rho_dense"]
            Xu_i = face_data["Xu_dense"]
            Xv_i = face_data["Xv_dense"]
            area_w_i = torch.linalg.norm(torch.cross(Xu_i, Xv_i, dim=1), dim=1).clamp_min(self.cfg.eps)
            rho_all.append(rho_i.detach().cpu().numpy())
            area_w_all.append(area_w_i.detach().cpu().numpy())

        rho_all = np.concatenate(rho_all, axis=0)
        area_w_all = np.concatenate(area_w_all, axis=0)
        area_w_sum = float(area_w_all.sum()) + float(self.cfg.eps)
        volfrac_cont = float((rho_all * area_w_all).sum() / area_w_sum)

        thr_used = thr
        if thr is None or (isinstance(thr, str) and str(thr).lower() == "auto"):
            target = self.cfg.target_volfrac if auto_target_volfrac is None else float(auto_target_volfrac)
            target = float(np.clip(target, 0.0, 1.0))

            # Weighted quantile so that area fraction above threshold ~= target.
            q = 1.0 - target
            order = np.argsort(rho_all)
            rho_s = rho_all[order]
            w_s = area_w_all[order]
            cdf = np.cumsum(w_s) / (np.sum(w_s) + float(self.cfg.eps))
            thr_used = float(np.interp(q, cdf, rho_s))
        else:
            thr_used = float(thr)

        volfrac_thr = float(area_w_all[rho_all >= thr_used].sum() / area_w_sum)
        print(
            f"[smooth_surface] thr={thr_used:.4f} | "
            f"volfrac_cont(rho)={volfrac_cont:.4f} | "
            f"volfrac_thr(binary)={volfrac_thr:.4f} | "
            f"target={self.cfg.target_volfrac:.4f} | "
            f"grid=({grid_res_u} x {grid_res_v})"
        )

        plotter = pv.Plotter()
        per_face = []

        for face_data in dense["per_face"]:
            xyz = face_data["xyz_dense"].detach().cpu().numpy().astype(np.float32)
            rho = face_data["rho_dense"].detach().cpu().numpy().astype(np.float32)
            face_id = face_data["face_id"]
            Nu, Nv = face_data["grid_shape"]
            mask = face_data["mask_dense_valid"].detach().cpu().numpy()
            uv = face_data["uv_dense"].detach().cpu().numpy().astype(np.float32)

            full_indices = -np.ones(mask.shape[0], dtype=np.int64)
            full_indices[mask] = np.arange(mask.sum(), dtype=np.int64)

            faces_keep = []

            def idx(i, j):
                return i * Nv + j

            for i in range(Nu - 1):
                for j in range(Nv - 1):
                    ids = [idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)]
                    mapped = [full_indices[k] for k in ids]
                    if any(m < 0 for m in mapped):
                        continue

                    i0, i1, i2, i3 = mapped
                    if rho[i0] >= thr_used and rho[i1] >= thr_used and rho[i2] >= thr_used:
                        faces_keep.append([i0, i1, i2])
                    if rho[i2] >= thr_used and rho[i1] >= thr_used and rho[i3] >= thr_used:
                        faces_keep.append([i2, i1, i3])

            faces_keep = np.asarray(faces_keep, dtype=np.int64)
            if faces_keep.size == 0:
                per_face.append({
                    "face_id": face_id,
                    "uv": uv,
                    "xyz": xyz,
                    "rho": rho,
                    "faces_keep": faces_keep,
                })
                continue

            pv_faces = np.empty((faces_keep.shape[0], 4), dtype=np.int64)
            pv_faces[:, 0] = 3
            pv_faces[:, 1:] = faces_keep
            mesh = pv.PolyData(xyz, pv_faces.reshape(-1))

            if show_density:
                mesh["rho"] = rho
                plotter.add_mesh(mesh, scalars="rho", cmap="viridis", clim=[0, 1])
            else:
                plotter.add_mesh(mesh, color="lightblue")

            per_face.append({
                "face_id": face_id,
                "uv": uv,
                "xyz": xyz,
                "rho": rho,
                "faces_keep": faces_keep,
            })
        plotter.show()
        return {
            "thr_used": float(thr_used),
            "volfrac_cont": float(volfrac_cont),
            "volfrac_thr": float(volfrac_thr),
            "dense": dense,
            "per_face": per_face,
        }
  
    def train(self, shape_path, face_tensors):
        cfg = self.cfg
        # time.perf_counter() returns the value (in fractional seconds) of a performance counter, i.e., a clock with the highest available resolution to measure a short duration.
        train_start_time = time.perf_counter()

        # Always train one face. If multiple faces are provided, use cfg.training_face_index.
        face_tensors = self._select_single_training_face(face_tensors)
        face_tensor = face_tensors[0]

        # validate the selected face tensor before training
        self._validate_face_tensors(face_tensors)
        self._auto_update_w_min_from_face_scale(face_tensor)

        # Assign device and data type used during training process
        ref_uv = face_tensor["uv"]
        device = ref_uv.device
        dtype = ref_uv.dtype
        mid_step = cfg.num_steps // 2

        # Total number of points used for training on the selected face
        gidx = face_tensor["global_vertex_idx"]
        vertices_number = int(gidx.max().item()) + 1
        # ------------------------------------------------------------
        # Build global vertex areas
        A_v = torch.zeros((vertices_number,), dtype=dtype, device=device)
        A_local = self.generator.vertex_area_lumped(
            face_tensor["uv"].shape[0],
            face_tensor["faces_ijk"],
            face_tensor["face_areas"],
        ).to(device=device, dtype=dtype)
        face_weight = A_local.sum().clamp_min(cfg.eps)
        A_v[gidx] += A_local

        # ------------------------------------------------------------
        # Build models / optimizer / scheduler
        # ------------------------------------------------------------
        decoder, ppnet = self._build_face_model(face_tensor=face_tensor, device=device)
        decoders = [decoder]
        ppnets = [ppnet]
        # Build initial seeds from the selected face tensor, which will be optimized during training.
        uv_init = self._init_face_seed(face_tensor)
        uv_anchor = uv_init.clone()
        uv_init_list = [uv_init]

        # Build the optimizer for all ppnet parameters. It includes the learning parameters,  optimizer type and learning rate are determined by the configuration (cfg).
        opt = self._build_optimizer(ppnet, decoder)
        # getatt(A,"S",None) is try to reach attribute "S" in object A, if it doesn't exist, it will return None instead of raising an error. 
        # here we are trying to get the "scheduler_milestones" attribute from the configuration (cfg). I
        #these milestones are specific training steps at which the learning rate will be adjusted according to a predefined schedule. 
        raw_milestones = getattr(cfg, "scheduler_milestones", None)
        if raw_milestones is None:
            milestones = []
        else:
            # isinstance(raw_milestones, (int, float)) checks if raw_milestones is a single number (int or float). 
            raw_seq = [raw_milestones] if isinstance(raw_milestones, (int, float)) else list(raw_milestones)
            milestones = []
            for m in raw_seq:
                m = float(m)
                # Support both fractional milestones (0..1] and absolute step indices (>1).
                step_m = int(round(m * cfg.num_steps)) if m <= 1.0 else int(round(m))
                if 0 < step_m < cfg.num_steps:
                    milestones.append(step_m)
            milestones = sorted(set(milestones))

        #print(f"scheduler_milestones: {milestones}")

        scheduler = self._build_scheduler(opt, milestones)

        # ------------------------------------------------------------
        # Optional timelapse setup
        # ------------------------------------------------------------
        recorder = None
        render_cache = None
        timelapse_output_folder = None
        if cfg.MakeTimelaps:
            case_name = shape_path.stem
            timelapse_output_folder = getattr(cfg, "timelapse_output_folder", None)
            if timelapse_output_folder:
                timelapse_output_folder = os.path.normpath(str(timelapse_output_folder))
                os.makedirs(timelapse_output_folder, exist_ok=True)
                frame_out_dir = os.path.join(timelapse_output_folder, "timelapse_frames")
                video_path = os.path.join(timelapse_output_folder, case_name + "_timelapse.avi")
            else:
                frame_out_dir = "timelapse_frames"
                video_path = case_name + "_timelapse.avi"
            # defining the timelapse recorder, which will save the training progress as a video. 
            # The output directory for the frames is "timelapse_frames", 
            # the video will be saved with the name "{case_name}_timelapse.avi". 
            # The frames per second (fps) for the video is set to 8.
            if self.shell_problem is not None and getattr(self.shell_problem, "mesh", None) is not None:
                fem_mesh = self.shell_problem.mesh
                fem_elems = int(fem_mesh["nelx"]) * int(fem_mesh["nely"]) * int(fem_mesh["nelz"])
            else:
                fem_elems = 0
            

            load_value = (
            float(getattr(self.shell_problem, "Load_magnitude", 0.0))
            if self.shell_problem is not None
            else 0.0
        )
            geometry_summary = self._timelapse_geometry_summary(face_tensors)
            recorder = TimelapseRecorder(
                out_dir=frame_out_dir,
                video_path=video_path,
                fps=8,
                header_title=(
                    f"{shape_path.name} ({geometry_summary}) | "
                    f"BC: {cfg.LoadingCasee} (F = {load_value:.3f} , FEM elements: {fem_elems}) | "
                    f"Target volfrac: {cfg.target_volfrac:.3f}"
                ),
                header_subtitle=self._timelapse_optimized_parameter_summary(),
            )
            # building a cache for rendering the timelapse, which likely includes precomputing certain data or settings that will be used 
            # repeatedly during the rendering of each frame in the timelapse video. 
            render_cache = self.build_timelapse_render_cache(
                face_tensors=face_tensors,
            )
            if self.timelapse_loading_img is None and self.shell_problem is not None:
                try:
                    self.timelapse_loading_img = self.shell_problem.show_voxels_surface_and_bc(
                        return_img=True,
                        off_screen=True,
                        window_size=(520, 280),
                    )
                    self.timelapse_loading_img = self._composite_to_white(self.timelapse_loading_img)
                except Exception as e:
                    tqdm.write(f"Failed to render timelapse loading panel: {e}")

        # ------------------------------------------------------------
        # Loss normalizers
        # ------------------------------------------------------------
        # These RunningNorm instances are used to keep track of the running mean and standard deviation of various loss components during training.
        # if on , it will normalize the loss components to have a more stable training process, especially when the scales of different loss terms vary significantly.
        norm_vol = RunningNorm()
        norm_rep = RunningNorm()
        norm_bnd = RunningNorm()
        norm_fem = RunningNorm()
        norm_curve_length = RunningNorm()
        norm_cell_edge_uniform = RunningNorm()

        # ------------------------------------------------------------
        # Best-state tracking
        # ------------------------------------------------------------
        best_score = float("inf")
        best_vol_frac = None
        best_comp = None
        best_w_geo = None
        best_step = -1
        best_active_count = None
        best_inactive_count = None
        best_rho = None
        best_fiber_surface = None
        best_seeds = None
        best_pred = None
        prune_best_score = float("inf")
        prune_best_step = -1
        prune_best_pred = None
        prune_best_uv_anchor = None
        prune_best_ppnet_state = None
        best_hard_score = float("inf")
        best_hard_vol_frac = None
        best_hard_comp = None
        best_hard_w_geo = None
        best_hard_step = -1
        best_hard_active_count = None
        best_hard_inactive_count = None
        best_hard_rho = None
        best_hard_fiber_surface = None
        best_hard_seeds = None
        best_hard_pred = None
        # ------------------------------------------------------------

        steps_since_improve = 0
        prune_events = []
        initial_shape_density = None
        mid_shape_density = None
        final_shape_density = None
        final_shape_fiber_direction = None
        seed_points_init = None
        seed_points_mid = None
        seed_points_final = None
        seed_points_init_uv = None
        seed_points_mid_uv = None
        seed_points_final_uv = None
        rho0 = None
        seeds0 = None
        anchor_update_allowed = True
        history = []

        self.current_face_tensors = face_tensors
        debug_anomaly_detection = bool(getattr(cfg, "debug_anomaly_detection", False))
        if debug_anomaly_detection:
            torch.autograd.set_detect_anomaly(True, check_nan=True)

        # ------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------
        # The main traiing loop iterates for a number of steps defined in the configuration (cfg.num_steps).
        # To have a progress bar for training, it uses the tqdm library, which provides a visual representation of the training progress in the console.
        # It is equal to for step in training_steps, but with an added progress bar that shows the current step and other relevant information during training.
        # desc="Training" sets the description of the progress bar to "Training", leave=True keeps the progress bar displayed after completion, and dynamic_ncols=True allows the progress bar to adjust its width dynamically based on the terminal size.
        with tqdm(
            range(cfg.num_steps),
            desc="Training",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for step in pbar:
                should_log = (
                    step == 0
                    or step % cfg.log_every == 0
                    or step == cfg.num_steps - 1
                )
    
                # if cfg.allow_seed_outside_domain is true and the warmup period is over, allow seeds to be placed outside the domain
                allow_seed_outside_domain_step = self.allow_seed_outside_domain_for_step(step)
                ppnet.allow_seed_outside_domain = allow_seed_outside_domain_step
                tau_step = self._tau_for_step(step)
                rho_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
                rho_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                fiber_acc = torch.zeros((vertices_number, 3), dtype=dtype, device=device)
                fiber_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)

                seeds_list = []
                pred_list = []
                density_post_stats_acc = {
                    "filter_delta_mean": 0.0,
                    "filter_delta_max": 0.0,
                    "projection_delta_mean": 0.0,
                    "projection_delta_max": 0.0,
                    "raw_mean": 0.0,
                    "filtered_mean": 0.0,
                    "projected_mean": 0.0,
                    "final_mean": 0.0,
                }
                density_post_stats_weight = 0.0

                rep_terms = []
                bnd_terms = []
                curve_length_terms = []
                curve_length_values = []
                cell_edge_uniform_terms = []
                w_geo_terms = []
                h_terms = []
                centerline_radius_terms = []
                participating_count_total = 0.0
                participating_frac_sum = 0.0
                inactive_count_total = 0.0
                inactive_frac_sum = 0.0

                # Activate losses based on their lambda values in the configuration (cfg). If a lambda value is set to 0.0, the corresponding loss will not be computed during training
                compute_rep_loss = cfg.lam_rep != 0.0
                compute_bnd_loss = cfg.lam_bnd != 0.0
                compute_vol_loss = cfg.lam_vol != 0.0
                compute_curve_length_loss = getattr(cfg, "lam_curve_length", 0.0) != 0.0
                compute_cell_edge_uniform_loss = (
                    getattr(cfg, "lam_cell_edge_uniform", 0.0) != 0.0
                )

                # Determine whether to update seed anchors based on the configuration and current step, seed anchors are reference points used in the training process.
                # if it is on, it will update the seed anchors after a certain warmup period, and the update is allowed based on the configuration settings.
                update_seed_anchors = (
                    cfg.use_rolling_seed_anchors
                    and step >= int(round(float(cfg.seed_anchor_warmup_frac) * float(cfg.num_steps)))
                    and (anchor_update_allowed or not cfg.guard_seed_anchor_updates)
                )

                seed_offset_scale_step = self.seed_offset_scale_for_step(step)
                #seed_offset_scale_step=cfg.Offset_scale
                uv_anchor_next = None

                face_idx = 0
                ft = face_tensor
                uv_anchor_i = uv_anchor
                face_weight_i = face_weight
                if True:
                    pred_i = ppnet(uv_anchor_i, offset_scale=seed_offset_scale_step)

                    seeds_raw_i = pred_i["seeds_raw"]
                    # repulse seed by itersionally projecting them to be more evenly spaced, which can help improve the stability and convergence of the training process.
                    if cfg.project_seed_spacing_each_step:
                        seeds_raw_i = self.project_seed_spacing(
                            seeds_list=[seeds_raw_i],
                            min_dist=float(cfg.collapse_min_seed_dist_factor) * float(0.01),
                            iters=int(cfg.seed_projection_iters),
                            detach=False,
                            clamp_to_domain=not allow_seed_outside_domain_step,
                        )[0]
                        pred_i["seeds_raw"] = pred_i["seeds_raw"].clone()
                        pred_i["seeds_raw"] = seeds_raw_i
                    w_raw_i = pred_i["w_raw"]

                    local_face_id = torch.zeros(ft["uv"].shape[0], dtype=torch.long, device=device)

                    boundary_uv_i = None
                    boundary_face_id_i = None
                    boundary_loop_id_i = None
                    true_bidx_i, boundary_loop_id_i = self._ordered_true_open_boundary(ft)
                    if true_bidx_i.numel() > 0:
                        boundary_uv_i = ft["uv"][true_bidx_i]
                        boundary_face_id_i = torch.zeros(
                            boundary_uv_i.shape[0],
                            dtype=torch.long,
                            device=device,
                        )
                    seed_domain_mask_i = self._seed_domain_mask_for_face(ft)


                    decoder_out = decoder(
                        seeds_uv=seeds_raw_i,
                        w_raw=w_raw_i,
                        generate_density_fiber=getattr(cfg, "generate_decoder_density_fiber", True),
                    )

                    self._require_decoder_keys(
                        decoder_out,
                        [
                            "seeds",
                        ],
                    )

                    if compute_curve_length_loss:
                        curve_length_values.append(self.curve_3d_edge_lengths(decoder_out).detach())
                        curve_length_terms.append(self.curve_length_similarity_loss(decoder_out))
                    if compute_cell_edge_uniform_loss:
                        cell_edge_uniform_terms.append(
                            self.cell_edge_uniformity_loss(decoder_out)
                        )

                    seeds_i = decoder_out["seeds"]
                    if getattr(cfg, "generate_decoder_density_fiber", True):
                        decoder_out, density_post_stats_i = apply_density_postprocess_to_output(
                            decoder_out,
                            ft,
                            cfg,
                            return_debug=True,
                        )
                        self._require_decoder_keys(
                            decoder_out,
                            [
                                "rho",
                                "fiber3d",
                            ],
                        )
                        rho_i = decoder_out["rho"]
                        fiber3d_i = decoder_out["fiber3d"]
                    else:
                        rho_i, fiber3d_i, density_post_stats_i = self.neutral_density_fiber_fields(
                            ft["uv"],
                            ft.get("Xu", None),
                        )
                    if "w_geo" in decoder_out:
                        w_geo_i = decoder_out["w_geo"]
                    elif hasattr(decoder, "width"):
                        w_geo_i = decoder.width(w_raw_i, seeds=seeds_i)
                    else:
                        w_geo_i = w_raw_i
                    h_i = decoder_out.get("h", torch.zeros((), dtype=dtype, device=device))
                    centerline_radius_i = decoder_out.get(
                        "centerline_radius",
                        _centerline_radius_raw_from_w(cfg, w_raw_i),
                    )
                    active_count_i = float(seeds_i.shape[0])
                    inactive_count_i = 0.0
                    total_seed_i = max(int(seeds_i.shape[0]), 1)
                    participating_count_total += active_count_i
                    participating_frac_sum += active_count_i / float(total_seed_i)
                    inactive_count_total += inactive_count_i
                    inactive_frac_sum += inactive_count_i / float(total_seed_i)

                    for name, t in {
                        "seeds_i": seeds_i,
                        "rho_i": rho_i,
                        "fiber3d_i": fiber3d_i,
                    }.items():
                        if not torch.isfinite(t).all():
                            tqdm.write(f"[step {step}] face {ft['face_id']} invalid tensor: {name}")
                            raise RuntimeError(
                                f"Invalid decoder output on face {ft['face_id']} at step {step}"
                            )

                    gidx = ft["global_vertex_idx"]
                    w_local = A_local.clamp_min(cfg.eps)
                    stats_weight_i = float(w_local.detach().sum().item())
                    density_post_stats_weight += stats_weight_i
                    for key, value in density_post_stats_i.items():
                        if key.endswith("_max"):
                            density_post_stats_acc[key] = max(
                                density_post_stats_acc[key],
                                float(value),
                            )
                        else:
                            density_post_stats_acc[key] += float(value) * stats_weight_i

                    rho_acc[gidx] += rho_i * w_local
                    rho_wgt[gidx] += w_local

                    fiber_acc[gidx] += fiber3d_i * w_local[:, None]
                    fiber_wgt[gidx] += w_local

                    seeds_list.append(seeds_i)
                    if update_seed_anchors:
                        anchor_alpha = float(cfg.seed_anchor_momentum)
                        uv_anchor_next_i = (
                            (1.0 - anchor_alpha) * uv_anchor_i + anchor_alpha * seeds_i.detach()
                        )
                    else:
                        uv_anchor_next_i = uv_anchor_i.detach().clone()
                    uv_anchor_next = uv_anchor_next_i

                    pred_list.append({
                        "face_id": self._face_id_key(ft.get("face_id", 0)),
                        "seeds_raw": seeds_raw_i.detach().clone(),
                        "w_raw": w_raw_i.detach().clone(),
                        "tau": tau_step.detach().clone() if isinstance(tau_step, torch.Tensor) else float(tau_step),
                        "w_geo": w_geo_i.detach().clone(),
                        "h": h_i.detach().clone() if isinstance(h_i, torch.Tensor) else h_i,
                        "centerline_radius": centerline_radius_i.detach().clone() if isinstance(centerline_radius_i, torch.Tensor) else centerline_radius_i,
                        "seeds_uv": decoder_out.get("seeds_uv", seeds_i).detach().clone(),
                        "seeds_xyz": decoder_out["seeds_xyz"].detach().clone() if isinstance(decoder_out.get("seeds_xyz"), torch.Tensor) else None,
                        "edge_curves_uv": decoder_out["edge_curves_uv"].detach().clone() if isinstance(decoder_out.get("edge_curves_uv"), torch.Tensor) else None,
                        "edge_curves_xyz": decoder_out["edge_curves_xyz"].detach().clone() if isinstance(decoder_out.get("edge_curves_xyz"), torch.Tensor) else None,
                    })

                    if compute_rep_loss:
                        rep_terms.append(
                            self.loss_rep(
                                seeds=seeds_i,
                                seed_active_weights=None,
                                sigma=cfg.seed_repulsion_sigma,
                                min_dist=float(cfg.collapse_min_seed_dist_factor) * float(cfg.w_min),
                                eps=cfg.eps,
                            )
                        )

                    if compute_bnd_loss:
                        bnd_terms.append(
                            self.loss_boundary(
                                seeds=seeds_i,
                                boundary_uv=boundary_uv_i,
                                seed_active_weights=None,
                                margin=cfg.boundary_margin,
                                eps=cfg.eps,
                            )
                        )

                    w_geo_terms.append(self._pair_upper_values(w_geo_i).mean().reshape(()))
                    h_terms.append(h_i.reshape(()))
                    if isinstance(centerline_radius_i, torch.Tensor) and centerline_radius_i.numel() > 0:
                        centerline_radius_terms.append(centerline_radius_i.mean().reshape(()))

                uv_anchor = uv_anchor_next

                # ----------------------------------------------------
                # Selected-face outputs
                # ----------------------------------------------------
                participating_count_mean = participating_count_total
                participating_frac_mean = participating_frac_sum
                inactive_count_mean = inactive_count_total
                inactive_frac_mean = inactive_frac_sum

                rho = rho_acc / rho_wgt.clamp_min(cfg.eps)
                density_post_stats = dict(density_post_stats_acc)
                if density_post_stats_weight > 0.0:
                    for key in [
                        "filter_delta_mean",
                        "projection_delta_mean",
                        "raw_mean",
                        "filtered_mean",
                        "projected_mean",
                        "final_mean",
                    ]:
                        density_post_stats[key] = (
                            density_post_stats_acc[key] / density_post_stats_weight
                        )

                fiber_surface = fiber_acc / fiber_wgt.clamp_min(cfg.eps)[:, None]
                fiber_norm = fiber_surface.norm(dim=1, keepdim=True).clamp_min(cfg.eps)
                fiber_surface = fiber_surface / fiber_norm

                zero = self._trainable_zero(ppnets, dtype=dtype, device=device)

                loss_rep = rep_terms[0] if compute_rep_loss and rep_terms else zero
                loss_bnd = bnd_terms[0] if compute_bnd_loss and bnd_terms else zero
                loss_curve_length = (
                    curve_length_terms[0]
                    if compute_curve_length_loss and curve_length_terms
                    else zero
                )
                loss_cell_edge_uniform = (
                    cell_edge_uniform_terms[0]
                    if compute_cell_edge_uniform_loss and cell_edge_uniform_terms
                    else zero
                )

                w_geo_mean = w_geo_terms[0] if w_geo_terms else zero
                h_mean = h_terms[0] if h_terms else zero
                centerline_radius_mean = centerline_radius_terms[0] if centerline_radius_terms else zero

                # ----------------------------------------------------
                # Volume loss
                # ----------------------------------------------------
                vol_frac_total = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                vol_frac_eff_total = self.loss_volume.powered_fraction(
                    rho=rho,
                    A_v=A_v,
                    power=cfg.effective_volume_power,
                    eps=cfg.eps,
                )
                vol_frac_eff = vol_frac_eff_total
                loss_vol = zero

                if compute_vol_loss:
                    loss_vol_total = self.loss_volume(
                        rho=rho,
                        A_v=A_v,
                        target_volfrac=cfg.target_volfrac,
                        eps=cfg.eps,
                    )

                    loss_vol_eff, vol_frac_eff = self.loss_volume.powered(
                        rho=rho,
                        A_v=A_v,
                        target_volfrac=cfg.target_volfrac,
                        power=cfg.effective_volume_power,
                        eps=cfg.eps,
                    )
                    loss_vol = loss_vol_eff



                # ----------------------------------------------------
                # FEM loss
                # ----------------------------------------------------
                fem_out = {
                    "fem_total": torch.zeros((), dtype=dtype, device=device),
                    "comp": torch.zeros((), dtype=dtype, device=device),
                    "compliance_loss": torch.zeros((), dtype=dtype, device=device),
                    "fem_valid": True,
                    "failure_reason": None,
                }

                if cfg.lam_fem != 0.0:
                    fem_out = self.loss_fem.evaluate(
                        rho_surface=rho,
                        fiber_surface=fiber_surface,
                        comp_normalize_by=cfg.comp_normalize_by,
                        density_floor=cfg.fem_density_floor,
                        eps=cfg.eps,
                        save_debug_history=getattr(cfg, "save_fem_debug_history", True),
                    )

                loss_fem = fem_out["fem_total"]
                loss_comp = fem_out["compliance_loss"]
                comp_val = fem_out["comp"]
                fem_is_valid = bool(fem_out["fem_valid"])
                fem_failure_reason = fem_out["failure_reason"]

                # ----------------------------------------------------
                # Normalize losses
                # ----------------------------------------------------
                if cfg.normalize_losses:
                    n_vol = norm_vol.update(loss_vol.detach().item())
                    n_rep = norm_rep.update(loss_rep.detach().item())
                    n_bnd = norm_bnd.update(loss_bnd.detach().item())
                    n_curve_length = norm_curve_length.update(loss_curve_length.detach().item())
                    n_cell_edge_uniform = norm_cell_edge_uniform.update(
                        loss_cell_edge_uniform.detach().item()
                    )
                    n_fem = norm_fem.update(loss_fem.detach().item()) if (cfg.lam_fem != 0.0 and fem_is_valid) else 1.0
                else:
                    n_vol = n_rep = n_bnd = n_fem = n_curve_length = n_cell_edge_uniform = 1.0

                # ----------------------------------------------------
                # Total loss
                # ----------------------------------------------------
                lam_width_active_eff = 0.0

                L_total = (
                    zero
                    + cfg.lam_vol * (loss_vol / n_vol)
                    + cfg.lam_rep * (loss_rep / n_rep)
                    + cfg.lam_bnd * (loss_bnd / n_bnd)
                    + cfg.lam_curve_length * (loss_curve_length / n_curve_length)
                    + cfg.lam_cell_edge_uniform
                    * (loss_cell_edge_uniform / n_cell_edge_uniform)
                )

                if cfg.lam_fem != 0.0:
                    if fem_is_valid:
                        L_total = L_total + cfg.lam_fem * (loss_fem / n_fem)
                    elif not cfg.skip_bad_fem_steps:
                        L_total = L_total + cfg.lam_fem * loss_fem

                total_is_finite = self._scalar_tensor_is_finite(L_total)
                loss_debug_terms = [
                    ("L_total", L_total),
                    ("loss_vol", loss_vol),
                    ("loss_rep", loss_rep),
                    ("loss_bnd", loss_bnd),
                    ("loss_curve_length", loss_curve_length),
                    ("loss_cell_edge_uniform", loss_cell_edge_uniform),
                    ("loss_fem", loss_fem),
                    ("loss_comp", loss_comp),
                ]

                # ----------------------------------------------------
                # Backprop
                # ----------------------------------------------------
                opt.zero_grad(set_to_none=True)

                if total_is_finite:
                    L_total.backward()

                    bad_grad_info = self._nonfinite_grad_info(ppnets)
                    if bad_grad_info:
                        cause_desc = self._nonfinite_grad_cause_summary(
                            ppnets,
                            bad_grad_info,
                            loss_terms=loss_debug_terms,
                            fem_is_valid=fem_is_valid,
                            fem_failure_reason=fem_failure_reason,
                        )
                        tqdm.write(
                            f"[step {step}] Non-finite gradients detected; optimizer step skipped. "
                            f"{cause_desc}."
                        )
                        for _mi, _pn, p in self._named_trainable_params(ppnets):
                            if p.grad is not None:
                                p.grad = None
                    else:
                        pre_step_snapshot = {
                            p: p.detach().clone()
                            for _mi, _pn, p in self._named_trainable_params(ppnets)
                        }

                        grad_clip_norm = getattr(cfg, "grad_clip_norm", None)
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            params = [p for p in ppnet.parameters() if p.requires_grad]
                            if params:
                                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)

                        bad_grad_info = self._nonfinite_grad_info(ppnets)
                        if bad_grad_info:
                            cause_desc = self._nonfinite_grad_cause_summary(
                                ppnets,
                                bad_grad_info,
                                loss_terms=loss_debug_terms,
                                fem_is_valid=fem_is_valid,
                                fem_failure_reason=fem_failure_reason,
                            )
                            tqdm.write(
                                f"[step {step}] Non-finite gradients remained after clipping, "
                                f"optimizer step skipped. {cause_desc}."
                            )
                            for _mi, _pn, p in self._named_trainable_params(ppnets):
                                if p.grad is not None:
                                    p.grad = None
                        else:
                            opt.step()

                            bad_param_info = self._nonfinite_param_info(ppnets)
                            if bad_param_info:
                                bad_param_desc = ", ".join(
                                    f"face={mi}:{pn}" for mi, pn in bad_param_info[:8]
                                )
                                self._restore_param_snapshot(pre_step_snapshot)
                                bad_param_set = set(bad_param_info)
                                affected_params = [
                                    p for mi, pn, p in self._named_trainable_params(ppnets)
                                    if (mi, pn) in bad_param_set
                                ]
                                self._clear_optimizer_state_for_params(opt, affected_params)
                                for _mi, _pn, p in self._named_trainable_params(ppnets):
                                    if p.grad is not None:
                                        p.grad = None
                                tqdm.write(
                                    f"[step {step}] Non-finite parameters after opt.step(); restored previous "
                                    f"parameters and cleared optimizer state. Examples: {bad_param_desc}"
                                )
                            elif scheduler is not None:
                                scheduler.step()
                else:
                    tqdm.write(f"[step {step}] L_total is non-finite, optimizer step skipped.")

                # ----------------------------------------------------
                # Logging / tracking
                # ----------------------------------------------------
                with torch.no_grad():
                    vol_frac = (rho * A_v).sum() / (A_v.sum() + cfg.eps)
                    vol_dev = torch.abs(vol_frac - cfg.target_volfrac)
                    vol_dev_eff = torch.abs(vol_frac_eff - cfg.target_volfrac)
                    min_seed_dist = self.min_pairwise_seed_distance(seeds_list)

                    score = float(L_total.detach().item()) if total_is_finite else float("inf")
                    if not (total_is_finite and fem_is_valid):
                        score = float("inf")

                    best_candidate_is_valid = (
                        ((cfg.lam_fem == 0.0) or fem_is_valid)
                        and total_is_finite
                        and participating_count_total >= float(cfg.min_active_seeds or 1)
                    )

                    prev_best_step = best_step
                    improvement_gap = (step - prev_best_step) if prev_best_step >= 0 else None

                    if step == 0:
                        initial_shape_density = rho.detach().clone()
                        seed_points_init_uv = seeds_i.detach().clone()
                        seed_points_init = self._seed_points_xyz(seeds_i, face_tensor)

                    if step == mid_step:
                        mid_shape_density = rho.detach().clone()
                        seed_points_mid_uv = seeds_i.detach().clone()
                        seed_points_mid = self._seed_points_xyz(seeds_i, face_tensor)

                    prune_best_improved = (
                        best_candidate_is_valid
                        and score < (prune_best_score - cfg.min_delta)
                    )
                    if prune_best_improved:
                        prune_best_score = score
                        prune_best_step = step
                        prune_best_pred = self._clone_pred_list(pred_list)
                        prune_best_uv_anchor = uv_anchor.detach().clone()
                        prune_best_ppnet_state = self._clone_module_state_dict(ppnet)
                        steps_since_improve = 0
                    elif best_candidate_is_valid:
                        steps_since_improve += 1

                    if best_candidate_is_valid and score < (best_score - cfg.min_delta):
                        best_score = score
                        best_step = step
                        best_vol_frac = float(vol_frac_eff.detach().item())
                        best_comp = float(comp_val.detach().item())
                        best_w_geo = float(w_geo_mean.detach().item())
                        best_active_count = float(participating_count_total)
                        best_inactive_count = float(inactive_count_total)
                        best_rho = rho.detach().clone()
                        best_fiber_surface = fiber_surface.detach().clone()
                        best_seeds = [s.detach().clone() for s in seeds_list]
                        best_pred = self._clone_pred_list(pred_list)

                        if improvement_gap is None or improvement_gap > 50:
                            tqdm.write(
                                f"New best_step={best_step} | "
                                f"best_score={best_score:.6f} | "
                                f"best_active_count={best_active_count:.1f} | "
                                f"VF_total={float(vol_frac.detach().item()):.6f} | "
                                f"VF_eff_total={float(vol_frac_eff_total.detach().item()):.6f} | "
                                f"VF_eff_int={best_vol_frac:.6f} | "
                                f"comp={best_comp:.6e} | "
                                f"w={best_w_geo:.6e}"
                            )

                    if rho0 is None:
                        rho0 = rho.detach().clone()
                    if seeds0 is None:
                        seeds0 = [s.detach().clone() for s in seeds_list]

                    drho = float((rho - rho0).abs().mean().item())
                    dseed_terms = [float((s - s0).abs().mean().item()) for s, s0 in zip(seeds_list, seeds0)]
                    dseed = sum(dseed_terms) / max(len(dseed_terms), 1)

                    rho_min = float(rho.min().item())
                    rho_mean = float(rho.mean().item())
                    rho_max = float(rho.max().item())
                    nonempty_curve_lengths = [
                        v.reshape(-1) for v in curve_length_values if v.numel() > 0
                    ]
                    if nonempty_curve_lengths:
                        curve_lengths = torch.cat(nonempty_curve_lengths, dim=0)
                    else:
                        curve_lengths = zero.new_empty((0,))
                    if curve_lengths.numel() > 0:
                        curve_length_min = float(curve_lengths.min().item())
                        curve_length_max = float(curve_lengths.max().item())
                        curve_length_mean = float(curve_lengths.mean().item())
                    else:
                        curve_length_min = float("nan")
                        curve_length_max = float("nan")
                        curve_length_mean = float("nan")

                    g_mean = 0.0
                    g_count = 0
                    for p in ppnet.parameters():
                        if p.grad is not None:
                            g_mean += float(p.grad.detach().abs().mean().item())
                            g_count += 1
                    g_mean = g_mean / max(g_count, 1)

                    row = {
                        "step": step,
                        "L_total": self._finite_or_default(L_total),
                        "loss_vol": self._finite_or_default(loss_vol),
                        "loss_rep": self._finite_or_default(loss_rep),
                        "loss_bnd": self._finite_or_default(loss_bnd),
                        "loss_curve_length": self._finite_or_default(loss_curve_length),
                        "curve_length_min": curve_length_min,
                        "curve_length_max": curve_length_max,
                        "curve_length_mean": curve_length_mean,
                        "loss_cell_edge_uniform": self._finite_or_default(loss_cell_edge_uniform),
                        "loss_fem": self._finite_or_default(loss_fem),
                        "loss_comp": self._finite_or_default(loss_comp),
                        "comp": self._finite_or_default(comp_val),
                        "vol_frac": float(vol_frac.detach().item()),
                        "vol_frac_internal": float(vol_frac.detach().item()),
                        "vol_frac_eff_total": float(vol_frac_eff_total.detach().item()),
                        "vol_frac_eff": float(vol_frac_eff.detach().item()),
                        "VF_total": float(vol_frac.detach().item()),
                        "VF_eff_total": float(vol_frac_eff_total.detach().item()),
                        "VF_int": float(vol_frac.detach().item()),
                        "VF_eff_int": float(vol_frac_eff.detach().item()),
                        "vol_dev": float(vol_dev.detach().item()),
                        "vol_dev_eff": float(vol_dev_eff.detach().item()),
                        "tau": float(tau_step),
                        "seed_offset_scale": float(seed_offset_scale_step),
                        "rho_min": rho_min,
                        "rho_mean": rho_mean,
                        "rho_max": rho_max,
                        "filter_delta_mean": density_post_stats["filter_delta_mean"],
                        "filter_delta_max": density_post_stats["filter_delta_max"],
                        "projection_delta_mean": density_post_stats["projection_delta_mean"],
                        "projection_delta_max": density_post_stats["projection_delta_max"],
                        "rho_raw_mean": density_post_stats["raw_mean"],
                        "rho_filtered_mean": density_post_stats["filtered_mean"],
                        "rho_projected_mean": density_post_stats["projected_mean"],
                        "rho_final_mean": density_post_stats["final_mean"],
                        "drho": drho,
                        "dseed": dseed,
                        "min_seed_dist": min_seed_dist,
                        "grad_mean": g_mean,
                        "best_score": best_score,
                        "best_step": best_step,
                        "best_hard_score": self._finite_or_default(best_hard_score),
                        "best_hard_step": float(best_hard_step),
                        "fem_valid": fem_is_valid,
                        "fem_failure_reason": fem_failure_reason,
                        "optimizer_step_skipped": not total_is_finite,
                        "w_geo_mean": self._finite_or_default(w_geo_mean),
                        "h_mean": self._finite_or_default(h_mean),
                        "centerline_radius_mean": self._finite_or_default(centerline_radius_mean),

                        "active_count_total": participating_count_total,
                        "active_count_mean": participating_count_mean,
                        "active_frac_mean": participating_frac_mean,
                        "inactive_count_total": inactive_count_total,
                        "inactive_count_mean": inactive_count_mean,
                        "inactive_frac_mean": inactive_frac_mean,
                        "anchor_update_allowed": 1.0 if anchor_update_allowed else 0.0,
                        "collapse_active": (
                            1.0
                            if participating_count_total < float(cfg.min_active_seeds or 1)
                            else 0.0
                        ),
                    }
                    history.append(row)

                    pbar.set_postfix(
                        loss=f"{row['L_total']:.3e}",
                        vol=f"{row['vol_frac_eff']:.3f}",
                        comp=f"{row['comp']:.2e}",
                        tau=f"{row['tau']:.2e}",
                        w=f"{row['w_geo_mean']:.3e}",
                        clr=f"{row['centerline_radius_mean']:.3e}",
                        lcurve=f"{row['loss_curve_length']:.2e}",
                        lcell=f"{row['loss_cell_edge_uniform']:.2e}",
                        dmin=f"{row['min_seed_dist']:.3e}",
                        active=f"{participating_count_mean:.1f}",
                        fem="OK" if fem_is_valid else "BAD",
                        refresh=False,
                    )

                    if cfg.MakeTimelaps and step % cfg.timelapse_frame_step == 0:
                        if getattr(cfg, "timelapse_show_3d_tubes", True):
                            cad_img = self._render_current_3d_tube_frame_cached(
                                seeds_list=seeds_list,
                                decoders=decoders,
                                pred_list=pred_list,
                                render_cache=render_cache,
                                loading_img=self.timelapse_loading_img,
                            )
                        else:
                            cad_img = self._render_current_cad_frame_cached(
                                seeds_list=seeds_list,
                                decoders=decoders,
                                pred_list=pred_list,
                                render_cache=render_cache,
                                thr=getattr(cfg, "vis_thr", cfg.TM_laps_Thr),
                                loading_img=self.timelapse_loading_img,
                            )

                        loss_dict = {
                            "L_Total": row["L_total"],
                            "L_Volume": row["loss_vol"],
                            "L_FEM": row["loss_fem"],
                            "L_Bnd": row["loss_bnd"],
                            "L_Rep": row["loss_rep"],
                            "L_CurveLen": row["loss_curve_length"],
                            "L_CellEdge": row["loss_cell_edge_uniform"],
                        }

                        recorder.add_frame(
                            step=step,
                            cad_img=cad_img,
                            loss_dict=loss_dict,
                            title_text=(
                                f"VF_total={row['VF_total']:.4f} | "
                                f"VF_eff_total={row['VF_eff_total']:.4f} | "
                                f"VF_int={row['VF_int']:.4f} | "
                                f"VF_eff_int={row['VF_eff_int']:.4f} | "
                                f"W={row['w_geo_mean']:.4g} | "
                                f"CLR={row['centerline_radius_mean']:.4g} | "
                                f"tau={row['tau']:.4g} | "
                                f"act={row['active_count_total']:.0f} | "
                                f"Δrho={drho:.2e} Δseed={dseed:.2e} "
                                f"dmin={min_seed_dist:.2e} grad_mean={g_mean:.2e} | "
                            ),
                        )

                    self._tb_log_step(
                        step=step,
                        row=row,
                        rho=rho,
                        fiber_surface=fiber_surface,
                        seeds_list=seeds_list,
                        pred_list=pred_list,
                    )

                    if (not fem_is_valid) and cfg.skip_bad_fem_steps:
                        self._print_fem_failure(step)

                    if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
                        fem_status = "OK" if fem_is_valid else f"BAD({fem_failure_reason})"
                        tqdm.write(
                            f"[{step:05d}] | "
                            f"Active Seeds/Total={participating_count_total:.0f}/{participating_count_total+inactive_count_total:.0f} | "

                            f"L_total={row['L_total']:.4e} | "
                            f"L_vol={row['loss_vol']:.3e} "
                            f"L_fem={row['loss_fem']:.3e} "
                            f"L_rep={row['loss_rep']:.3e} "
                            f"L_bnd={row['loss_bnd']:.3e} "
                            f"L_curve={row['loss_curve_length']:.3e} "
                            f"L(min/max/mean)={row['curve_length_min']:.3e}/{row['curve_length_max']:.3e}/{row['curve_length_mean']:.3e} "
                            f"L_cell_edge={row['loss_cell_edge_uniform']:.3e} |"
                            f"VF_total={row['VF_total']:.3f} "
                            f"VF_eff_total={row['VF_eff_total']:.3f} "
                            f"VF_int={row['VF_int']:.3f} "
                            f"VF_eff_int={row['VF_eff_int']:.3f} "
                            f"(/{cfg.target_volfrac:.3f}) "
                            f"tau={row['tau']:.3e} "
                            f"os={row['seed_offset_scale']:.2e} "
                            f"comp={row['comp']:.3e} | "
                            f"w={row['w_geo_mean']:.3e} "
                            f"clr={row['centerline_radius_mean']:.3e} "
                            f"h={row['h_mean']:.3e} | "
                            f"rho(min/mean/max)={rho_min:.3f}/{rho_mean:.3f}/{rho_max:.3f} "
                            f"Δrho={drho:.2e} Δseed={dseed:.2e} "
                            f"dmin={min_seed_dist:.2e} grad_mean={g_mean:.2e} | "
                                                        f"Filter Δrho mean={row['filter_delta_mean']:.2e} "
                            f"Filter Δrho max={row['filter_delta_max']:.2e} "
                            f"Projection Δrho mean={row['projection_delta_mean']:.2e} "
                            f"Projection Δrho max={row['projection_delta_max']:.2e} | "
                            f"rho_raw_mean={row['rho_raw_mean']:.3f} "
                            f"rho_filtered_mean={row['rho_filtered_mean']:.3f} "
                            f"rho_final_mean={row['rho_final_mean']:.3f} | "
                            f"fem={fem_status} | "
                            f"best={best_score:.4e}@{best_step} | "
                            f"best_hard={best_hard_score:.4e}@{best_hard_step}"
                        )

                    rep_value = float(row["loss_rep"])
                    bnd_value = float(row["loss_bnd"])
                    vol_eff_value = float(row["vol_frac_eff"])
                    w_geo_value = float(row["w_geo_mean"])
                    min_seed_dist_value = float(row["min_seed_dist"])
                    min_seed_dist_limit = float(cfg.anchor_guard_min_seed_dist_factor) * float(cfg.w_min)

                    anchor_update_allowed = (
                        rep_value <= float(cfg.anchor_guard_rep_max)
                        and bnd_value <= float(cfg.anchor_guard_bnd_max)
                        and vol_eff_value >= float(cfg.anchor_guard_vol_eff_min)
                        and w_geo_value >= float(cfg.anchor_guard_width_factor_min) * float(cfg.w_min)
                        and min_seed_dist_value >= min_seed_dist_limit
                    )

                    prune_wait = int(cfg.prune_patience or cfg.patience)
                    if (
                        bool(cfg.prune_inactive_on_plateau)
                        and prune_best_step >= 0
                        and steps_since_improve >= prune_wait
                        and prune_best_pred
                        and prune_best_uv_anchor is not None
                        and prune_best_ppnet_state is not None
                    ):
                        old_seed_count_current = int(getattr(ppnet, "n_seeds", cfg.seed_number))
                        prune_active_mask = prune_best_pred[0].get("seed_active_mask", None)
                        if prune_active_mask is not None:
                            prune_active_mask = prune_active_mask.detach().to(
                                device=prune_best_uv_anchor.device,
                                dtype=torch.bool,
                            ).reshape(-1)
                            old_seed_count_for_prune = int(prune_active_mask.numel())
                            new_seed_count_for_prune = int(prune_active_mask.to(torch.long).sum().item())
                            removed_count_for_prune = old_seed_count_for_prune - new_seed_count_for_prune
                        else:
                            old_seed_count_for_prune = old_seed_count_current
                            new_seed_count_for_prune = old_seed_count_current
                            removed_count_for_prune = 0

                        can_prune_best = (
                            removed_count_for_prune > 0
                            and new_seed_count_for_prune >= int(cfg.min_active_seeds or 1)
                            and old_seed_count_for_prune == old_seed_count_current
                        )
                        if not can_prune_best:
                            removed_count = removed_count_for_prune
                            pruned = False
                        else:
                            ppnet.load_state_dict(prune_best_ppnet_state)
                            pruned, uv_anchor_new, old_seed_count, removed_count = self._prune_inactive_seeds(
                                ppnet=ppnet,
                                decoder=decoder,
                                uv_anchor=prune_best_uv_anchor,
                                pred_i=prune_best_pred[0],
                            )
                        if pruned:
                            pruned_from_best_step = int(prune_best_step)
                            uv_anchor = uv_anchor_new
                            cfg.seed_number = int(getattr(ppnet, "n_seeds", uv_anchor.shape[0]))
                            uv_init_list = [uv_anchor.detach().clone()]
                            opt_new = self._build_optimizer(ppnet, decoder)
                            self._copy_optimizer_lrs(opt, opt_new)
                            opt = opt_new
                            remaining_milestones = [
                                max(1, int(m) - int(step))
                                for m in milestones
                                if int(m) > int(step)
                            ]
                            scheduler = self._build_scheduler(opt, sorted(set(remaining_milestones)))
                            seeds0 = None
                            prune_best_score = float("inf")
                            prune_best_step = -1
                            prune_best_pred = None
                            prune_best_uv_anchor = None
                            prune_best_ppnet_state = None
                            steps_since_improve = 0
                            prune_events.append({
                                "step": int(step),
                                "pruned_from_best_step": pruned_from_best_step,
                                "old_seed_count": int(old_seed_count),
                                "new_seed_count": int(cfg.seed_number),
                                "removed_count": int(removed_count),
                            })
                            tqdm.write(
                                f"Pruned inactive seeds at step {step}: "
                                f"{old_seed_count} -> {cfg.seed_number} "
                                f"(removed {removed_count}) using best segment step "
                                f"{prune_events[-1]['pruned_from_best_step']}. Continuing training; "
                                f"global best remains step {best_step}."
                            )
                            continue
                        if removed_count > 0:
                            tqdm.write(
                                f"Plateau reached at step {step}, but pruning was skipped: "
                                f"removing {removed_count} inactive seeds would leave fewer than "
                                f"min_active_seeds={int(cfg.min_active_seeds or 1)}."
                            )
                        elif old_seed_count_current <= int(cfg.min_active_seeds or 1):
                            tqdm.write(
                                f"Plateau reached at step {step}, but only "
                                f"{old_seed_count_current} seed slots remain."
                            )

                    if step >= self.early_stop_start_step() and steps_since_improve >= cfg.patience:
                        tqdm.write(
                            f"Early stopping at step {step} | "
                            f"best_step={best_step} | best_score={best_score:.6f} |"
                        )
                        break

        # ------------------------------------------------------------
        # Fallback best state
        # ------------------------------------------------------------
        if best_rho is None:
            with torch.no_grad():
                best_rho = rho.detach().clone()
                best_seeds = [s.detach().clone() for s in seeds_list]
                best_pred = self._clone_pred_list(pred_list)
                best_step = step
                best_score = float("inf") if not self._scalar_tensor_is_finite(L_total) else float(L_total.detach().item())

                if best_vol_frac is None:
                    best_vol_frac = float(vol_frac_eff.detach().item())
                if best_comp is None:
                    best_comp = float(comp_val.detach().item())
                if best_w_geo is None:
                    best_w_geo = float(w_geo_mean.detach().item())
                if best_active_count is None:
                    best_active_count = float(participating_count_total)
                if best_inactive_count is None:
                    best_inactive_count = float(inactive_count_total)

        use_hard_result = False
        returned_best_source = "global"
        if use_hard_result:
            best_score = best_hard_score
            best_step = best_hard_step
            best_vol_frac = best_hard_vol_frac
            best_comp = best_hard_comp
            best_w_geo = best_hard_w_geo
            best_active_count = best_hard_active_count
            best_inactive_count = best_hard_inactive_count
            best_rho = best_hard_rho
            best_fiber_surface = best_hard_fiber_surface
            best_seeds = best_hard_seeds
            best_pred = best_hard_pred
            returned_best_source = "hard"

        # ------------------------------------------------------------
        # Final outputs
        # ------------------------------------------------------------
        with torch.no_grad():
            hard_rho_acc = torch.zeros((vertices_number,), dtype=dtype, device=device)
            hard_rho_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)
            hard_fiber_acc = torch.zeros((vertices_number, 3), dtype=dtype, device=device)
            hard_fiber_wgt = torch.zeros((vertices_number,), dtype=dtype, device=device)
            pred_i = best_pred[0] if best_pred else None
            if pred_i is not None:
                decoder_seed_state = self._decoder_seed_state_for_pred(decoder, pred_i, device)
                local_face_id = torch.zeros(face_tensor["uv"].shape[0], dtype=torch.long, device=device)
                boundary_uv_i = None
                boundary_face_id_i = None
                boundary_loop_id_i = None
                true_bidx_i, boundary_loop_id_i = self._ordered_true_open_boundary(face_tensor)
                if true_bidx_i.numel() > 0:
                    boundary_uv_i = face_tensor["uv"][true_bidx_i]
                    boundary_face_id_i = torch.zeros(
                        boundary_uv_i.shape[0],
                        dtype=torch.long,
                        device=device,
                    )
                try:
                    hard_out_i = decoder(
                        seeds_uv=pred_i["seeds_raw"],
                        w_raw=pred_i["w_raw"],
                        generate_density_fiber=getattr(cfg, "generate_decoder_density_fiber", True),
                    )
                finally:
                    self._restore_decoder_seed_state(decoder, decoder_seed_state)

                if getattr(cfg, "generate_decoder_density_fiber", True):
                    hard_out_i = apply_density_postprocess_to_output(
                        hard_out_i,
                        face_tensor,
                        cfg,
                        return_debug=False,
                    )
                    hard_rho_i = hard_out_i["rho"]
                    hard_fiber_i = hard_out_i["fiber3d"]
                else:
                    hard_rho_i, hard_fiber_i, _ = self.neutral_density_fiber_fields(
                        face_tensor["uv"],
                        face_tensor.get("Xu", None),
                    )

                w_local = A_local.clamp_min(cfg.eps)
                hard_rho_acc[gidx] += hard_rho_i * w_local
                hard_rho_wgt[gidx] += w_local
                hard_fiber_acc[gidx] += hard_fiber_i * w_local[:, None]
                hard_fiber_wgt[gidx] += w_local

            final_shape_density = hard_rho_acc / hard_rho_wgt.clamp_min(cfg.eps)
            final_shape_fiber_direction = hard_fiber_acc / hard_fiber_wgt.clamp_min(cfg.eps)[:, None]
            final_fiber_norm = final_shape_fiber_direction.norm(dim=1, keepdim=True)
            final_shape_fiber_direction = torch.where(
                final_fiber_norm > cfg.eps,
                final_shape_fiber_direction / final_fiber_norm.clamp_min(cfg.eps),
                torch.zeros_like(final_shape_fiber_direction),
            )
            seed_points_final = self._seed_points_xyz(best_seeds[0], face_tensor)
            seed_points_final_uv = best_seeds[0].detach().clone()

            if mid_shape_density is None:
                mid_shape_density = final_shape_density.clone()
                seed_points_mid_uv = seed_points_final_uv
                seed_points_mid = seed_points_final

        computation_time_sec = time.perf_counter() - train_start_time
        final_centerline_radius = float("nan")
        if history and best_step >= 0:
            for hist_row in reversed(history):
                if int(hist_row["step"]) == int(best_step):
                    final_centerline_radius = float(hist_row.get("centerline_radius_mean", float("nan")))
                    break

        tqdm.write(
            f"FINAL RETURNED: best_step={best_step}, best_score={best_score:.6f} | "
            f"VF_eff_int={best_vol_frac:.3e}, "
            f"comp={best_comp:.3e}, w_geo={best_w_geo:.3e}, "
            f"centerline_radius={final_centerline_radius:.3e} | "
            f"active={float(best_active_count or 0.0):.0f}, inactive={float(best_inactive_count or 0.0):.0f} | "
            f"source={returned_best_source} | "
            f"time={self._format_elapsed_time(computation_time_sec)}"
        )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        best_row = None
        if history and best_step >= 0:
            for row in reversed(history):
                if int(row["step"]) == int(best_step):
                    best_row = row
                    break

        if best_row is not None:
            tqdm.write(
                "BEST VOLUME METRICS: "
                f"VF_total={best_row['VF_total']:.6g} | "
                f"VF_eff_total={best_row['VF_eff_total']:.6g} | "
                f"VF_int={best_row['VF_int']:.6g} | "
                f"VF_eff_int={best_row['VF_eff_int']:.6g}"
            )

        optimization_log_dir = None
        try:
            optimization_log_dir = self._save_optimization_logs(
                output_folder=timelapse_output_folder or getattr(cfg, "timelapse_output_folder", None),
                history=history,
                best_row=best_row,
                best_score=best_score,
                best_step=best_step,
                computation_time_sec=computation_time_sec,
                returned_best_source=returned_best_source,
            )
            if optimization_log_dir is not None:
                tqdm.write(f"Saved optimization logs: {optimization_log_dir}")
        except Exception as e:
            tqdm.write(f"Failed to save optimization logs: {e}")

        if cfg.MakeTimelaps:
            try:
                total_seed_slots = (
                    int(best_pred[0]["seeds_raw"].shape[0])
                    if best_pred
                    else int(cfg.seed_number)
                )
                active_seed_count = int(round(float(best_active_count or 0.0)))
                best_vol_total = (
                    float(best_row["vol_frac"])
                    if best_row is not None and "vol_frac" in best_row
                    else float(best_vol_frac)
                )
                best_vol_internal = (
                    float(best_row["vol_frac_internal"])
                    if best_row is not None and "vol_frac_internal" in best_row
                    else float(best_vol_frac)
                )
                best_vol_eff = float(best_vol_frac)
                best_vol_eff_total = (
                    float(best_row["VF_eff_total"])
                    if best_row is not None and "VF_eff_total" in best_row
                    else float("nan")
                )
                tuned_param_summary = {
                    "best_step": f"{int(best_step)}",
                    "active_seeds": f"{active_seed_count}/{total_seed_slots}",
                    "w": f"{float(best_w_geo):.6g}",
                    "tau": f"{self._fallback_tau_value():.6g}",
                }
                if best_pred:
                    def _mean_from_best_pred(key):
                        vals = []
                        for p in best_pred:
                            v = p.get(key)
                            if isinstance(v, torch.Tensor):
                                vals.append(float(v.detach().mean().item()))
                        if vals:
                            return float(sum(vals) / len(vals))
                        return float("nan")

                    tau_mean = _mean_from_best_pred("tau")
                    h_mean = _mean_from_best_pred("h")

                    tuned_param_summary = {
                        "best_step": f"{int(best_step)}",
                        "active_seeds": f"{active_seed_count}/{total_seed_slots}",
                        "w": f"{float(best_w_geo):.6g}",
                        "tau": (
                            f"{(tau_mean if math.isfinite(tau_mean) else self._fallback_tau_value()):.6g}"
                        ),
                        "h": f"{h_mean:.6g}" if math.isfinite(h_mean) else "nan",
                    }

                best_loss_dict = {
                    "L_Total": float(best_score),
                    "L_Volume": float(best_row["loss_vol"]) if best_row is not None else float("nan"),
                    "L_FEM": float(best_row["loss_fem"]) if best_row is not None else float("nan"),
                    "L_Bnd": float(best_row["loss_bnd"]) if best_row is not None else float("nan"),
                    "L_Rep": float(best_row["loss_rep"]) if best_row is not None else float("nan"),
                    "L_CurveLen": float(best_row["loss_curve_length"]) if best_row is not None else float("nan"),
                    "L_CellEdge": float(best_row["loss_cell_edge_uniform"]) if best_row is not None else float("nan"),
                }
                results_text = (
                    f"VF_total={best_vol_total:.6g} | "
                    f"VF_eff_total={best_vol_eff_total:.6g} | "
                    f"VF_int={best_vol_internal:.6g} | "
                    f"VF_eff_int={best_vol_eff:.6g} | "
                    f"fem={float(best_comp):.6g} | "
                    f"compute_time={self._format_elapsed_time(computation_time_sec)}"
                )
                tuned_param_title = " | ".join(f"{key}={value}" for key, value in tuned_param_summary.items())

                decoder_seed_state = None
                if best_pred:
                    decoder_seed_state = self._decoder_seed_state_for_pred(decoder, best_pred[0], device)
                try:
                    if getattr(cfg, "timelapse_show_3d_tubes", True):
                        best_cad_img = self._render_current_3d_tube_frame_cached(
                            seeds_list=best_seeds,
                            decoders=decoders,
                            pred_list=best_pred,
                            render_cache=render_cache,
                            loading_img=self.timelapse_loading_img,
                        )
                    else:
                        best_cad_img = self._render_current_cad_frame_cached(
                            seeds_list=best_seeds,
                            decoders=decoders,
                            pred_list=best_pred,
                            render_cache=render_cache,
                            thr=getattr(cfg, "vis_thr", cfg.TM_laps_Thr),
                            loading_img=self.timelapse_loading_img,
                        )
                finally:
                    if decoder_seed_state is not None:
                        self._restore_decoder_seed_state(decoder, decoder_seed_state)
                best_frame_path = recorder.add_frame(
                    step=cfg.num_steps + 1,
                    cad_img=best_cad_img,
                    loss_dict=best_loss_dict,
                    title_text=tuned_param_title,
                    highlight_best=True,
                    chart_title="Best Result Losses",
                    summary_title="Tuned Parameters",
                    prefix_step_in_summary=False,
                    results_title="Results",
                    results_text=results_text,
                )
                if timelapse_output_folder:
                    shutil.copy2(
                        best_frame_path,
                        os.path.join(timelapse_output_folder, "best_result_frame.png"),
                    )
                recorder.build_video(hold_last_seconds=10.0)
            except Exception as e:
                tqdm.write(f"Failed to build timelapse video: {e}")

        optimized_function_path = None
        optimized_function_dir = timelapse_output_folder
        if optimized_function_dir is None:
            cfg_output_folder = getattr(cfg, "timelapse_output_folder", None)
            if cfg_output_folder:
                optimized_function_dir = os.path.normpath(str(cfg_output_folder))

        if optimized_function_dir is not None:
            try:
                optimized_function_path = self._save_optimized_shell_function(
                    save_dir=optimized_function_dir,
                    decoder=decoder,
                    ppnet=ppnet,
                    face_tensor=face_tensor,
                    best_pred=best_pred[0] if best_pred else None,
                    best_score=best_score,
                    best_step=best_step,
                    returned_best_source=returned_best_source,
                    final_shape_density=final_shape_density,
                    final_shape_fiber_direction=final_shape_fiber_direction,
                )
                if optimized_function_path is not None:
                    tqdm.write(f"Saved optimized shell function: {optimized_function_path}")
            except Exception as e:
                tqdm.write(f"Failed to save optimized shell function: {e}")

        if debug_anomaly_detection:
            torch.autograd.set_detect_anomaly(False)

        return {
            "decoders": decoders,
            "ppnets": ppnets,
            "optimizer": opt,
            "history": history,
            "prune_events": prune_events,
            "best_score": best_score,
            "best_step": best_step,
            "best_active_count": float(best_active_count or 0.0),
            "best_inactive_count": float(best_inactive_count or 0.0),
            "best_hard_score": best_hard_score,
            "best_hard_step": best_hard_step,
            "best_hard_active_count": float(best_hard_active_count or 0.0),
            "best_hard_inactive_count": float(best_hard_inactive_count or 0.0),
            "returned_best_source": returned_best_source,
            "best_rho": best_rho,
            "best_seeds": best_seeds,
            "best_pred": best_pred,
            "Initial_shape_density": initial_shape_density,
            "Mid_shape_density": mid_shape_density,
            "Final_shape_density": final_shape_density,
            "Final_shape_fiber_direction": final_shape_fiber_direction,
            "seed_points_init": seed_points_init,
            "seed_points_mid": seed_points_mid,
            "seed_points_final": seed_points_final,
            "seed_points_init_uv": seed_points_init_uv,
            "seed_points_mid_uv": seed_points_mid_uv,
            "seed_points_final_uv": seed_points_final_uv,
            "best_seed_points_uv": seed_points_final_uv,
            "best_seed_points_xyz": seed_points_final,
            "best_edge_curves_uv": best_pred[0].get("edge_curves_uv") if best_pred else None,
            "best_edge_curves_xyz": best_pred[0].get("edge_curves_xyz") if best_pred else None,
            "A_v": A_v,
            "uv_init_list": uv_init_list,
            "uv_anchor_list": [uv_anchor],
            "face_tensors": face_tensors,
            "fem_debug_history": self.fem_debug_history,
            "last_fem_debug": self.last_fem_debug,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "optimization_log_dir": optimization_log_dir,
            "shape_path": shape_path,
            "optimized_function_path": optimized_function_path,
        }
