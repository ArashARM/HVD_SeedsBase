import math
import torch


class Loss_FEM:
    def __init__(self, trainer):
        self.trainer = trainer

    def __call__(
        self,
        rho_surface: torch.Tensor,
        fiber_surface: torch.Tensor,
        comp_normalize_by: float | None = None,
        density_floor: float = 0.02,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> torch.Tensor:
        return self.evaluate(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            comp_normalize_by=comp_normalize_by,
            density_floor=density_floor,
            eps=eps,
            save_debug_history=save_debug_history,
        )["fem_total"]

    @staticmethod
    def compliance(
        comp: torch.Tensor,
        normalize_by: float | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        comp = comp.reshape(())
        if normalize_by is not None:
            return comp / (float(normalize_by) + eps)
        return comp

    @staticmethod
    def _scalar_tensor_is_finite(x: torch.Tensor | float | int) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).reshape(()).detach().item())
        return math.isfinite(float(x))

    def _record_invalid(self, debug: dict, reason: str, save_debug_history: bool):
        self.trainer._record_invalid_fem_debug(debug, reason, save_debug_history)

    def evaluate(
        self,
        rho_surface: torch.Tensor,
        fiber_surface: torch.Tensor,
        comp_normalize_by: float | None = None,
        density_floor: float = 0.02,
        eps: float = 1e-12,
        save_debug_history: bool = True,
    ) -> dict:
        device = rho_surface.device
        dtype = rho_surface.dtype

        fem_fields = self.trainer.shell_problem.build_fem_fields_from_decoder_torch(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
        )
        density_raw = fem_fields["density"].to(device=device, dtype=dtype)
        density = density_raw.clamp_min(density_floor)

        phi = fem_fields["phi"].to(device=device, dtype=dtype)
        theta = fem_fields["theta"].to(device=device, dtype=dtype)

        fiber_norm = torch.linalg.norm(fiber_surface, dim=1)

        debug = {
            "rho_surface_shape": tuple(rho_surface.shape),
            "fiber_surface_shape": tuple(fiber_surface.shape),
            "density_shape": tuple(density.shape),
            "phi_shape": tuple(phi.shape),
            "theta_shape": tuple(theta.shape),
            "density_floor": float(density_floor),
            "density_raw_min": float(density_raw.min().detach().item()),
            "density_raw_mean": float(density_raw.mean().detach().item()),
            "density_raw_max": float(density_raw.max().detach().item()),
            "density_min": float(density.min().detach().item()),
            "density_mean": float(density.mean().detach().item()),
            "density_max": float(density.max().detach().item()),
            "phi_has_nan": bool(torch.isnan(phi).any().detach().item()),
            "phi_has_inf": bool(torch.isinf(phi).any().detach().item()),
            "theta_has_nan": bool(torch.isnan(theta).any().detach().item()),
            "theta_has_inf": bool(torch.isinf(theta).any().detach().item()),
            "fiber_has_nan": bool(torch.isnan(fiber_surface).any().detach().item()),
            "fiber_has_inf": bool(torch.isinf(fiber_surface).any().detach().item()),
            "fiber_norm_min": float(fiber_norm.min().detach().item()),
            "fiber_norm_mean": float(fiber_norm.mean().detach().item()),
            "fiber_norm_max": float(fiber_norm.max().detach().item()),
            "void_fraction_lt_1e_2_raw": float((density_raw < 1e-2).float().mean().detach().item()),
            "void_fraction_lt_5e_2_raw": float((density_raw < 5e-2).float().mean().detach().item()),
            "void_fraction_lt_floor_raw": float((density_raw < density_floor).float().mean().detach().item()),
        }

        if (
            debug["phi_has_nan"] or debug["phi_has_inf"] or
            debug["theta_has_nan"] or debug["theta_has_inf"] or
            debug["fiber_has_nan"] or debug["fiber_has_inf"]
        ):
            reason = "Invalid phi/theta/fiber fields before FEM solve"
            self._record_invalid(debug, reason, save_debug_history)
            nan_scalar = torch.full((), float("nan"), dtype=dtype, device=device)
            return {
                "fem_total": nan_scalar,
                "comp": nan_scalar,
                "compliance_loss": nan_scalar,
                "fem_valid": False,
                "failure_reason": reason,
            }

        try:
            _stress_unused, comp = self.trainer.fem(density, phi, theta, penal=3)
        except Exception as e:
            reason = f"FEM solve raised exception: {repr(e)}"
            self._record_invalid(debug, reason, save_debug_history)
            nan_scalar = torch.full((), float("nan"), dtype=dtype, device=device)
            return {
                "fem_total": nan_scalar,
                "comp": nan_scalar,
                "compliance_loss": nan_scalar,
                "fem_valid": False,
                "failure_reason": reason,
            }

        debug.update({
            "comp_is_finite": self._scalar_tensor_is_finite(comp),
            "comp_value": float(torch.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
        })

        if not debug["comp_is_finite"]:
            reason = "Non-finite compliance returned by FEM solve"
            self._record_invalid(debug, reason, save_debug_history)
            nan_scalar = torch.full((), float("nan"), dtype=dtype, device=device)
            return {
                "fem_total": nan_scalar,
                "comp": comp,
                "compliance_loss": nan_scalar,
                "fem_valid": False,
                "failure_reason": reason,
            }

        loss_comp = self.compliance(comp=comp, normalize_by=comp_normalize_by, eps=eps)

        debug.update({
            "loss_comp_is_finite": self._scalar_tensor_is_finite(loss_comp),
            "fem_total_is_finite": self._scalar_tensor_is_finite(loss_comp),
            "loss_comp_value": float(torch.nan_to_num(loss_comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
            "fem_total_value": float(torch.nan_to_num(loss_comp, nan=0.0, posinf=0.0, neginf=0.0).detach().item()),
        })

        fem_valid = debug["loss_comp_is_finite"]
        debug["fem_valid"] = fem_valid
        debug["failure_reason"] = None if fem_valid else "Non-finite compliance loss"

        self.trainer.last_fem_debug = debug
        if save_debug_history:
            self.trainer.fem_debug_history.append(debug.copy())

        return {
            "fem_total": loss_comp,
            "comp": comp,
            "compliance_loss": loss_comp,
            "fem_valid": fem_valid,
            "failure_reason": debug["failure_reason"],
        }
