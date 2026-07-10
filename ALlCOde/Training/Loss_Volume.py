import torch


class Loss_Volume:
    def __call__(
        self,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return self.constant_height(
            rho=rho,
            A_v=A_v,
            target_volfrac=target_volfrac,
            eps=eps,
        )

    @staticmethod
    def constant_height(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        vol_frac = (rho * A_v).sum() / (A_v.sum() + eps)
        target = torch.as_tensor(target_volfrac, device=rho.device, dtype=rho.dtype)
        return ((vol_frac - target) / target.clamp_min(eps)) ** 2

    @staticmethod
    def powered_fraction(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        rho_eff = rho.clamp(0.0, 1.0).pow(power)
        return (rho_eff * A_v).sum() / (A_v.sum() + eps)

    def powered(
        self,
        rho: torch.Tensor,
        A_v: torch.Tensor,
        target_volfrac: float,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vol_frac_eff = self.powered_fraction(rho=rho, A_v=A_v, power=power, eps=eps)
        target = torch.as_tensor(target_volfrac, device=rho.device, dtype=rho.dtype)
        return ((vol_frac_eff - target) / target.clamp_min(eps)) ** 2, vol_frac_eff

    @staticmethod
    def with_boundary_discount(
        rho: torch.Tensor,
        A_v: torch.Tensor,
        rho_boundary: torch.Tensor,
        target_volfrac: float,
        boundary_weight: float = 0.20,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = 1.0 - rho_boundary + boundary_weight * rho_boundary
        vol_frac_eff = (rho * weight * A_v).sum() / ((weight * A_v).sum() + eps)
        target = torch.as_tensor(target_volfrac, device=rho.device, dtype=rho.dtype)
        return ((vol_frac_eff - target) / target.clamp_min(eps)) ** 2, vol_frac_eff
