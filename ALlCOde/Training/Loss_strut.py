import torch


class Loss_strut:
    def __call__(
        self,
        rho: torch.Tensor,
        w_soft: torch.Tensor,
        rho_b: torch.Tensor | None = None,
        void_threshold: float = 0.85,
        edge_threshold: float = 0.45,
        temp: float = 0.05,
        rho_edge_min: float = 0.75,
        lam_void: float = 2.0,
        lam_edge: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        loss, _loss_edge, _loss_void, _edge_mask, _void_mask = self.with_components(
            rho=rho,
            w_soft=w_soft,
            rho_b=rho_b,
            void_threshold=void_threshold,
            edge_threshold=edge_threshold,
            temp=temp,
            rho_edge_min=rho_edge_min,
            lam_void=lam_void,
            lam_edge=lam_edge,
            eps=eps,
        )
        return loss

    @staticmethod
    def with_components(
        rho: torch.Tensor,
        w_soft: torch.Tensor,
        rho_b: torch.Tensor | None = None,
        void_threshold: float = 0.85,
        edge_threshold: float = 0.45,
        temp: float = 0.05,
        rho_edge_min: float = 0.75,
        lam_void: float = 2.0,
        lam_edge: float = 1.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = rho.clamp(0.0, 1.0)

        max_w = w_soft.max(dim=1).values
        void_mask = torch.sigmoid((max_w - void_threshold) / (temp + eps))

        ambiguity = (1.0 - w_soft.pow(2).sum(dim=1)).clamp(0.0, 1.0)
        edge_mask = torch.sigmoid((ambiguity - edge_threshold) / (temp + eps))

        if rho_b is not None:
            rho_b = rho_b.to(device=rho.device, dtype=rho.dtype).clamp(0.0, 1.0)
            void_mask = void_mask * (1.0 - rho_b)
            edge_mask = torch.maximum(edge_mask, rho_b)

        loss_void = (void_mask * rho.pow(2)).sum() / (void_mask.sum() + eps)
        loss_edge = (
            edge_mask * torch.relu(rho_edge_min - rho).pow(2)
        ).sum() / (edge_mask.sum() + eps)
        loss = lam_void * loss_void + lam_edge * loss_edge

        return loss, loss_edge, loss_void, edge_mask, void_mask
