import torch


def build_mesh_edges(faces):
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]

    edges = torch.cat([e01, e12, e20], dim=0)
    edges = torch.cat([edges, edges.flip(1)], dim=0)

    return edges[:, 0], edges[:, 1]

def smooth_heaviside_projection(
    rho,
    beta=8.0,
    eta=0.5,
    strength=1.0,
    eps=1e-8,
    debug=False,
):
    if strength <= 0.0:
        return rho

    beta_t = torch.as_tensor(beta, device=rho.device, dtype=rho.dtype).clamp_min(eps)
    eta_t = torch.as_tensor(eta, device=rho.device, dtype=rho.dtype).clamp(eps, 1.0 - eps)
    strength_t = torch.as_tensor(strength, device=rho.device, dtype=rho.dtype).clamp(0.0, 1.0)

    num = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (rho - eta_t))
    den = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (1.0 - eta_t))

    rho_proj = (num / den.clamp_min(eps)).clamp(0.0, 1.0)
    result = ((1.0 - strength_t) * rho + strength_t * rho_proj).clamp(0.0, 1.0)

    if debug:
        delta = (result - rho).abs()
        print(
            f"Projection Δrho mean={delta.mean().item():.4e} "
            f"max={delta.max().item():.4e} "
            f"rho_mean={rho.mean().item():.4f} "
            f"rho_proj_mean={result.mean().item():.4f}"
        )

    return result


def surface_density_filter(
    rho,
    points_xyz,
    faces,
    radius,
    self_weight=1.0,
    eps=1e-8,
):
    """
    Differentiable 3D surface graph density filter.

    This smooths the density field on the 3D shell surface.
    It is differentiable, but it does not directly enforce uniform strut width.
    """
    src, dst = build_mesh_edges(faces)

    xi = points_xyz[dst]
    xj = points_xyz[src]

    d2 = ((xi - xj) ** 2).sum(dim=-1)
    weights = torch.exp(-d2 / (2.0 * radius ** 2))

    num = torch.zeros_like(rho)
    den = torch.zeros_like(rho)

    num.index_add_(0, dst, weights * rho[src])
    den.index_add_(0, dst, weights)

    num = num + self_weight * rho
    den = den + self_weight

    return num / den.clamp_min(eps)


def surface_density_filter_metric_aware(
    rho,
    points_xyz,
    faces,
    Xu,
    Xv,
    base_radius,
    self_weight=1.0,
    eps=1e-8,
):
    src, dst = build_mesh_edges(faces)

    xi = points_xyz[dst]
    xj = points_xyz[src]

    d2 = ((xi - xj) ** 2).sum(dim=-1)

    Xu_n = torch.linalg.norm(Xu, dim=1).clamp_min(eps)
    Xv_n = torch.linalg.norm(Xv, dim=1).clamp_min(eps)

    local_scale = torch.sqrt(Xu_n * Xv_n).clamp_min(eps)

    # normalize so average radius stays close to base_radius
    local_scale = local_scale / local_scale.mean().clamp_min(eps)

    r_i = base_radius * local_scale[dst]
    r_j = base_radius * local_scale[src]
    r_ij = 0.5 * (r_i + r_j)

    weights = torch.exp(-d2 / (2.0 * r_ij.pow(2).clamp_min(eps)))

    num = torch.zeros_like(rho)
    den = torch.zeros_like(rho)

    num.index_add_(0, dst, weights * rho[src])
    den.index_add_(0, dst, weights)

    num = num + self_weight * rho
    den = den + self_weight

    return num / den.clamp_min(eps)
