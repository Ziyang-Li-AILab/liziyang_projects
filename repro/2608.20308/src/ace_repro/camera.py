"""Camera geometry for the ray-based solver (paper §3.3, §3.5; gaps_filled C/H).

Conventions
-----------
* Intrinsics are a (..., 6) tensor ``[fx, fy, cx, cy, W, H]`` in pixels of the
  *encode* resolution, the same layout the released projector consumes.
* The token grid has ``Ht x Wt`` cells; cell centres sit at
  ``u = (j + 0.5) * W / Wt`` (paper §3.3 "one unit ray per token-grid cell").
* Fitted cameras are kept in *normalised* units (image width/height = 1) so the
  fit is resolution independent; ``fit_to_intrinsics`` converts back to pixels.
"""
from __future__ import annotations

import torch

# Guards of the closed-form fit (paper App. B.1: "variance floors, a focal
# bracket, a principal point inside the image and forward-facing rays"). The
# numbers are [guess]: the paper gives the guard types but not the thresholds.
FIT_VAR_FLOOR = 1e-6          # variance of the bearing coordinate across the grid
FIT_FOCAL_BRACKET = (0.35, 3.0)  # f / image_size
FIT_PRINCIPAL_MARGIN = 0.0    # principal point must lie inside [0, 1]
FIT_MIN_FORWARD_FRAC = 0.9    # fraction of rays with rz > 0 required


def cell_centres(Ht: int, Wt: int, device=None, dtype=torch.float32):
    """Normalised cell-centre coordinates ``(u, v)`` each of shape (Ht, Wt)."""
    us = (torch.arange(Wt, device=device, dtype=dtype) + 0.5) / Wt
    vs = (torch.arange(Ht, device=device, dtype=dtype) + 0.5) / Ht
    v, u = torch.meshgrid(vs, us, indexing="ij")
    return u, v


def rays_from_intrinsics(K: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    """Unit viewing rays at the token-grid cell centres. K: (B, 6) -> (B, 3, Ht, Wt)."""
    K = K.float()
    u, v = cell_centres(Ht, Wt, device=K.device)
    u_px = u[None] * K[:, 4, None, None]
    v_px = v[None] * K[:, 5, None, None]
    x = (u_px - K[:, 2, None, None]) / K[:, 0, None, None]
    y = (v_px - K[:, 3, None, None]) / K[:, 1, None, None]
    d = torch.stack([x, y, torch.ones_like(x)], dim=1)
    return d / d.norm(dim=1, keepdim=True)


def bearings_from_intrinsics(K: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    """Calibrated bearings ``((u-cx)/fx, (v-cy)/fy)`` at the cell centres: (B, 2, Ht, Wt)."""
    r = rays_from_intrinsics(K, Ht, Wt)
    return r[:, :2] / r[:, 2:3].clamp_min(1e-6)


def _ols(u: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    """Weighted per-sample OLS of x on u (both (B, N)). Returns slope, intercept, var(x)."""
    wsum = w.sum(-1).clamp_min(1.0)
    mu_u = (w * u).sum(-1) / wsum
    mu_x = (w * x).sum(-1) / wsum
    du, dx = u - mu_u[:, None], x - mu_x[:, None]
    cov = (w * du * dx).sum(-1) / wsum
    var_u = (w * du * du).sum(-1) / wsum
    var_x = (w * dx * dx).sum(-1) / wsum
    slope = cov / var_u.clamp_min(1e-12)
    return slope, mu_x - slope * mu_u, var_x


def fit_pinhole(rays: torch.Tensor) -> dict[str, torch.Tensor]:
    """Closed-form per-axis pinhole fit of a predicted ray field (paper §3.5).

    rays: (B, 3, Ht, Wt) predicted (not necessarily unit) rays.
    Bearing coordinates ``x = rx/rz`` are regressed on the normalised cell
    coordinate ``u``: ``x = u / f_x - c_x / f_x`` gives ``f_x = 1/slope`` and
    ``c_x = -intercept / slope``. Differentiable w.r.t. ``rays``.

    Returns dict with ``f`` (B, 2), ``c`` (B, 2) in normalised image units and
    ``valid`` (B,) bool from the guards.
    """
    B, _, Ht, Wt = rays.shape
    r = rays.float()
    u, v = cell_centres(Ht, Wt, device=r.device)
    rz = r[:, 2]
    fwd = (rz > 1e-4)
    w = fwd.float().reshape(B, -1)
    rz_safe = torch.where(fwd, rz, torch.ones_like(rz) * 1e-4)
    bx = (r[:, 0] / rz_safe).reshape(B, -1)
    by = (r[:, 1] / rz_safe).reshape(B, -1)
    su, iu, vx = _ols(u.reshape(1, -1).expand(B, -1), bx, w)
    sv, iv, vy = _ols(v.reshape(1, -1).expand(B, -1), by, w)
    slope = torch.stack([su, sv], -1).clamp_min(1e-6)       # forward-facing => positive slope
    f = 1.0 / slope
    c = -torch.stack([iu, iv], -1) / slope
    lo, hi = FIT_FOCAL_BRACKET
    valid = (
        (fwd.float().mean(dim=(1, 2)) >= FIT_MIN_FORWARD_FRAC)
        & (vx > FIT_VAR_FLOOR) & (vy > FIT_VAR_FLOOR)
        & (su > 0) & (sv > 0)
        & (f > lo).all(-1) & (f < hi).all(-1)
        & (c > -FIT_PRINCIPAL_MARGIN).all(-1) & (c < 1.0 + FIT_PRINCIPAL_MARGIN).all(-1)
    )
    return {"f": f, "c": c, "valid": valid}


def fit_to_intrinsics(fit: dict[str, torch.Tensor], img_w: float, img_h: float) -> torch.Tensor:
    """Fitted camera -> (B, 6) pixel intrinsics ``[fx, fy, cx, cy, W, H]``."""
    f, c = fit["f"], fit["c"]
    wh = torch.tensor([img_w, img_h], device=f.device, dtype=f.dtype)
    return torch.cat([f * wh, c * wh, wh.expand(f.shape[0], 2)], dim=-1)


def fitted_bearings(fit: dict[str, torch.Tensor], Ht: int, Wt: int) -> torch.Tensor:
    """Bearings of the fitted pinhole at the cell centres: (B, 2, Ht, Wt)."""
    u, v = cell_centres(Ht, Wt, device=fit["f"].device)
    bx = (u[None] - fit["c"][:, 0, None, None]) / fit["f"][:, 0, None, None]
    by = (v[None] - fit["c"][:, 1, None, None]) / fit["f"][:, 1, None, None]
    return torch.stack([bx, by], dim=1)


def fit_loss(fit: dict[str, torch.Tensor], K: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    """L_fit: l1 bearing error between fitted and calibrated cameras over the grid (paper §3.4).

    The gradient reaches the ray field only through the four fitted parameters,
    exactly as the paper states. Rows whose fit failed the guards are excluded.
    """
    pred = fitted_bearings(fit, Ht, Wt)
    gt = bearings_from_intrinsics(K, Ht, Wt)
    per = (pred - gt).abs().mean(dim=(1, 2, 3))          # (B,)
    m = fit["valid"].float()
    return (per * m).sum() / m.sum().clamp_min(1.0)


def intrinsics_to_list(K: torch.Tensor) -> list[dict]:
    """(B, 6) intrinsics -> the list-of-dicts the released projector expects."""
    out = []
    for row in K.detach().cpu().tolist():
        fx, fy, cx, cy, w, h = row
        out.append({"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "image_width": w, "image_height": h})
    return out


def project(points_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole projection to *normalised* [0,1] image coords.

    points_cam: (B, ..., 3); K: (B, 6). Returns (B, ..., 2).
    """
    shape = [K.shape[0]] + [1] * (points_cam.dim() - 2)
    fx, fy, cx, cy, w, h = [K[:, i].reshape(shape) for i in range(6)]
    z = points_cam[..., 2].clamp_min(1e-4)
    u = (fx * points_cam[..., 0] / z + cx) / w
    v = (fy * points_cam[..., 1] / z + cy) / h
    return torch.stack([u, v], dim=-1)
