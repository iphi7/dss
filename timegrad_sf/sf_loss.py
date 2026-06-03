"""
sf_loss.py – Differentiable Stylized Fact alignment losses (SFAG-style)

Reference: Zhang et al. "Beyond Visual Realism: Toward Reliable Financial
           Time Series Generation" (arXiv:2601.12990, 2026)

All functions operate on **normalized** return sequences. Since ACF and
Pearson correlation are scale-invariant, normalization does not affect the
statistics relative to the reference sequence.

Input convention: seq shape = (B, T, 2)
  col 0 = sp500 (daily return, normalized)
  col 1 = DGS10 (daily change, normalized)
"""

import torch


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _rolling_std(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Differentiable rolling standard deviation.
    x       : (B, T)
    returns : (B, T - window + 1)
    """
    wins = x.unfold(-1, window, 1)          # (B, T-W+1, W)
    return wins.std(dim=-1, unbiased=False) + 1e-8


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation per sample.
    a, b : (B, T)
    returns: (B,)
    """
    a_c = a - a.mean(-1, keepdim=True)
    b_c = b - b.mean(-1, keepdim=True)
    cov  = (a_c * b_c).mean(-1)
    denom = (a_c.std(-1, unbiased=False) + 1e-8) * (b_c.std(-1, unbiased=False) + 1e-8)
    return cov / denom


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def acf_sq_loss(pred: torch.Tensor, real: torch.Tensor,
                max_lag: int = 20) -> torch.Tensor:
    """
    L_ACF : MSE between lag-k ACF of r² at lags 1..max_lag.

    Computed for both dimensions (sp500 and DGS10) and averaged.

    pred, real : (B, T, 2)
    """
    loss = pred.new_zeros(1).squeeze()
    for dim in range(pred.shape[-1]):
        r2_p = pred[:, :, dim] ** 2   # (B, T)
        r2_r = real[:, :, dim] ** 2
        for k in range(1, max_lag + 1):
            rho_p = _pearson_corr(r2_p[:, :-k], r2_p[:, k:]).mean()
            rho_r = _pearson_corr(r2_r[:, :-k], r2_r[:, k:]).mean()
            loss = loss + (rho_p - rho_r) ** 2
    return loss / (max_lag * pred.shape[-1])


def leverage_loss(pred: torch.Tensor, real: torch.Tensor,
                  window: int = 20) -> torch.Tensor:
    """
    L_Lev : |corr(r_t, σ_{t+1:t+W}) − corr(r̂_t, σ̂_{t+1:t+W})|

    Computed on sp500 (equity leverage effect).
    σ is the realized vol of the W days immediately following r_t.

    pred, real : (B, T, 2)
    """
    sp_p = pred[:, :, 0]   # (B, T)
    sp_r = real[:, :, 0]
    T = sp_p.shape[1]

    if T <= window:
        return pred.new_zeros(1).squeeze()

    # vol[t] = std(r[t+1 : t+W+1])  → use sp[1:] with rolling window W
    vol_p = _rolling_std(sp_p[:, 1:], window)   # (B, T-W)
    vol_r = _rolling_std(sp_r[:, 1:], window)

    n = vol_p.shape[1]
    r_p = sp_p[:, :n]   # r[t] aligned with vol[t]
    r_r = sp_r[:, :n]

    corr_p = _pearson_corr(r_p, vol_p).mean()
    corr_r = _pearson_corr(r_r, vol_r).mean()
    return (corr_p - corr_r).abs()


def cfvc_loss(pred: torch.Tensor, real: torch.Tensor,
              windows: tuple = (5, 20, 60, 120)) -> torch.Tensor:
    """
    L_CFVC : Frobenius-norm gap between pairwise correlation matrices of
             realized volatility at multiple time scales.

    Computed on sp500 (col 0).

    pred, real : (B, T, 2)
    """
    sp_p = pred[:, :, 0]   # (B, T)
    sp_r = real[:, :, 0]
    T = sp_p.shape[1]

    max_w = max(windows)
    if T <= max_w:
        return pred.new_zeros(1).squeeze()

    def _vol_stack(x: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, n_windows, T_min) rolling vols aligned to shortest."""
        vols = [_rolling_std(x, w) for w in windows]
        t_min = min(v.shape[1] for v in vols)
        return torch.stack([v[:, -t_min:] for v in vols], dim=1)   # (B, n_win, T_min)

    def _corr_matrix(vm: torch.Tensor) -> torch.Tensor:
        """vm: (B, n_win, T) → (B, n_win, n_win) Pearson corr matrix."""
        vm_c = vm - vm.mean(-1, keepdim=True)
        std   = vm_c.std(-1, keepdim=True, unbiased=False) + 1e-8
        vm_n  = vm_c / std                                        # (B, n_win, T)
        return torch.bmm(vm_n, vm_n.transpose(1, 2)) / vm_n.shape[-1]

    cm_p = _corr_matrix(_vol_stack(sp_p))   # (B, n_win, n_win)
    cm_r = _corr_matrix(_vol_stack(sp_r))

    return ((cm_p - cm_r) ** 2).sum(dim=(-1, -2)).sqrt().mean()


# ---------------------------------------------------------------------------
# Kurtosis loss
# ---------------------------------------------------------------------------

def kurtosis_loss(pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    L_kurt : |γ₂(r_pred) − γ₂(r_real)|

    Differentiable Pearson kurtosis (γ₂ = μ₄ / σ⁴).
    Computed on sp500 (col 0) over the full sequence (ctx + pred).

    Note: ctx is identical in both pred and real (real data), so the
    gradient signal flows through the last T_pred generated steps only.

    pred, real : (B, T, 2)
    """
    def _kurtosis(x: torch.Tensor) -> torch.Tensor:
        """x : (B, T) → (B,) Pearson kurtosis per sample."""
        mu  = x.mean(-1, keepdim=True)
        x_c = x - mu
        var = (x_c ** 2).mean(-1) + 1e-8
        m4  = (x_c ** 4).mean(-1)
        return m4 / (var ** 2)

    kurt_p = _kurtosis(pred[:, :, 0]).mean()
    kurt_r = _kurtosis(real[:, :, 0]).mean()
    return (kurt_p - kurt_r).abs()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def compute_sf_loss(pred: torch.Tensor, real: torch.Tensor,
                    w_acf: float = 1.0, w_lev: float = 1.0,
                    w_cfvc: float = 1.0, w_kurt: float = 0.0) -> tuple:
    """
    Combined stylized-fact alignment loss.

    pred, real : (B, T_ctx + T_pred, 2)  — context is shared (real data);
                 last T_pred steps differ (generated vs ground-truth).

    Returns (total_sf_loss, l_acf, l_lev, l_cfvc, l_kurt) for logging.
    """
    l_acf  = acf_sq_loss(pred, real)
    l_lev  = leverage_loss(pred, real)
    l_cfvc = cfvc_loss(pred, real)
    l_kurt = kurtosis_loss(pred, real) if w_kurt > 0 else pred.new_zeros(1).squeeze()
    total  = w_acf * l_acf + w_lev * l_lev + w_cfvc * l_cfvc + w_kurt * l_kurt
    return total, l_acf, l_lev, l_cfvc, l_kurt
