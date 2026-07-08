from __future__ import annotations

import numpy as np
import pandas as pd


def _acf(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag + 2:
        return float("nan")
    a = x[:-lag] - x[:-lag].mean()
    b = x[lag:] - x[lag:].mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def _kurtosis_pearson(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    centered = x - x.mean()
    var = np.mean(centered**2)
    return float(np.mean(centered**4) / (var**2)) if var > 0 else float("nan")


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    centered = x - x.mean()
    std = np.std(centered)
    return float(np.mean((centered / std) ** 3)) if std > 0 else float("nan")


def summarize_stylized_facts(df: pd.DataFrame, label: str) -> dict:
    sp = df["sp500"].astype(float).to_numpy()
    dg = df["DGS10"].astype(float).to_numpy()

    lev_vals = []
    for lag in range(1, 21):
        if len(sp) > lag + 2:
            lev_vals.append(_acf(np.r_[sp[:-lag], np.full(lag, np.nan)] * 0 + sp, lag))
    # Standard leverage proxy: Corr(r_t, r_{t+lag}^2), averaged over lag 1..20.
    leverage = []
    for lag in range(1, 21):
        if len(sp) > lag + 2:
            a = sp[:-lag]
            b = sp[lag:] ** 2
            if np.std(a) > 0 and np.std(b) > 0:
                leverage.append(float(np.corrcoef(a, b)[0, 1]))

    out = {
        "label": label,
        "n": len(df),
        "mean_sp500": float(np.mean(sp)),
        "std_sp500": float(np.std(sp)),
        "skew_sp500": _skew(sp),
        "kurt_sp500": _kurtosis_pearson(sp),
        "absacf1_sp500": _acf(np.abs(sp), 1),
        "absacf5_sp500": _acf(np.abs(sp), 5),
        "sqacf1_sp500": _acf(sp**2, 1),
        "leverage_sp500_lag1_20": float(np.nanmean(leverage)) if leverage else float("nan"),
        "mean_DGS10": float(np.mean(dg)),
        "std_DGS10": float(np.std(dg)),
        "skew_DGS10": _skew(dg),
        "kurt_DGS10": _kurtosis_pearson(dg),
        "absacf1_DGS10": _acf(np.abs(dg), 1),
        "sp_dgs10_corr": float(np.corrcoef(sp, dg)[0, 1]) if np.std(sp) > 0 and np.std(dg) > 0 else float("nan"),
    }
    return out

