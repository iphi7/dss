"""
generate.py – GBM / SABR による金融時系列生成

Usage
-----
# GBM
python classical/generate.py --model gbm --csv output.csv --n_paths 20 --business_days 252

# SABR
python classical/generate.py --model sabr --csv output.csv --n_paths 20 --business_days 252
"""

import argparse
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────────────────

def load_data(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    sp_ret  = df["sp500"].values.astype(np.float64)   # simple return (S_t/S_{t-1} - 1)
    dgs_chg = df["DGS10"].values.astype(np.float64)   # daily change in yield
    sp_log  = np.log1p(sp_ret)                         # log return
    last_sp  = float(df["sp500_abs"].iloc[-1])
    last_dgs = float(df["DGS10_abs"].iloc[-1])
    last_date = df["Date"].iloc[-1]
    return sp_log, dgs_chg, last_sp, last_dgs, last_date


# ─────────────────────────────────────────────────────────────
# GBM
# ─────────────────────────────────────────────────────────────

def calibrate_gbm(sp_log: np.ndarray, dgs_chg: np.ndarray) -> dict:
    """
    Calibrate 2D correlated GBM from historical data.

    SP500: log(S_t/S_{t-1}) = (μ_sp - σ_sp²/2) + σ_sp · ε_1
    DGS10: Δy_t = μ_dgs + σ_dgs · ε_2  (arithmetic BM for interest rate)
    corr(ε_1, ε_2) = ρ
    """
    mu_sp    = float(np.mean(sp_log))
    sigma_sp = float(np.std(sp_log))
    mu_dgs   = float(np.mean(dgs_chg))
    sigma_dgs = float(np.std(dgs_chg))
    rho      = float(np.corrcoef(sp_log, dgs_chg)[0, 1])

    print(f"[GBM] μ_sp={mu_sp:.6f}  σ_sp={sigma_sp:.6f}  "
          f"μ_dgs={mu_dgs:.6f}  σ_dgs={sigma_dgs:.6f}  ρ={rho:.4f}")
    return dict(mu_sp=mu_sp, sigma_sp=sigma_sp,
                mu_dgs=mu_dgs, sigma_dgs=sigma_dgs, rho=rho)


def simulate_gbm(params: dict, n_paths: int, T: int,
                 last_sp: float, last_dgs: float, last_date) -> pd.DataFrame:
    mu_sp     = params["mu_sp"]
    sigma_sp  = params["sigma_sp"]
    mu_dgs    = params["mu_dgs"]
    sigma_dgs = params["sigma_dgs"]
    rho       = params["rho"]

    # Cholesky factor for 2D correlated normals
    L = np.array([[1.0, 0.0],
                  [rho, np.sqrt(max(1.0 - rho**2, 1e-12))]])

    biz_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=T)
    rows = []

    for path_id in range(n_paths):
        sp_cur  = last_sp
        dgs_cur = last_dgs

        Z = np.random.randn(2, T)            # (2, T) i.i.d.
        W = L @ Z                            # (2, T) correlated

        log_r_sp = (mu_sp - 0.5 * sigma_sp**2) + sigma_sp * W[0]
        r_sp  = np.expm1(log_r_sp)           # simple return = e^(log_r) - 1
        r_dgs = mu_dgs + sigma_dgs * W[1]   # arithmetic BM

        for d, rs, rd in zip(biz_dates, r_sp, r_dgs):
            sp_cur  = sp_cur * (1.0 + rs)
            dgs_cur = dgs_cur + rd
            rows.append({"path_id": path_id, "Date": d.strftime("%Y-%m-%d"),
                         "sp500_abs": sp_cur, "DGS10_abs": dgs_cur,
                         "sp500": rs, "DGS10": rd})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# SABR (stochastic alpha, beta, rho)
# ─────────────────────────────────────────────────────────────

def calibrate_sabr(sp_log: np.ndarray, dgs_chg: np.ndarray,
                   vol_window: int = 21) -> dict:
    """
    Calibrate SABR-style stochastic volatility model.

    SP500 (lognormal, β=1):
        log(S_t/S_{t-1}) = (μ_sp - σ_t²/2) + σ_t · ε_1
        σ_t = σ_{t-1} · exp(ν_sp · η_1 - ν_sp²/2)   (lognormal vol process)
        corr(ε_1, η_1) = ρ_sp                          (leverage effect)

    DGS10 (Normal/arithmetic, β=0):
        Δy_t = μ_dgs + σ_dgs_t · ε_2
        σ_dgs_t = σ_dgs_{t-1} · exp(ν_dgs · η_2 - ν_dgs²/2)
        corr(ε_2, η_2) = ρ_dgs

    Cross-asset: corr(ε_1, ε_2) = ρ_cross

    Calibration via:
      σ_0    : std of recent vol_window returns
      ν      : std of consecutive log-vol changes (vol-of-vol)
      ρ_lev  : corr(returns, forward log-vol change)  — leverage effect
      ρ_cross: corr(sp_log, dgs_chg)
    """
    n = len(sp_log)

    # Drift
    mu_sp  = float(np.mean(sp_log))
    mu_dgs = float(np.mean(dgs_chg))

    # Initial vol: std of last vol_window observations
    sigma0_sp  = float(np.std(sp_log[-vol_window:]))
    sigma0_dgs = float(np.std(dgs_chg[-vol_window:]))

    # Rolling vol series
    vol_sp  = np.array([np.std(sp_log[max(0, i-vol_window):i])
                        for i in range(vol_window, n)])
    vol_dgs = np.array([np.std(dgs_chg[max(0, i-vol_window):i])
                        for i in range(vol_window, n)])

    # Vol-of-vol: std of log(σ_t / σ_{t-1})
    lvc_sp  = np.diff(np.log(vol_sp  + 1e-12))
    lvc_dgs = np.diff(np.log(vol_dgs + 1e-12))
    nu_sp   = float(np.std(lvc_sp))
    nu_dgs  = float(np.std(lvc_dgs))

    # Leverage correlation: corr(r_t, Δlog_vol_{t+1})
    L = min(len(sp_log[vol_window:-1]), len(lvc_sp))
    rho_sp  = float(np.corrcoef(sp_log[vol_window:vol_window+L],  lvc_sp[:L])[0, 1])
    rho_dgs = float(np.corrcoef(dgs_chg[vol_window:vol_window+L], lvc_dgs[:L])[0, 1])

    # Cross-asset correlation
    rho_cross = float(np.corrcoef(sp_log, dgs_chg)[0, 1])

    print(f"[SABR] SP500  σ_0={sigma0_sp:.6f}  ν={nu_sp:.4f}  ρ_lev={rho_sp:.4f}")
    print(f"[SABR] DGS10  σ_0={sigma0_dgs:.6f}  ν={nu_dgs:.4f}  ρ_lev={rho_dgs:.4f}")
    print(f"[SABR] ρ_cross={rho_cross:.4f}")

    return dict(mu_sp=mu_sp, sigma0_sp=sigma0_sp, nu_sp=nu_sp, rho_sp=rho_sp,
                mu_dgs=mu_dgs, sigma0_dgs=sigma0_dgs, nu_dgs=nu_dgs, rho_dgs=rho_dgs,
                rho_cross=rho_cross)


def simulate_sabr(params: dict, n_paths: int, T: int,
                  last_sp: float, last_dgs: float, last_date) -> pd.DataFrame:
    mu_sp      = params["mu_sp"]
    sigma0_sp  = params["sigma0_sp"]
    nu_sp      = params["nu_sp"]
    rho_sp     = params["rho_sp"]

    mu_dgs     = params["mu_dgs"]
    sigma0_dgs = params["sigma0_dgs"]
    nu_dgs     = params["nu_dgs"]
    rho_dgs    = params["rho_dgs"]

    rho_cross  = params["rho_cross"]

    # 4-factor noise: [ε_1, ε_2, η_1, η_2]
    # Correlation structure (assuming η_1 ⊥ ε_2, η_2 ⊥ ε_1):
    #   ε_1 ↔ ε_2 : ρ_cross
    #   ε_1 ↔ η_1 : ρ_sp      (leverage)
    #   ε_2 ↔ η_2 : ρ_dgs     (leverage)
    #   others    : 0
    #
    # Build lower Cholesky of 4×4 corr matrix via factor decomposition:
    #   ε_1 = w_1
    #   ε_2 = ρ_cross · w_1 + √(1-ρ_cross²) · w_2
    #   η_1 = ρ_sp    · w_1 + √(1-ρ_sp²)    · w_3
    #   η_2 = ρ_dgs   · √(1-ρ_cross²) · w_2 + ... (requires careful construction)
    #
    # Simplification: treat (ε_2, η_2) as independent of (ε_1, η_1) except via ρ_cross.
    # This is an approximation but keeps the Cholesky valid for all parameter values.

    rc  = rho_cross
    rs  = rho_sp
    rd  = rho_dgs
    rc_ = np.sqrt(max(1.0 - rc**2, 1e-12))
    rs_ = np.sqrt(max(1.0 - rs**2, 1e-12))
    rd_ = np.sqrt(max(1.0 - rd**2, 1e-12))

    biz_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=T)
    rows = []

    for path_id in range(n_paths):
        sp_cur    = last_sp
        dgs_cur   = last_dgs
        sigma_sp  = sigma0_sp
        sigma_dgs = sigma0_dgs

        for d in biz_dates:
            w = np.random.randn(4)         # 4 i.i.d. normals

            eps1 = w[0]                    # SP500 price noise
            eps2 = rc * w[0] + rc_ * w[1] # DGS10 price noise (corr with eps1)
            eta1 = rs * w[0] + rs_ * w[2] # SP500 vol noise (leverage corr with eps1)
            eta2 = rd * w[1] + rd_ * w[3] # DGS10 vol noise (leverage corr with eps2 via w[1])

            # SP500: lognormal SABR (β=1)
            log_r_sp = (mu_sp - 0.5 * sigma_sp**2) + sigma_sp * eps1
            r_sp     = np.expm1(log_r_sp)

            # DGS10: Normal SABR (β=0, arithmetic BM)
            r_dgs = mu_dgs + sigma_dgs * eps2

            # Update stochastic vols (lognormal vol-of-vol)
            sigma_sp  = sigma_sp  * np.exp(nu_sp  * eta1 - 0.5 * nu_sp**2)
            sigma_dgs = sigma_dgs * np.exp(nu_dgs * eta2 - 0.5 * nu_dgs**2)

            sp_cur  = sp_cur  * (1.0 + r_sp)
            dgs_cur = dgs_cur + r_dgs

            rows.append({"path_id": path_id, "Date": d.strftime("%Y-%m-%d"),
                         "sp500_abs": sp_cur, "DGS10_abs": dgs_cur,
                         "sp500": r_sp, "DGS10": r_dgs})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GBM / SABR path generation")
    parser.add_argument("--model",          choices=["gbm", "sabr"], required=True)
    parser.add_argument("--csv",            default="output.csv")
    parser.add_argument("--n_paths",        type=int, default=20)
    parser.add_argument("--business_days",  type=int, default=252)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--out",            default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)

    out_path = args.out or f"classical/generated_paths_{args.model}.csv"

    sp_log, dgs_chg, last_sp, last_dgs, last_date = load_data(args.csv)

    if args.model == "gbm":
        params = calibrate_gbm(sp_log, dgs_chg)
        df_out = simulate_gbm(params, args.n_paths, args.business_days,
                              last_sp, last_dgs, last_date)
    else:
        params = calibrate_sabr(sp_log, dgs_chg)
        df_out = simulate_sabr(params, args.n_paths, args.business_days,
                               last_sp, last_dgs, last_date)

    df_out.to_csv(out_path, index=False)
    print(f"saved {len(df_out)} rows → {out_path}")


if __name__ == "__main__":
    main()
