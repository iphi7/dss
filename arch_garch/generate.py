"""
generate.py – ARCH / GARCH によるパス生成

SP500 と DGS10 にそれぞれ GARCH(1,1) または ARCH(q) をキャリブレーションし、
残差の相関構造を Cholesky 分解で再現しながらパスを生成する。

使い方:
  python arch_garch/generate.py                          # GARCH(1,1)
  python arch_garch/generate.py --model arch --q 5      # ARCH(5)
  python arch_garch/generate.py --n_paths 20 --business_days 252
"""

import argparse
import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


def fit_and_extract(returns: np.ndarray, model_type: str, p: int, q: int):
    """
    ARCH/GARCH をキャリブレーション。
    Returns: params dict, std_resid array
    """
    r = returns * 100  # arch は % 単位
    if model_type == "arch":
        am = arch_model(r, mean="Zero", vol="ARCH", p=q)
    else:
        am = arch_model(r, mean="Zero", vol="GARCH", p=p, q=q)
    res = am.fit(disp="off", show_warning=False)

    params = dict(res.params)
    std_resid = np.asarray(res.std_resid)
    cond_vol  = np.asarray(res.conditional_volatility) / 100
    last_vol  = cond_vol[-1]
    last_r    = returns[-1]

    return params, std_resid, last_vol, last_r, res


def simulate_garch11(omega, alpha, beta, last_var, last_r2,
                     z: np.ndarray) -> np.ndarray:
    """GARCH(1,1) を 1 ステップずつシミュレート。z: (T,) iid N(0,1)"""
    T = len(z)
    sigma2 = np.zeros(T)
    r      = np.zeros(T)
    var_t  = last_var**2
    r2_t   = last_r2**2
    for t in range(T):
        var_t    = omega + alpha * r2_t + beta * var_t
        r[t]     = np.sqrt(var_t) * z[t]
        r2_t     = r[t]**2
        sigma2[t] = var_t
    return r


def simulate_arch(omega, alphas, last_rs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """ARCH(q) シミュレート。alphas: (q,), last_rs: 直近 q 日のリターン"""
    T = len(z)
    q = len(alphas)
    r_hist = list(last_rs[-q:])
    r = np.zeros(T)
    for t in range(T):
        var_t = omega + sum(alphas[i] * r_hist[-(i+1)]**2 for i in range(q))
        r[t]  = np.sqrt(max(var_t, 1e-10)) * z[t]
        r_hist.append(r[t])
    return r


def main(args):
    df = pd.read_csv(args.csv, parse_dates=["Date"]).sort_values("Date").dropna()
    sp_ret = df["sp500"].values.astype(float)
    dg_ret = df["DGS10"].values.astype(float)

    print(f"Fitting {args.model.upper()} to SP500 ...")
    params_sp, std_sp, last_vol_sp, last_r_sp, res_sp = fit_and_extract(
        sp_ret, args.model, args.p, args.q)
    print(res_sp.summary().tables[1])

    print(f"\nFitting {args.model.upper()} to DGS10 ...")
    params_dg, std_dg, last_vol_dg, last_r_dg, res_dg = fit_and_extract(
        dg_ret, args.model, args.p, args.q)
    print(res_dg.summary().tables[1])

    # 標準化残差の相関行列 → Cholesky
    n_min = min(len(std_sp), len(std_dg))
    corr  = np.corrcoef(std_sp[-n_min:], std_dg[-n_min:])
    print(f"\nResidual correlation: {corr[0,1]:.4f}")
    L = np.linalg.cholesky(corr)

    rng = np.random.default_rng(args.seed)
    T   = args.business_days

    # start_date 解決
    if args.start_date is not None:
        mask = df["Date"] >= pd.Timestamp(args.start_date)
        if not mask.any(): raise ValueError(f"start_date={args.start_date} がデータ範囲外")
        start_idx = df[mask].index[0]
        start_idx = df.index.get_loc(start_idx)
        # フィット用データをstart_dateまでに限定
        sp_ret = sp_ret[:start_idx + 1]
        dg_ret = dg_ret[:start_idx + 1]
    else:
        start_idx = len(df) - 1
    rows = []
    last_row  = df.iloc[start_idx]
    last_date = last_row["Date"]
    last_sp   = float(last_row["sp500_abs"])
    last_dgs  = float(last_row["DGS10_abs"])
    bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=T)
    if args.start_date is not None:
        print(f"  開始日: {last_date.date()}  SP500={last_sp:.2f}  DGS10={last_dgs:.4f}")

    print(f"\nSimulating {args.n_paths} paths × {T} days ...")

    for pid in range(args.n_paths):
        # 相関付き標準正規乱数
        z_ind = rng.standard_normal((2, T))  # (2, T)
        z_cor = (L @ z_ind)                  # (2, T)
        z_sp, z_dg = z_cor[0], z_cor[1]

        if args.model == "garch":
            omega_sp = params_sp["omega"] / 1e4  # %² → 小数²
            alpha_sp = params_sp["alpha[1]"]
            beta_sp  = params_sp["beta[1]"]
            r_sp_sim = simulate_garch11(omega_sp, alpha_sp, beta_sp,
                                        last_vol_sp, last_r_sp, z_sp)

            omega_dg = params_dg["omega"] / 1e4
            alpha_dg = params_dg["alpha[1]"]
            beta_dg  = params_dg["beta[1]"]
            r_dg_sim = simulate_garch11(omega_dg, alpha_dg, beta_dg,
                                        last_vol_dg, last_r_dg, z_dg)
        else:
            omega_sp = params_sp["omega"] / 1e4
            alphas_sp = np.array([params_sp[f"alpha[{i+1}]"] for i in range(args.q)])
            r_sp_sim  = simulate_arch(omega_sp, alphas_sp, sp_ret[-args.q:], z_sp)

            omega_dg  = params_dg["omega"] / 1e4
            alphas_dg = np.array([params_dg[f"alpha[{i+1}]"] for i in range(args.q)])
            r_dg_sim  = simulate_arch(omega_dg, alphas_dg, dg_ret[-args.q:], z_dg)

        sp_cur, dg_cur = last_sp, last_dgs
        for t in range(T):
            r_sp = r_sp_sim[t]
            r_dg = r_dg_sim[t]
            sp_cur = sp_cur * (1.0 + r_sp)
            dg_cur = dg_cur + r_dg
            rows.append({
                "path_id":   pid,
                "Date":      bdays[t].strftime("%Y-%m-%d"),
                "sp500_abs": sp_cur,
                "DGS10_abs": dg_cur,
                "sp500":     r_sp,
                "DGS10":     r_dg,
            })

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           default="output.csv")
    parser.add_argument("--model",         default="garch", choices=["garch", "arch"])
    parser.add_argument("--p",             type=int, default=1, help="GARCH p")
    parser.add_argument("--q",             type=int, default=1, help="GARCH/ARCH q")
    parser.add_argument("--n_paths",       type=int, default=20)
    parser.add_argument("--business_days", type=int, default=252)
    parser.add_argument("--start_date",    default=None,
                        help="生成開始日 YYYY-MM-DD。省略時はデータ末尾。")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--out",           default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = f"arch_garch/generated_paths_{args.model}.csv"
    main(args)
