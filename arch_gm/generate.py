"""
generate.py – ARCH' (ARCH with Gaussian Mixture noise)

標準 ARCH の i.i.d. N(0,1) ノイズを Gaussian Mixture に置き換えたモデル。

ARCH'(q) の定義:
  r_t = σ_t · ε_t
  σ_t² = ω + Σᵢ αᵢ r²_{t-i}
  ε_t ~ (1-p)·N(0, σ₁²) + p·N(0, σ₂²)  normalized to Var(ε_t) = 1

ε_t の正規化: GM サンプルを sqrt((1-p)σ₁² + pσ₂²) で割って単位分散にする。
σ₁, σ₂ の絶対値は正規化で消えるため、実質的に比率 σ₂/σ₁ と p だけが分布の形を決める。

デフォルトのハイパーパラメータ（実データから導出）:
  p     = 0.013  （|r| > 3σ の観測割合 ≈ 1.36%）
  k     = 0.001  （σ₁ = k, σ₂ = 4.5k）
  → 正規化後: σ₁' ≈ 0.894, σ₂' ≈ 4.024
  → 尖度 γ₂ ≈ 3(1 + (1-p)/p) ≈ 3/p ≈ 3/0.013 ≈ 231（理論値）

  ※ γ₂ が非常に大きく見えるが、これは p=0.013 で σ₂/σ₁=4.5 のときの理論値。
    ARCH の σ_t が時変なため、生成されたリターンの実効尖度はこれより小さくなる。

使い方:
  python arch_gm/generate.py
  python arch_gm/generate.py --p 0.013 --k 0.001 --q 5
"""

import argparse
import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


# ── Gaussian Mixture サンプラー ────────────────────────────────────
def gm_sampler(p: float, sigma1: float, sigma2: float,
               size: int, rng: np.random.Generator) -> np.ndarray:
    """
    (1-p)N(0,σ₁²) + p·N(0,σ₂²) からサンプリングし、単位分散に正規化して返す。
    """
    z = rng.standard_normal(size)
    flag = rng.random(size) < p          # True なら tail コンポーネント
    scale = np.where(flag, sigma2, sigma1)
    raw = z * scale
    var_gm = (1 - p) * sigma1**2 + p * sigma2**2
    return raw / np.sqrt(var_gm)         # Var = 1 に正規化


# ── ARCH フィット ──────────────────────────────────────────────────
def fit_arch(returns: np.ndarray, q: int):
    r = returns * 100
    am = arch_model(r, mean="Zero", vol="ARCH", p=q)
    res = am.fit(disp="off", show_warning=False)
    params = dict(res.params)
    last_rs = returns[-q:]            # シミュレーション開始時の直近 q 日
    return params, last_rs, res


# ── 1系列シミュレーション ──────────────────────────────────────────
def simulate_arch_gm(omega: float, alphas: np.ndarray,
                     last_rs: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    eps: 正規化済み GM ノイズ (T,)
    Returns: r (T,) in original scale
    """
    T = len(eps)
    q = len(alphas)
    r_hist = list(last_rs[-q:])
    r = np.zeros(T)
    for t in range(T):
        sigma2_t = omega + sum(alphas[i] * r_hist[-(i+1)]**2 for i in range(q))
        r[t] = np.sqrt(max(sigma2_t, 1e-12)) * eps[t]
        r_hist.append(r[t])
    return r


def main(args):
    df = pd.read_csv(args.csv, parse_dates=["Date"]).sort_values("Date").dropna()
    sp_ret = df["sp500"].values.astype(float)
    dg_ret = df["DGS10"].values.astype(float)

    print(f"Fitting ARCH({args.q}) to SP500 ...")
    params_sp, last_rs_sp, res_sp = fit_arch(sp_ret, args.q)
    print(res_sp.summary().tables[1])

    print(f"\nFitting ARCH({args.q}) to DGS10 ...")
    params_dg, last_rs_dg, res_dg = fit_arch(dg_ret, args.q)
    print(res_dg.summary().tables[1])

    # ARCH パラメータ抽出（% → 小数²）
    omega_sp  = params_sp["omega"] / 1e4
    alphas_sp = np.array([params_sp[f"alpha[{i+1}]"] for i in range(args.q)])
    omega_dg  = params_dg["omega"] / 1e4
    alphas_dg = np.array([params_dg[f"alpha[{i+1}]"] for i in range(args.q)])

    # Gaussian Mixture パラメータ
    sigma1 = args.k
    sigma2 = args.ratio * args.k
    var_gm = (1 - args.p) * sigma1**2 + args.p * sigma2**2
    sigma1_norm = sigma1 / np.sqrt(var_gm)
    sigma2_norm = sigma2 / np.sqrt(var_gm)
    kurt_eps = 3 * ((1 - args.p) * sigma1_norm**4 + args.p * sigma2_norm**4)

    print(f"\n--- Gaussian Mixture ノイズ ---")
    print(f"  p = {args.p},  σ₁ = {sigma1} (norm: {sigma1_norm:.4f}),  "
          f"σ₂ = {sigma2} (norm: {sigma2_norm:.4f})")
    print(f"  Var(ε) after normalization = 1.0 (by construction)")
    print(f"  γ₂(ε) = {kurt_eps:.2f}")

    # 残差相関（ARCH標準化残差）
    std_sp = np.asarray(res_sp.std_resid)
    std_dg = np.asarray(res_dg.std_resid)
    n_min  = min(len(std_sp), len(std_dg))
    corr   = np.corrcoef(std_sp[-n_min:], std_dg[-n_min:])
    L      = np.linalg.cholesky(corr)
    print(f"  Residual correlation: {corr[0,1]:.4f}")

    rng   = np.random.default_rng(args.seed)
    T     = args.business_days
    # start_date 解決
    if args.start_date is not None:
        mask = df["Date"] >= pd.Timestamp(args.start_date)
        if not mask.any(): raise ValueError(f"start_date={args.start_date} がデータ範囲外")
        start_idx = df[mask].index[0]
        start_idx = df.index.get_loc(start_idx)
        sp_ret = sp_ret[:start_idx + 1]
        dg_ret = dg_ret[:start_idx + 1]
    else:
        start_idx = len(df) - 1
    rows  = []
    last_row  = df.iloc[start_idx]
    last_date = last_row["Date"]
    last_sp   = float(last_row["sp500_abs"])
    last_dgs  = float(last_row["DGS10_abs"])
    bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=T)
    if args.start_date is not None:
        print(f"  開始日: {last_date.date()}  SP500={last_sp:.2f}  DGS10={last_dgs:.4f}")

    print(f"\nSimulating {args.n_paths} paths × {T} days ...")

    for pid in range(args.n_paths):
        # 独立 GM ノイズを生成してから相関付け
        raw1 = gm_sampler(args.p, sigma1_norm, sigma2_norm, T, rng)
        raw2 = gm_sampler(args.p, sigma1_norm, sigma2_norm, T, rng)
        eps_mat = L @ np.vstack([raw1, raw2])  # (2, T): 相関付き GM ノイズ
        eps_sp, eps_dg = eps_mat[0], eps_mat[1]

        r_sp = simulate_arch_gm(omega_sp, alphas_sp, last_rs_sp, eps_sp)
        r_dg = simulate_arch_gm(omega_dg, alphas_dg, last_rs_dg, eps_dg)

        sp_cur, dg_cur = last_sp, last_dgs
        for t in range(T):
            sp_cur = sp_cur * (1.0 + r_sp[t])
            dg_cur = dg_cur + r_dg[t]
            rows.append({
                "path_id":   pid,
                "Date":      bdays[t].strftime("%Y-%m-%d"),
                "sp500_abs": sp_cur,
                "DGS10_abs": dg_cur,
                "sp500":     r_sp[t],
                "DGS10":     r_dg[t],
            })

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           default="output.csv")
    parser.add_argument("--q",             type=int,   default=5,
                        help="ARCH の次数")
    parser.add_argument("--p",             type=float, default=0.01,
                        help="GM のテール確率")
    parser.add_argument("--k",             type=float, default=0.001,
                        help="σ₁ = k, σ₂ = ratio * k（正規化前。比率のみが有効）")
    parser.add_argument("--ratio",         type=float, default=7.0,
                        help="σ₂ / σ₁ の比率")
    parser.add_argument("--n_paths",       type=int,   default=20)
    parser.add_argument("--business_days", type=int,   default=252)
    parser.add_argument("--start_date",    default=None,
                        help="生成開始日 YYYY-MM-DD。省略時はデータ末尾。")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--out",           default="arch_gm/generated_paths.csv")
    args = parser.parse_args()
    main(args)
