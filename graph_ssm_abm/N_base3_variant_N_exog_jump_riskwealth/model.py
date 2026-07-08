from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    seed: int = 42
    n_days: int = 1260
    n_firms: int = 80
    n_investors: int = 60
    ba_m: int = 3
    n_sectors: int = 8

    # 状態次元 d = n_pub + n_sec + n_priv = 20
    n_pub:  int = 6
    n_sec:  int = 8
    n_priv: int = 6

    phi_dims: Tuple[float, ...] = (
        0.88, 0.86, 0.84, 0.83, 0.82, 0.80,
        0.75, 0.73, 0.72, 0.70, 0.68, 0.67, 0.65, 0.63,
        0.61, 0.59, 0.57, 0.55, 0.54, 0.52,
    )
    rho_dims: Tuple[float, ...] = (
        0.06, 0.07, 0.08, 0.09, 0.10, 0.12,
        0.17, 0.18, 0.20, 0.21, 0.22, 0.23, 0.25, 0.27,
        0.28, 0.29, 0.30, 0.32, 0.33, 0.35,
    )
    eta_dims: Tuple[float, ...] = (
        0.015, 0.016, 0.016, 0.017, 0.018, 0.018,
        0.020, 0.021, 0.022, 0.023, 0.023, 0.024, 0.025, 0.026,
        0.014, 0.014, 0.013, 0.013, 0.012, 0.012,
    )

    obs_sigma_pub: float = 0.040
    obs_sigma_sec: float = 0.050

    dim_weights: Tuple[float, ...] = (
        0.080, 0.080, 0.080, 0.080, 0.080, 0.080,
        0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040,
        0.033, 0.033, 0.034, 0.033, 0.033, 0.034,
    )

    mktcap_pareto_a:     float = 1.2
    mktcap_degree_power: float = 0.7
    # zero graph ablation でサイズ分布だけは同じ BA 参照グラフに合わせるための設定
    # "true": 実際に状態伝播で使う true_w の次数を使う
    # "ba_reference": 辺をゼロ化する前の BA 参照グラフの次数を使う
    mktcap_degree_reference: str = "true"

    rare_shock_prob:  float = 0.018
    rare_shock_sigma: float = 0.12

    price_impact:       float = 0.0080
    idio_vol:           float = 0.0075
    market_vol:         float = 0.0060   # 投資家が参照する実現ボラ EWMA の基準値
    common_shock_beta:  float = 1.00
    # GARCH は安定条件 α×(5/3) + β < 1 を満たすよう設定
    # α=0.05, β=0.88 → 0.083 + 0.88 = 0.963 < 1
    market_garch_alpha: float = 0.050
    market_garch_beta:  float = 0.880
    leverage_vol_beta:  float = 0.010
    vol_persistence:    float = 0.94

    # c_t の外生ジャンプ化:
    # 通常時は小さな共通ノイズ、まれにジャンプするだけにする。
    # market_stress や取引量では c_t の分散を増幅しない。
    exog_common_sigma:      float = 0.0015
    exog_common_jump_prob:  float = 0.010
    exog_common_jump_sigma: float = 0.030
    exog_common_clip:       float = 0.080
    realized_vol_lambda:    float = 0.94

    # 旧 GARCH チャネルは比較用に残すが、N 系の主実験では使わない
    garch_stress_scale: float = 0.0
    garch_down_scale:   float = 0.3    # 下落時の小さな非対称補正のみ残す

    # リスク選好パラメータ (variant_N の核心)
    # vol_sensitivity > 0 → リスク選好型: 高ボラ時に積極売買
    # vol_sensitivity < 0 → リスク回避型: 高ボラ時に縮小
    # 平均が正 (0.40) → 市場全体として弱いリスク選好バイアス
    vol_sensitivity_mean: float = 0.40
    vol_sensitivity_std:  float = 0.60

    # リスク選好型投資家が資産規模でも優勢になりやすいようにする
    wealth_sigma:          float = 1.00
    wealth_vol_corr:       float = 0.80
    wealth_clip_min:       float = 0.10
    wealth_clip_max:       float = 12.0

    # 高ボラ時に注文サイズだけでなく、売買参加確率も変える
    participation_vol_power: float = 1.00

    # 取引量が通常より大きいとき、c_t ではなく注文インパクトを増幅する。
    # これにより「大口リスク選好投資家の取引量増」が価格変動へ直接つながる。
    impact_activity_scale: float = 0.0
    impact_activity_clip:  float = 4.0

    # 取引量 (trading volume) → GARCH omega チャネル
    # 取引量が baseline より高いとき base_var を増大させる
    # → risk-seeking 投資家が高ボラ時に大量売買 → volume 増大 → ボラ持続
    vol_activity_scale:        float = 0.0   # 0=無効, >0 で有効
    vol_activity_ewma_lambda:  float = 0.94  # baseline vol の EWMA 減衰

    initial_sp500_abs: float | None = None
    initial_dgs10_abs: float | None = None
    use_graph: bool = True

    # グラフ構造のアブレーション
    # graph_topology="ba": 真の企業ネットワークは BA グラフ
    # graph_topology="zero": 真の企業ネットワークも投資家グラフも辺0本
    # subjective_graph_mode="partial": 欠損・誤認ありの主観グラフ
    # subjective_graph_mode="perfect": 全投資家が真の企業ネットワークを完全に知る
    graph_topology: str = "ba"
    subjective_graph_mode: str = "partial"


def make_ba_graph(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if m < 1 or m >= n:
        raise ValueError("ba_m must satisfy 1 <= m < n_firms")
    adj = np.zeros((n, n), dtype=float)
    degrees = np.zeros(n, dtype=float)
    start = m + 1
    for u in range(start):
        for v in range(u + 1, start):
            adj[u, v] = adj[v, u] = 1.0
            degrees[u] += 1
            degrees[v] += 1
    for new in range(start, n):
        probs = degrees[:new] / degrees[:new].sum()
        targets = rng.choice(new, size=m, replace=False, p=probs)
        for target in targets:
            adj[new, target] = adj[target, new] = 1.0
            degrees[new] += 1
            degrees[target] += 1
    weights = rng.uniform(0.25, 1.0, size=(n, n))
    adj *= (weights + weights.T) / 2.0
    return row_normalize(adj)


def row_normalize(mat: np.ndarray) -> np.ndarray:
    row_sum = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)


def make_subjective_graphs(
    true_w: np.ndarray,
    sectors: np.ndarray,
    n_investors: int,
    rng: np.random.Generator,
    vol_sensitivity_mean: float = 0.40,
    vol_sensitivity_std:  float = 0.60,
) -> tuple[np.ndarray, pd.DataFrame]:
    n = true_w.shape[0]
    graphs = np.zeros((n_investors, n, n), dtype=float)
    rows = []

    for i in range(n_investors):
        expertise = int(rng.integers(0, sectors.max() + 1))
        base_keep = rng.uniform(0.15, 0.55)
        expert_bonus = rng.uniform(0.20, 0.40)
        false_edge_prob = rng.uniform(0.002, 0.015)

        keep_prob = np.full((n, n), base_keep)
        expert_mask = (sectors[:, None] == expertise) | (sectors[None, :] == expertise)
        keep_prob = np.where(expert_mask, np.minimum(0.95, keep_prob + expert_bonus), keep_prob)
        keep = rng.random((n, n)) < keep_prob
        subjective = np.where(keep, true_w, 0.0)
        false_edges = (rng.random((n, n)) < false_edge_prob) & (true_w == 0)
        subjective = np.where(false_edges, rng.uniform(0.02, 0.15, size=(n, n)), subjective)
        np.fill_diagonal(subjective, 0.0)
        subjective = row_normalize(subjective)
        graphs[i] = subjective

        # vol_sensitivity: 正寄りの正規分布 (-2.0, 3.0) でクリップ
        vol_sens = float(np.clip(rng.normal(vol_sensitivity_mean, vol_sensitivity_std), -2.0, 3.0))

        rows.append({
            "investor_id": i,
            "expertise_sector": expertise,
            "edge_keep_base": base_keep,
            "risk_tolerance": rng.lognormal(mean=-2.6, sigma=0.35),
            "vol_sensitivity": vol_sens,
            "trend_weight": rng.normal(0.28, 0.12),
            "value_weight": rng.normal(1.00, 0.25),
            "uncertainty_aversion": rng.uniform(0.15, 0.75),
            "rate_sensitivity": rng.uniform(0.02, 0.18),
            "temperature": rng.uniform(2.0, 6.0),
            "loss_asymmetry": rng.uniform(1.0, 1.9),
            "belief_phi": float(np.clip(rng.normal(0.72, 0.07), 0.50, 0.90)),
            "belief_rho_s": float(np.clip(rng.normal(0.18, 0.06), 0.04, 0.30)),
            "belief_rho_h": float(np.clip(rng.normal(0.28, 0.08), 0.10, 0.52)),
            "obs_var": rng.lognormal(mean=np.log(0.055**2), sigma=0.45),
            "proc_var": rng.lognormal(mean=np.log(0.022**2), sigma=0.45),
            "recognized_edges": int((subjective > 0).sum()),
        })

    return graphs, pd.DataFrame(rows)


def simulate_market(
    output_df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    n = config.n_firms
    t_max = config.n_days
    n_pub  = config.n_pub
    n_sec  = config.n_sec
    n_priv = config.n_priv
    d = n_pub + n_sec + n_priv   # = 20

    if len(output_df) < t_max + 2:
        raise ValueError("output.csv is shorter than requested n_days")

    historical_tail = output_df.tail(t_max).reset_index(drop=True)
    last_row = output_df.iloc[-1]
    initial_sp    = float(config.initial_sp500_abs or last_row["sp500_abs"])
    initial_dgs10 = float(config.initial_dgs10_abs or historical_tail.iloc[0]["DGS10_abs"])

    dates = pd.bdate_range(
        pd.to_datetime(last_row["Date"]) + pd.offsets.BDay(1), periods=t_max
    )

    ba_reference_w = make_ba_graph(n, config.ba_m, rng)
    if config.graph_topology == "zero":
        true_w = np.zeros((n, n), dtype=float)
    elif config.graph_topology == "ba":
        true_w = ba_reference_w.copy()
    else:
        raise ValueError(f"unknown graph_topology: {config.graph_topology}")

    sectors = rng.integers(0, config.n_sectors, size=n)
    subjective_graphs, investor_df = make_subjective_graphs(
        true_w, sectors, config.n_investors, rng,
        vol_sensitivity_mean=config.vol_sensitivity_mean,
        vol_sensitivity_std=config.vol_sensitivity_std,
    )
    if config.graph_topology == "zero":
        subjective_graphs[:] = 0.0
        investor_df["recognized_edges"] = 0
    elif config.subjective_graph_mode == "perfect":
        subjective_graphs[:] = true_w[None, :, :]
        investor_df["recognized_edges"] = int((true_w > 0).sum())
    elif config.subjective_graph_mode != "partial":
        raise ValueError(f"unknown subjective_graph_mode: {config.subjective_graph_mode}")

    # 投資家資産: heavy-tail + vol_sensitivity との正相関。
    # リスク選好型が数だけでなく資金量でも市場に残りやすい構造を作る。
    vol_s = investor_df["vol_sensitivity"].to_numpy(dtype=float)
    vol_s_z = (vol_s - vol_s.mean()) / (vol_s.std() + 1e-12)
    wealth_log = rng.normal(0.0, config.wealth_sigma, size=config.n_investors)
    wealth_log += config.wealth_vol_corr * vol_s_z
    wealth_factor = np.exp(wealth_log)
    wealth_factor = np.clip(wealth_factor, config.wealth_clip_min, config.wealth_clip_max)
    wealth_factor = wealth_factor / wealth_factor.mean()
    investor_df["wealth_factor"] = wealth_factor

    # 時価総額: Pareto + BA次数相関
    if config.mktcap_degree_reference == "true":
        degree_for_mktcap = (true_w > 0).sum(axis=1).astype(float)
    elif config.mktcap_degree_reference == "ba_reference":
        degree_for_mktcap = (ba_reference_w > 0).sum(axis=1).astype(float)
    else:
        raise ValueError(f"unknown mktcap_degree_reference: {config.mktcap_degree_reference}")

    degree = (true_w > 0).sum(axis=1).astype(float)
    base_shares = rng.pareto(a=config.mktcap_pareto_a, size=n) + 1.0
    if degree_for_mktcap.mean() > 0:
        degree_factor = (degree_for_mktcap / degree_for_mktcap.mean()) ** config.mktcap_degree_power
    else:
        degree_factor = np.ones(n, dtype=float)
    shares = base_shares * degree_factor
    shares = shares / shares.mean()

    firm_prices = np.full(n, initial_sp) * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    market_caps = firm_prices * shares

    phi    = np.array(config.phi_dims)
    rho    = np.array(config.rho_dims)
    eta    = np.array(config.eta_dims)
    dim_w  = np.array(config.dim_weights)

    x      = rng.normal(0.0, 0.020, size=(n, d))
    x_prev = x.copy()

    cash     = wealth_factor.astype(float).copy()
    holdings = rng.lognormal(mean=-5.0, sigma=0.4, size=(config.n_investors, n)) * wealth_factor[:, None]

    firm_vol          = np.full(n, config.idio_vol)
    market_var        = config.market_vol ** 2
    prev_common_noise = 0.0
    # 投資家が見るボラは c_t の内部 variance ではなく、S&P リターンの EWMA 実現ボラ
    prev_market_vol_t = config.market_vol
    realized_var_ewma = config.market_vol ** 2
    # 取引量 fast/slow 二重 EWMA:
    #   fast (λ=vol_activity_ewma_lambda ≈0.94, 半減期~11日): 直近の変動を捉える
    #   slow (λ=0.99, 半減期~69日): 長期 baseline を追跡
    #   volume_ratio = fast / slow → mean-revert + spike 後 ~11日の持続
    vol_trade_ewma_fast  = 0.0
    vol_trade_ewma_slow  = 0.0
    vol_trade_ewma_initialized = False
    sp_abs            = initial_sp
    records           = []
    firm_records      = []

    params = investor_df.to_dict("records")

    for t in range(t_max):
        dgs10_abs    = float(historical_tail.loc[t, "DGS10_abs"])
        dgs10_change = float(historical_tail.loc[t, "DGS10"])
        rf_level     = dgs10_abs / 100.0

        # --- 真の状態遷移 ---
        rare = rng.random(n) < config.rare_shock_prob
        for k in range(d):
            noise_k = rng.normal(0.0, eta[k], size=n)
            if k == 0:
                noise_k += rare * rng.standard_t(df=3, size=n) * config.rare_shock_sigma
            x[:, k] = phi[k] * x[:, k] + rho[k] * (true_w @ x[:, k]) + noise_k

        # --- 観測生成 ---
        y_pub = np.empty((n, n_pub))
        for k in range(n_pub):
            mom = 0.35 if k == 0 else 0.15
            y_pub[:, k] = x[:, k] + mom * (x[:, k] - x_prev[:, k]) + rng.normal(0.0, config.obs_sigma_pub, size=n)

        y_sec = np.empty((n, n_sec))
        for k_loc in range(n_sec):
            k_glob = n_pub + k_loc
            y_sec[:, k_loc] = x[:, k_glob] + rng.normal(0.0, config.obs_sigma_sec, size=n)

        # モメンタム (前期 sp500 リターン)
        momentum = records[-1]["sp500"] if t > 0 else 0.0

        buy_value  = np.zeros(n)
        sell_value = np.zeros(n)
        agg_total_est = np.zeros(n)

        # --- 投資家ループ ---
        for i, p in enumerate(params):
            expertise   = int(p["expertise_sector"])
            expert_mask = (sectors == expertise)

            est_pub = y_pub @ dim_w[:n_pub]
            y_i_sec = y_sec * expert_mask[:, None]
            est_sec = y_i_sec @ dim_w[n_pub:n_pub+n_sec]

            y_pub_scalar = y_pub.mean(axis=1)
            if config.use_graph:
                neighbor_signal = subjective_graphs[i] @ y_pub_scalar
                est_priv = dim_w[n_pub+n_sec:].sum() * p["belief_rho_h"] * neighbor_signal
            else:
                est_priv = np.zeros(n)

            total_est = est_pub + est_sec + est_priv
            agg_total_est += total_est

            if config.use_graph:
                pred_next = (
                    p["belief_phi"] * total_est
                    + p["belief_rho_s"] * (subjective_graphs[i] @ total_est)
                )
            else:
                pred_next = p["belief_phi"] * total_est

            delta       = pred_next - total_est
            uncertainty = np.sqrt(p["obs_var"] + p["proc_var"])
            score = (
                p["value_weight"] * pred_next
                + p["trend_weight"] * delta
                - p["uncertainty_aversion"] * uncertainty
                - p["rate_sensitivity"] * rf_level
                + 0.25 * momentum
            )

            # vol_sensitivity: 前期の実現ボラ水準に基づいて売買参加率とポジションサイズを調整
            # vol_factor > 1 → リスク選好型が高ボラ時に積極売買
            # vol_factor < 1 → リスク回避型が高ボラ時に縮小
            vol_ratio = prev_market_vol_t / config.market_vol  # 1.0 が baseline
            vol_factor = float(np.clip(1.0 + p["vol_sensitivity"] * (vol_ratio - 1.0), 0.05, 4.0))
            participation_factor = float(np.clip(vol_factor ** config.participation_vol_power, 0.10, 5.0))

            z_buy  = np.exp(np.clip( p["temperature"] * score, -20, 20)) * participation_factor
            z_sell = np.exp(np.clip(-p["loss_asymmetry"] * p["temperature"] * score, -20, 20)) * participation_factor
            denom  = z_buy + z_sell + 1.0
            p_buy  = z_buy  / denom
            p_sell = z_sell / denom

            actions   = rng.random(n)
            buy_mask  = actions < p_buy
            sell_mask = (actions >= p_buy) & (actions < p_buy + p_sell)

            conviction = np.minimum(1.0, np.abs(score) / 0.12)

            size_frac  = p["risk_tolerance"] * vol_factor * (0.25 + conviction)
            size_frac *= rng.lognormal(mean=0.0, sigma=0.45, size=n)
            size_frac  = np.clip(size_frac, 0.0002, 0.080)

            buy_orders  = buy_mask  * cash[i]          * size_frac
            sell_orders = sell_mask * holdings[i] * firm_prices * size_frac

            total_buy = buy_orders.sum()
            if total_buy > cash[i]:
                buy_orders *= cash[i] / (total_buy + 1e-12)

            buy_value  += buy_orders
            sell_value += sell_orders
            cash[i]    += sell_orders.sum() - buy_orders.sum()
            holdings[i] += buy_orders  / firm_prices
            holdings[i] -= sell_orders / firm_prices
            holdings[i]  = np.maximum(holdings[i], 0.0)

        imbalance = (buy_value - sell_value) / (buy_value + sell_value + 1e-9)
        total_trade = float(buy_value.sum() + sell_value.sum())

        # --- 価格形成 (c_t の外生ジャンプ化) ---
        # c_t は市場活動・market_stress・取引量から切り離す。
        # 通常時は小さい共通ノイズ、まれにジャンプする外乱としてのみ入る。
        agg_est_mean    = agg_total_est / config.n_investors
        market_stress   = float(np.mean(np.abs(agg_est_mean)) + 0.6 * abs(np.mean(imbalance)))
        if vol_trade_ewma_initialized:
            volume_ratio = vol_trade_ewma_fast / max(vol_trade_ewma_slow, 1e-12)
        else:
            volume_ratio = 1.0

        common_noise = rng.normal(0.0, config.exog_common_sigma)
        if rng.random() < config.exog_common_jump_prob:
            common_noise += rng.standard_t(df=3) * config.exog_common_jump_sigma
        common_noise = float(np.clip(config.common_shock_beta * common_noise,
                                     -config.exog_common_clip, config.exog_common_clip))
        market_vol_t = prev_market_vol_t

        firm_vol = (
            config.vol_persistence * firm_vol
            + (1.0 - config.vol_persistence) * config.idio_vol
        )
        noise = rng.standard_t(df=5, size=n) * firm_vol

        impact_factor = float(np.clip(
            1.0 + config.impact_activity_scale * max(0.0, volume_ratio - 1.0),
            0.25,
            config.impact_activity_clip,
        ))
        firm_return = config.price_impact * impact_factor * imbalance + common_noise + noise
        firm_return = np.clip(firm_return, -0.18, 0.18)

        firm_prices = firm_prices * (1.0 + firm_return)
        firm_prices = np.maximum(firm_prices, 1e-3)
        market_caps = firm_prices * shares
        weights     = market_caps / market_caps.sum()
        sp_ret      = float(np.sum(weights * firm_return))
        sp_abs      = float(sp_abs * (1.0 + sp_ret))

        records.append({
            "path_id": 0,
            "Date": dates[t].strftime("%Y-%m-%d"),
            "sp500_abs": sp_abs,
            "DGS10_abs": dgs10_abs if t > 0 else initial_dgs10,
            "sp500": sp_ret,
            "DGS10": dgs10_change,
        })

        if t in {0, 1, 2, 20, 60, 252, 504, 756, 1008, t_max - 1}:
            for j in range(n):
                rec = {"day": t, "firm_id": j, "sector": int(sectors[j])}
                for k in range(d):
                    rec[f"x{k}"] = float(x[j, k])
                rec.update({
                    "price": float(firm_prices[j]),
                    "return": float(firm_return[j]),
                    "market_weight": float(weights[j]),
                    "imbalance": float(imbalance[j]),
                    "market_vol_t": market_vol_t,
                })
                firm_records.append(rec)

        prev_common_noise = float(common_noise)
        lam_rv = config.realized_vol_lambda
        realized_var_ewma = lam_rv * realized_var_ewma + (1.0 - lam_rv) * (sp_ret ** 2)
        prev_market_vol_t = float(np.sqrt(max(realized_var_ewma, 1e-10)))
        x_prev = x.copy()

        # --- 取引量 fast/slow EWMA をループ末尾で更新（診断用; c_t には入れない） ---
        if not vol_trade_ewma_initialized:
            vol_trade_ewma_fast = vol_trade_ewma_slow = total_trade
            vol_trade_ewma_initialized = True
        else:
            lam_f = config.vol_activity_ewma_lambda        # fast: ~0.94, 半減期 11日
            lam_s = max(0.99, config.vol_activity_ewma_lambda)  # slow: ≥0.99, 半減期 69日
            vol_trade_ewma_fast = lam_f * vol_trade_ewma_fast + (1.0 - lam_f) * total_trade
            vol_trade_ewma_slow = lam_s * vol_trade_ewma_slow + (1.0 - lam_s) * total_trade

    firms_df = pd.DataFrame({
        "firm_id": np.arange(n),
        "sector": sectors,
        "initial_market_cap_weight": market_caps / market_caps.sum(),
        "true_degree": (true_w > 0).sum(axis=1),
    })

    config_dict = {k: getattr(config, k) for k in Config.__dataclass_fields__}

    return (
        pd.DataFrame(records),
        firms_df,
        investor_df,
        {"config": config_dict, "firm_snapshots": pd.DataFrame(firm_records)},
    )
