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
    #
    # PUBLIC  (dim 0-5 , 6次元): 全投資家が観測
    #   財務健全性, 収益性, 流動性, 負債水準, 売上成長率, 株主還元
    #
    # SECTOR  (dim 6-13, 8次元): 専門セクター投資家のみ観測
    #   需要動向, 市場シェア, 原材料コスト, 設備投資, 在庫水準,
    #   受注残, 競合動向, 規制リスク
    #
    # PRIVATE (dim 14-19, 6次元): 直接観測不能, W_i 経由で推定
    #   内部効率, 人材・組織力, R&Dパイプライン,
    #   未発表プロジェクト, 経営陣の質, 将来競争優位
    n_pub:  int = 6
    n_sec:  int = 8
    n_priv: int = 6   # total d = 20

    # 次元ごとの動態 (安定条件: phi[k] + rho[k] < 1)
    # public : 高持続性, 弱伝播   phi+rho ≈ 0.94
    # sector : 中持続性, 中伝播   phi+rho ≈ 0.92
    # private: 低持続性, 強伝播   phi+rho ≈ 0.90
    phi_dims: Tuple[float, ...] = (
        0.88, 0.86, 0.84, 0.83, 0.82, 0.80,          # public  (6)
        0.75, 0.73, 0.72, 0.70, 0.68, 0.67, 0.65, 0.63,  # sector  (8)
        0.61, 0.59, 0.57, 0.55, 0.54, 0.52,           # private (6)
    )
    rho_dims: Tuple[float, ...] = (
        0.06, 0.07, 0.08, 0.09, 0.10, 0.12,           # public
        0.17, 0.18, 0.20, 0.21, 0.22, 0.23, 0.25, 0.27,  # sector
        0.28, 0.29, 0.30, 0.32, 0.33, 0.35,           # private
    )
    eta_dims: Tuple[float, ...] = (
        0.015, 0.016, 0.016, 0.017, 0.018, 0.018,     # public
        0.020, 0.021, 0.022, 0.023, 0.023, 0.024, 0.025, 0.026,  # sector
        0.014, 0.014, 0.013, 0.013, 0.012, 0.012,     # private
    )

    obs_sigma_pub: float = 0.040
    obs_sigma_sec: float = 0.050

    # スコアの次元ウェイト (合計 1.0)
    # public  6次元: 各 0.08 → 合計 0.48
    # sector  8次元: 各 0.04 → 合計 0.32
    # private 6次元: 各 0.033 → 合計 ≈ 0.20
    dim_weights: Tuple[float, ...] = (
        0.080, 0.080, 0.080, 0.080, 0.080, 0.080,     # public
        0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040,  # sector
        0.033, 0.033, 0.034, 0.033, 0.033, 0.034,     # private (合計 0.200)
    )

    # 時価総額の偏在化
    # mktcap_pareto_a: Pareto 分布の形状パラメータ (小さいほど集中度が高い)
    # mktcap_degree_power: BA次数との相関の強さ (0=相関なし, 1=強相関)
    mktcap_pareto_a:      float = 1.2
    mktcap_degree_power:  float = 0.7

    rare_shock_prob:  float = 0.018
    rare_shock_sigma: float = 0.12

    price_impact:        float = 0.0080
    idio_vol:            float = 0.0075
    market_vol:          float = 0.0060
    common_shock_beta:   float = 1.00
    market_garch_alpha:  float = 0.080
    market_garch_beta:   float = 0.90
    leverage_vol_beta:   float = 0.015
    vol_persistence:     float = 0.94
    # market_stress が base_var を動かすスケール
    garch_stress_scale:  float = 6.0
    garch_down_scale:    float = 2.0
    # 実現ボラティリティ EWMA を GARCH の omega に使う
    # True: omega_t = EWMA(common_noise²) × (1-α-β)  → 過去の実現ショックから推定
    #       初期値は output.csv 直近252日のヒストリカルボラから設定
    # False: 従来の推定値ベース market_stress → base_var が毎期変動
    use_realized_vol:         bool  = False
    realized_vol_ewma_lambda: float = 0.94   # RiskMetrics 式の減衰係数

    initial_sp500_abs: float | None = None
    initial_dgs10_abs: float | None = None

    # use_graph=False → W_i を無視 (private 推定なし・pred_next も W_i なし)
    # セクター観測 (sector dims) は use_graph に関わらず常に有効
    use_graph: bool = True


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

        rows.append({
            "investor_id": i,
            "expertise_sector": expertise,
            "edge_keep_base": base_keep,
            "risk_tolerance": rng.lognormal(mean=-2.6, sigma=0.35),
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

    true_w = make_ba_graph(n, config.ba_m, rng)
    sectors = rng.integers(0, config.n_sectors, size=n)
    subjective_graphs, investor_df = make_subjective_graphs(
        true_w, sectors, config.n_investors, rng
    )

    # 時価総額の偏在化: Pareto 分布 + BA次数との相関
    # ハブ企業（高次数）ほど大企業になる (現実の大企業≈多くのセクターに関与)
    degree = (true_w > 0).sum(axis=1).astype(float)
    base_shares = rng.pareto(a=config.mktcap_pareto_a, size=n) + 1.0
    degree_factor = (degree / degree.mean()) ** config.mktcap_degree_power
    shares = base_shares * degree_factor
    shares = shares / shares.mean()  # 平均1に正規化

    firm_prices = np.full(n, initial_sp) * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    market_caps = firm_prices * shares

    phi    = np.array(config.phi_dims)    # (d,)
    rho    = np.array(config.rho_dims)
    eta    = np.array(config.eta_dims)
    dim_w  = np.array(config.dim_weights) # (d,)

    x      = rng.normal(0.0, 0.020, size=(n, d))
    x_prev = x.copy()

    # 実現ボラティリティ (外部注入方式):
    # 実データ S&P500 の rolling 63 日分散を各シミュレーション日の base_var として使う。
    # フィードバックループなし → t₅ 分布の E[z²]=5/3 による発散を回避。
    hist_returns = output_df["sp500"].dropna().values
    hist_total = len(hist_returns)
    start_idx_hist = hist_total - t_max  # simulation day 0 が対応する output_df の位置
    rv_window = 63  # rolling window for realized variance
    hist_realized_var_series = np.empty(t_max)
    for _t in range(t_max):
        _end = start_idx_hist + _t
        _st  = max(0, _end - rv_window)
        _w   = hist_returns[_st:_end]
        hist_realized_var_series[_t] = float(np.var(_w)) if len(_w) >= 5 else config.market_vol ** 2
    # EWMA は削除 (外部注入方式では不要)
    realized_common_sq_ewma = 0.0  # unused placeholder

    cash     = np.full(config.n_investors, 1.0, dtype=float)
    holdings = rng.lognormal(mean=-5.0, sigma=0.4, size=(config.n_investors, n))

    firm_vol          = np.full(n, config.idio_vol)
    market_var        = config.market_vol ** 2
    prev_common_noise = 0.0
    sp_abs            = initial_sp
    records           = []
    firm_records      = []

    params = investor_df.to_dict("records")

    for t in range(t_max):
        dgs10_abs    = float(historical_tail.loc[t, "DGS10_abs"])
        dgs10_change = float(historical_tail.loc[t, "DGS10"])
        rf_level     = dgs10_abs / 100.0

        # --- 真の状態遷移 (次元独立) ---
        rare = rng.random(n) < config.rare_shock_prob
        for k in range(d):
            noise_k = rng.normal(0.0, eta[k], size=n)
            if k == 0:
                noise_k += rare * rng.standard_t(df=3, size=n) * config.rare_shock_sigma
            x[:, k] = phi[k] * x[:, k] + rho[k] * (true_w @ x[:, k]) + noise_k

        # --- 観測生成 ---
        # public dims: モメンタム項付き (dim 0 のみ 0.35、他 0.15)
        y_pub = np.empty((n, n_pub))
        for k in range(n_pub):
            mom = 0.35 if k == 0 else 0.15
            y_pub[:, k] = x[:, k] + mom * (x[:, k] - x_prev[:, k]) + rng.normal(0.0, config.obs_sigma_pub, size=n)

        # sector dims: 専門家のみ使う観測 (生成はここ、マスクは投資家ループ内)
        y_sec = np.empty((n, n_sec))
        for k_loc in range(n_sec):
            y_sec[:, k_loc] = x[:, n_pub + k_loc] + rng.normal(0.0, config.obs_sigma_sec, size=n)

        # private dims: 直接観測不能 (x[:, n_pub+n_sec:] は誰にも見えない)

        buy_value      = np.zeros(n)
        sell_value     = np.zeros(n)
        agg_total_est  = np.zeros(n)
        momentum       = np.zeros(n) if t == 0 else firm_return

        for i, p in enumerate(params):
            expertise   = int(p["expertise_sector"])
            expert_mask = (sectors == expertise)   # shape (n,), 専門セクター企業

            # [1] public: 全員が観測 → 加重和
            est_pub = y_pub @ dim_w[:n_pub]   # (n, n_pub) @ (n_pub,) → (n,)

            # [2] sector: 専門企業のみ観測、他は 0
            y_i_sec  = y_sec * expert_mask[:, None]   # (n, n_sec)
            est_sec  = y_i_sec @ dim_w[n_pub:n_pub+n_sec]  # (n,)

            # [3] private: W_i @ (公開dims の平均) で隣接企業の内部状態を推定
            y_pub_scalar = y_pub.mean(axis=1)   # (n,) — 公開情報のスカラー集約
            if config.use_graph:
                neighbor_signal = subjective_graphs[i] @ y_pub_scalar  # (n,)
                est_priv = dim_w[n_pub+n_sec:].sum() * p["belief_rho_h"] * neighbor_signal
            else:
                est_priv = np.zeros(n)

            total_est     = est_pub + est_sec + est_priv
            agg_total_est += total_est

            # 来期予測
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

            z_buy  = np.exp(np.clip( p["temperature"] * score, -20, 20))
            z_sell = np.exp(np.clip(-p["loss_asymmetry"] * p["temperature"] * score, -20, 20))
            denom  = z_buy + z_sell + 1.0
            p_buy  = z_buy  / denom
            p_sell = z_sell / denom

            actions   = rng.random(n)
            buy_mask  = actions < p_buy
            sell_mask = (actions >= p_buy) & (actions < p_buy + p_sell)

            conviction = np.minimum(1.0, np.abs(score) / 0.12)
            size_frac  = p["risk_tolerance"] * (0.25 + conviction)
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

        # --- 価格形成 ---
        agg_est_mean  = agg_total_est / config.n_investors
        downside_stress = float(np.maximum(-np.mean(agg_est_mean), 0.0))

        if config.use_realized_vol:
            # 外部注入方式: 実データ rolling 63 日分散を base_var として使う
            # → market_stress ベースの大きな base_var 変動を抑制
            # → フィードバックなし (t₅ 分散ファクター問題を回避)
            base_var = float(hist_realized_var_series[t]) * (
                1.0 + 0.5 * downside_stress  # 下落時の小さな非対称補正
            )
            market_var = (
                (1.0 - config.market_garch_alpha - config.market_garch_beta) * base_var
                + config.market_garch_alpha * prev_common_noise ** 2
                + config.market_garch_beta * market_var
                + config.leverage_vol_beta * max(-prev_common_noise, 0.0) ** 2
            )
            market_var = min(max(market_var, 1e-8), 0.012 ** 2)
        else:
            market_stress = float(np.mean(np.abs(agg_est_mean)) + 0.6 * abs(np.mean(imbalance)))
            base_var = (config.market_vol ** 2) * (
                1.0 + config.garch_stress_scale * market_stress
                + config.garch_down_scale * downside_stress
            )
            market_var = (
                (1.0 - config.market_garch_alpha - config.market_garch_beta) * base_var
                + config.market_garch_alpha * prev_common_noise ** 2
                + config.market_garch_beta * market_var
                + config.leverage_vol_beta * max(-prev_common_noise, 0.0) ** 2
            )
            market_var = min(max(market_var, 1e-8), 0.012 ** 2)
        market_vol_t  = float(np.sqrt(market_var))
        common_noise  = config.common_shock_beta * rng.standard_t(df=5) * market_vol_t

        firm_vol = (
            config.vol_persistence * firm_vol
            + (1.0 - config.vol_persistence) * config.idio_vol
        )
        noise = rng.standard_t(df=5, size=n) * firm_vol

        firm_return = config.price_impact * imbalance + common_noise + noise
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
                })
                firm_records.append(rec)

        prev_common_noise = float(common_noise)
        x_prev = x.copy()

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
