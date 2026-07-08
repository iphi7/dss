"""
P_base5_variant_P_leverage/model.py

O19 をベースに、leverage effect（下落後にボラが上がる非対称性）を出すための
複数機構を検証する。各機構は独立した Config スイッチで有効/無効を切り替えられる。

追加機構:
  1. GJR-GARCH 的 c_t 分散増幅 (gjr_scale > 0)
       下落後に c_t の sigma を一時的に上昇させる。
       down_var_ewma = decay * down_var_{t-1} + (1-decay) * max(-r_SP_t, 0)^2
       sigma_c_t = sigma_c * sqrt(1 + gjr_scale * down_var_ewma / sigma_c^2)

  2. 非対称価格インパクト (asym_pi_scale > 0)
       下落後に lambda_t（price impact 係数）を増幅する。
       lambda_t *= 1 + asym_pi_scale * down_var_ewma / market_vol^2

  3. 損失トリガー強制売り / stop-loss (stoploss_scale > 0)
       市場下落時（前期 sp_ret < 0）に、リスク回避型投資家（vol_sensitivity < 0）の
       売りオッズを増幅する。損失を被った投資家が追加で売りを出す行動を模倣。

  4. 非対称クラッシュ (asym_crash_sell_only = True)
       流動性クラッシュを売り超過時のみ発動させる。
"""
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
    mktcap_degree_reference: str = "true"

    rare_shock_prob:  float = 0.018
    rare_shock_sigma: float = 0.12

    price_impact:       float = 0.050
    idio_vol:           float = 0.0075
    market_vol:         float = 0.0060
    common_shock_beta:  float = 1.00
    market_garch_alpha: float = 0.050
    market_garch_beta:  float = 0.880
    leverage_vol_beta:  float = 0.010
    vol_persistence:    float = 0.94

    exog_common_sigma:      float = 0.0040
    exog_common_jump_prob:  float = 0.006
    exog_common_jump_sigma: float = 0.035
    exog_common_clip:       float = 0.100
    realized_vol_lambda:    float = 0.985

    garch_stress_scale: float = 0.0
    garch_down_scale:   float = 0.3

    vol_sensitivity_mean: float = 0.80
    vol_sensitivity_std:  float = 0.80

    wealth_sigma:          float = 1.20
    wealth_vol_corr:       float = 1.20
    wealth_clip_min:       float = 0.10
    wealth_clip_max:       float = 12.0

    participation_vol_power: float = 1.80

    impact_activity_scale: float = 2.50
    impact_activity_clip:  float = 6.00
    impact_crash_threshold: float = 1.20
    impact_crash_scale:     float = 4.00
    impact_crash_power:     float = 2.00

    vol_activity_scale:        float = 0.0
    vol_activity_ewma_lambda:  float = 0.94

    initial_sp500_abs: float | None = None
    initial_dgs10_abs: float | None = None
    exog_drift: float = 0.0   # 日次リターンへの外生ドリフト (長期価格安定化用)
    use_graph: bool = True

    graph_topology: str = "ba"
    subjective_graph_mode: str = "partial"

    alpha_max: float = 0.85
    alpha_pub_base: float = 0.45
    alpha_pub_nonexpert_scale: float = 0.70
    alpha_sec_expert: float = 0.60
    alpha_sec_nonexpert: float = 0.05
    alpha_priv_base: float = 0.25
    alpha_priv_nonexpert_scale: float = 0.30
    update_style_sigma: float = 0.25

    private_anchor_scale: float = 1.00
    rho_h_dims: Tuple[float, ...] = (0.90, 1.00, 1.20, 1.00, 0.90, 1.10)

    # 投資家の銘柄選択スコアを「絶対評価」から「相対評価」へ寄せる係数。
    # 1.0なら各投資家内で平均スコアを差し引き、全銘柄への共通な売り/買いバイアスを弱める。
    # uncertainty や金利ペナルティが全銘柄に同じ符号で乗ると長期で市場全体が沈みやすいため、
    # 60年生成ではここを少し有効化する。
    score_centering: float = 0.0
    market_risk_premium_score: float = 0.0
    # raw return ACF 抑制: 前期リターン momentum 項の係数。P/Qでは固定 0.25 だった。
    momentum_score_weight: float = 0.25
    # ACF/ボラ過剰抑制: 価格形成で使う firm return の日次クリップ。
    firm_return_clip: float = 0.18
    # 60年生成用: 投資家の保有/現金が極端な長期レジームを作らないようにする弱いturnover。
    # 各日、ポートフォリオ価値の一部を市場ウェイトのターゲットへ寄せる。
    portfolio_rebalance_rate: float = 0.0
    portfolio_cash_target: float = 0.05

    # delayed medium-term volatility after negative market events.
    # 目的: decile10 leverage の1d/3dを強めず、10d/20dだけ少し戻す。
    delayed_vol_scale: float = 0.0
    delayed_vol_decay: float = 0.90
    delayed_vol_delay: int = 3
    delayed_vol_clip: float = 2.0

    # ===== Leverage 機構 (P 系追加) =====
    # 1. GJR-GARCH: 下落後に c_t sigma を増幅
    gjr_scale: float = 0.0        # > 0 で有効
    gjr_centered: bool = False    # True = ratio to realized_var (平常時に影響なし)

    # 2. 非対称価格インパクト: 下落後に lambda_t を増幅
    asym_pi_scale: float = 0.0    # > 0 で有効
    asym_pi_centered: bool = False  # True = ratio to realized_var (平常時に影響なし)

    # 共通 EWMA 減衰率 (gjr / asym_pi 両方で使用)
    down_ewma_decay: float = 0.95

    # 3a. stop-loss (リスク回避型のみ): 前期下落時に売りオッズ増幅
    stoploss_scale: float = 0.0   # > 0 で有効 (vol_sensitivity < 0 の投資家のみ)
    # 3b. market-wide fear: 下落後に全投資家の売りオッズを増幅
    stoploss_universal_scale: float = 0.0     # > 0 で有効 (全投資家)
    stoploss_universal_threshold: float = 0.005  # 発動する最小下落幅 (例: 0.5%)

    # 4. 非対称クラッシュ: 売り超過時のみ発動
    asym_crash_sell_only: bool = False

    # ===== 長期安定化: ファンダメンタル価値アンカー (Q系) =====
    # 企業価格が基準価値から大きく乖離したときだけ、弱い平均回帰リターンを追加する。
    fundamental_strength: float = 0.0      # κ。0で無効
    fundamental_gap_scale: float = 0.50    # tanh のスケール。log gap がこの程度で飽和
    fundamental_clip: float = 0.006        # 1日あたりの最大アンカー寄与
    fundamental_drift: float = 0.00025     # 基準価値の平均日次成長
    fundamental_noise: float = 0.0008      # 基準価値の企業別日次ノイズ
    fundamental_state_sensitivity: float = 0.0000  # 企業潜在状態から基準価値成長への寄与

    # 市場全体の長期ファンダメンタル指数アンカー。
    # 個別企業ではなくSP500水準そのものの崩壊/爆発を抑える弱い復元力。
    market_anchor_strength: float = 0.0
    market_anchor_gap_scale: float = 0.70
    market_anchor_clip: float = 0.004
    market_anchor_drift: float = 0.00025


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
    vol_sensitivity_mean: float = 0.80,
    vol_sensitivity_std:  float = 0.80,
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
    d = n_pub + n_sec + n_priv

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

    if np.mean(np.abs(true_w)) > 0:
        graph_quality = []
        denom_w = np.mean(np.abs(true_w)) + 1e-12
        for i in range(config.n_investors):
            q = 1.0 - np.mean(np.abs(subjective_graphs[i] - true_w)) / denom_w
            graph_quality.append(float(np.clip(q, 0.0, 1.0)))
        investor_df["graph_quality"] = graph_quality
    else:
        investor_df["graph_quality"] = 0.0

    investor_df["update_style"] = np.clip(
        rng.lognormal(mean=0.0, sigma=config.update_style_sigma, size=config.n_investors),
        0.5, 1.5,
    )

    vol_s = investor_df["vol_sensitivity"].to_numpy(dtype=float)
    vol_s_z = (vol_s - vol_s.mean()) / (vol_s.std() + 1e-12)
    wealth_log = rng.normal(0.0, config.wealth_sigma, size=config.n_investors)
    wealth_log += config.wealth_vol_corr * vol_s_z
    wealth_factor = np.exp(wealth_log)
    wealth_factor = np.clip(wealth_factor, config.wealth_clip_min, config.wealth_clip_max)
    wealth_factor = wealth_factor / wealth_factor.mean()
    investor_df["wealth_factor"] = wealth_factor

    if config.mktcap_degree_reference == "true":
        degree_for_mktcap = (true_w > 0).sum(axis=1).astype(float)
    elif config.mktcap_degree_reference == "ba_reference":
        degree_for_mktcap = (ba_reference_w > 0).sum(axis=1).astype(float)
    else:
        raise ValueError(f"unknown mktcap_degree_reference: {config.mktcap_degree_reference}")

    base_shares = rng.pareto(a=config.mktcap_pareto_a, size=n) + 1.0
    if degree_for_mktcap.mean() > 0:
        degree_factor = (degree_for_mktcap / degree_for_mktcap.mean()) ** config.mktcap_degree_power
    else:
        degree_factor = np.ones(n, dtype=float)
    shares = base_shares * degree_factor
    shares = shares / shares.mean()

    firm_prices = np.full(n, initial_sp) * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    market_caps = firm_prices * shares

    phi   = np.array(config.phi_dims)
    rho   = np.array(config.rho_dims)
    eta   = np.array(config.eta_dims)
    dim_w = np.array(config.dim_weights)

    x      = rng.normal(0.0, 0.020, size=(n, d))
    x_prev = x.copy()

    cash     = wealth_factor.astype(float).copy()
    holdings = rng.lognormal(mean=-5.0, sigma=0.4, size=(config.n_investors, n)) * wealth_factor[:, None]
    belief_state = rng.normal(0.0, 0.010, size=(config.n_investors, n, d))
    rho_h_dim = np.array(config.rho_h_dims, dtype=float)

    firm_vol          = np.full(n, config.idio_vol)
    market_var        = config.market_vol ** 2
    prev_common_noise = 0.0
    prev_market_vol_t = config.market_vol
    realized_var_ewma = config.market_vol ** 2
    vol_trade_ewma_fast  = 0.0
    vol_trade_ewma_slow  = 0.0
    vol_trade_ewma_initialized = False
    sp_abs = initial_sp

    # Leverage 機構用の状態変数
    # down_var_ewma: 下落リターンの EWMA 二乗平均 (GJR / 非対称 PI 共用)
    down_var_ewma = config.exog_common_sigma ** 2
    prev_sp_ret = 0.0  # stop-loss 判定用

    records      = []
    firm_records = []

    params = investor_df.to_dict("records")

    for t in range(t_max):
        dgs10_abs    = float(historical_tail.loc[t, "DGS10_abs"])
        dgs10_change = float(historical_tail.loc[t, "DGS10"])
        rf_level     = dgs10_abs / 100.0

        rare = rng.random(n) < config.rare_shock_prob
        for k in range(d):
            noise_k = rng.normal(0.0, eta[k], size=n)
            if k == 0:
                noise_k += rare * rng.standard_t(df=3, size=n) * config.rare_shock_sigma
            x[:, k] = phi[k] * x[:, k] + rho[k] * (true_w @ x[:, k]) + noise_k

        y_pub = np.empty((n, n_pub))
        for k in range(n_pub):
            mom = 0.35 if k == 0 else 0.15
            y_pub[:, k] = x[:, k] + mom * (x[:, k] - x_prev[:, k]) + rng.normal(0.0, config.obs_sigma_pub, size=n)

        y_sec = np.empty((n, n_sec))
        for k_loc in range(n_sec):
            k_glob = n_pub + k_loc
            y_sec[:, k_loc] = x[:, k_glob] + rng.normal(0.0, config.obs_sigma_sec, size=n)

        y_obs_global = np.concatenate([y_pub, y_sec], axis=1)
        anchor_specs = [
            [(0, 0.60), (3, 0.40)],
            [(1, 0.50), (6, 0.50)],
            [(8, 0.50), (9, 0.50)],
            [(4, 0.50), (10, 0.50)],
            [(2, 0.50), (3, 0.50)],
            [(11, 0.34), (12, 0.33), (13, 0.33)],
        ]
        anchors = np.zeros((n, n_priv), dtype=float)
        for kp, entries in enumerate(anchor_specs):
            for dim_idx, weight in entries:
                anchors[:, kp] += weight * y_obs_global[:, dim_idx]

        momentum = records[-1]["sp500"] if t > 0 else 0.0

        buy_value  = np.zeros(n)
        sell_value = np.zeros(n)
        agg_total_est = np.zeros(n)

        # --- 投資家ループ ---
        for i, p in enumerate(params):
            expertise   = int(p["expertise_sector"])
            expert_mask = (sectors == expertise)
            style = float(p.get("update_style", 1.0))
            graph_quality = float(p.get("graph_quality", 1.0))
            W_i = subjective_graphs[i]

            prev_belief = belief_state[i]
            pred_state = np.empty_like(prev_belief)

            for k in range(d):
                rho_i = p["belief_rho_h"] if k >= n_pub + n_sec else p["belief_rho_s"]
                if config.use_graph:
                    graph_term = W_i @ prev_belief[:, k]
                else:
                    graph_term = 0.0
                pred_state[:, k] = p["belief_phi"] * prev_belief[:, k] + rho_i * graph_term

            updated = pred_state.copy()

            alpha_pub = config.alpha_pub_base * style * np.where(
                expert_mask, 1.0, config.alpha_pub_nonexpert_scale
            )
            alpha_pub = np.clip(alpha_pub, 0.0, config.alpha_max)
            for k in range(n_pub):
                updated[:, k] = (1.0 - alpha_pub) * pred_state[:, k] + alpha_pub * y_pub[:, k]

            alpha_sec = style * np.where(
                expert_mask, config.alpha_sec_expert, config.alpha_sec_nonexpert
            )
            alpha_sec = np.clip(alpha_sec, 0.0, config.alpha_max)
            for ks in range(n_sec):
                kg = n_pub + ks
                updated[:, kg] = (1.0 - alpha_sec) * pred_state[:, kg] + alpha_sec * y_sec[:, ks]

            if config.use_graph:
                alpha_priv = (
                    config.alpha_priv_base
                    * style
                    * graph_quality
                    * np.where(expert_mask, 1.0, config.alpha_priv_nonexpert_scale)
                )
                alpha_priv = np.clip(alpha_priv, 0.0, config.alpha_max)
                for kp in range(n_priv):
                    kg = n_pub + n_sec + kp
                    pseudo_obs = config.private_anchor_scale * p["belief_rho_h"] * rho_h_dim[kp] * (W_i @ anchors[:, kp])
                    updated[:, kg] = (1.0 - alpha_priv) * pred_state[:, kg] + alpha_priv * pseudo_obs

            belief_state[i] = updated

            total_est = updated @ dim_w
            agg_total_est += total_est

            pred_next_state = np.empty_like(updated)
            for k in range(d):
                rho_i = p["belief_rho_h"] if k >= n_pub + n_sec else p["belief_rho_s"]
                if config.use_graph:
                    graph_term = W_i @ updated[:, k]
                else:
                    graph_term = 0.0
                pred_next_state[:, k] = p["belief_phi"] * updated[:, k] + rho_i * graph_term
            pred_next = pred_next_state @ dim_w

            delta       = pred_next - total_est
            uncertainty = np.sqrt(p["obs_var"] + p["proc_var"])
            score = (
                p["value_weight"] * pred_next
                + p["trend_weight"] * delta
                - p["uncertainty_aversion"] * uncertainty
                - p["rate_sensitivity"] * rf_level
                + 0.25 * momentum
            )

            vol_ratio  = prev_market_vol_t / config.market_vol
            vol_factor = float(np.clip(1.0 + p["vol_sensitivity"] * (vol_ratio - 1.0), 0.05, 4.0))
            participation_factor = float(np.clip(vol_factor ** config.participation_vol_power, 0.10, 5.0))

            z_buy  = np.exp(np.clip( p["temperature"] * score, -20, 20)) * participation_factor
            z_sell = np.exp(np.clip(-p["loss_asymmetry"] * p["temperature"] * score, -20, 20)) * participation_factor

            # 3a. stop-loss (リスク回避型のみ)
            if config.stoploss_scale > 0.0 and prev_sp_ret < 0.0 and p["vol_sensitivity"] < 0.0:
                loss_fear = 1.0 + config.stoploss_scale * abs(prev_sp_ret) / (config.market_vol + 1e-10)
                z_sell = z_sell * float(np.clip(loss_fear, 1.0, 5.0))

            # 3b. market-wide fear: 下落後に全投資家の売りオッズを増幅
            if config.stoploss_universal_scale > 0.0 and prev_sp_ret < -config.stoploss_universal_threshold:
                fear_mult = 1.0 + config.stoploss_universal_scale * abs(prev_sp_ret) / (config.market_vol + 1e-10)
                z_sell = z_sell * float(np.clip(fear_mult, 1.0, 6.0))

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

        imbalance   = (buy_value - sell_value) / (buy_value + sell_value + 1e-9)
        total_trade = float(buy_value.sum() + sell_value.sum())

        # --- 価格形成 ---
        agg_est_mean  = agg_total_est / config.n_investors
        market_stress = float(np.mean(np.abs(agg_est_mean)) + 0.6 * abs(np.mean(imbalance)))
        if vol_trade_ewma_initialized:
            volume_ratio = vol_trade_ewma_fast / max(vol_trade_ewma_slow, 1e-12)
        else:
            volume_ratio = 1.0

        # 機構1: GJR-GARCH — 下落後に c_t sigma を増幅 (lagged down_var_ewma 使用)
        if config.gjr_scale > 0.0:
            if config.gjr_centered:
                # centered: realized_var (lagged) の半分との比率で正規化。平常時は増幅しない。
                baseline_dv = realized_var_ewma / 2.0 + 1e-12
                gjr_excess = max(down_var_ewma / baseline_dv - 1.0, 0.0)
                gjr_factor = 1.0 + config.gjr_scale * gjr_excess
            else:
                gjr_factor = 1.0 + config.gjr_scale * down_var_ewma / (config.exog_common_sigma ** 2 + 1e-12)
            current_common_sigma = config.exog_common_sigma * float(np.sqrt(max(gjr_factor, 1.0)))
        else:
            current_common_sigma = config.exog_common_sigma

        common_noise = rng.normal(0.0, current_common_sigma)
        if rng.random() < config.exog_common_jump_prob:
            common_noise += rng.standard_t(df=3) * config.exog_common_jump_sigma
        common_noise = float(np.clip(
            config.common_shock_beta * common_noise,
            -config.exog_common_clip,
             config.exog_common_clip,
        ))
        market_vol_t = prev_market_vol_t

        firm_vol = (
            config.vol_persistence * firm_vol
            + (1.0 - config.vol_persistence) * config.idio_vol
        )
        noise = rng.standard_t(df=5, size=n) * firm_vol

        # 機構2: 非対称価格インパクト — 下落後に lambda 増幅 (lagged down_var_ewma 使用)
        if config.asym_pi_scale > 0.0:
            if config.asym_pi_centered:
                # centered: realized_var/2 との比率の超過分で増幅。平常時は影響なし。
                baseline_dv = realized_var_ewma / 2.0 + 1e-12
                pi_excess = max(down_var_ewma / baseline_dv - 1.0, 0.0)
                asym_factor = 1.0 + config.asym_pi_scale * pi_excess
            else:
                asym_factor = 1.0 + config.asym_pi_scale * down_var_ewma / (config.market_vol ** 2 + 1e-12)
        else:
            asym_factor = 1.0

        # 通常の impact 計算 (activity + crash)
        impact_factor_raw = asym_factor * (1.0 + config.impact_activity_scale * max(0.0, volume_ratio - 1.0))

        # 機構4: 非対称クラッシュ — 売り超過時のみ発動
        if config.asym_crash_sell_only:
            total_sell = sell_value.sum()
            total_buy_v = buy_value.sum()
            sell_pressure_ratio = total_sell / (total_buy_v + 1e-9)
            if sell_pressure_ratio > 1.1:
                crash_excess = max(0.0, volume_ratio - config.impact_crash_threshold)
                impact_factor_raw += config.impact_crash_scale * (crash_excess ** config.impact_crash_power)
        else:
            crash_excess = max(0.0, volume_ratio - config.impact_crash_threshold)
            impact_factor_raw += config.impact_crash_scale * (crash_excess ** config.impact_crash_power)

        impact_factor = float(np.clip(impact_factor_raw, 0.25, config.impact_activity_clip))

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
                    "down_var_ewma": down_var_ewma,
                })
                firm_records.append(rec)

        prev_common_noise = float(common_noise)
        lam_rv = config.realized_vol_lambda
        realized_var_ewma = lam_rv * realized_var_ewma + (1.0 - lam_rv) * (sp_ret ** 2)
        prev_market_vol_t = float(np.sqrt(max(realized_var_ewma, 1e-10)))
        x_prev = x.copy()

        # 取引量 EWMA 更新
        if not vol_trade_ewma_initialized:
            vol_trade_ewma_fast = vol_trade_ewma_slow = total_trade
            vol_trade_ewma_initialized = True
        else:
            lam_f = config.vol_activity_ewma_lambda
            lam_s = max(0.99, config.vol_activity_ewma_lambda)
            vol_trade_ewma_fast = lam_f * vol_trade_ewma_fast + (1.0 - lam_f) * total_trade
            vol_trade_ewma_slow = lam_s * vol_trade_ewma_slow + (1.0 - lam_s) * total_trade

        # 下落 EWMA 更新 — ループ末尾で更新し、次期の leverage 機構に使う
        neg_ret = max(-sp_ret, 0.0)
        down_var_ewma = (
            config.down_ewma_decay * down_var_ewma
            + (1.0 - config.down_ewma_decay) * neg_ret ** 2
        )
        prev_sp_ret = sp_ret

    firms_df = pd.DataFrame({
        "firm_id": np.arange(n),
        "sector": sectors,
        "initial_market_cap_weight": market_caps / market_caps.sum(),
        "true_degree": ((true_w > 0).sum(axis=1) if config.graph_topology != "zero" else np.zeros(n, dtype=int)),
    })

    config_dict = {k: getattr(config, k) for k in Config.__dataclass_fields__}

    return (
        pd.DataFrame(records),
        firms_df,
        investor_df,
        {"config": config_dict, "firm_snapshots": pd.DataFrame(firm_records)},
    )
