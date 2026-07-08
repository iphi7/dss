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

    # 公開成分 s の真の動態
    phi_s: float = 0.82
    rho_s: float = 0.13
    eta_s_sigma: float = 0.020
    obs_sigma: float = 0.045
    rare_shock_prob: float = 0.018
    rare_shock_sigma: float = 0.12

    # 非公開成分 h の真の動態
    phi_h: float = 0.90
    rho_h: float = 0.35
    eta_h_sigma: float = 0.015

    # 価格形成（注文不均衡のみ）
    price_impact: float = 0.0080
    idio_vol: float = 0.0075
    market_vol: float = 0.0060
    common_shock_beta: float = 1.00
    market_garch_alpha: float = 0.080
    market_garch_beta: float = 0.90
    leverage_vol_beta: float = 0.015
    vol_persistence: float = 0.94

    initial_sp500_abs: float | None = None
    initial_dgs10_abs: float | None = None

    # アブレーションフラグ
    # full_obs: 全員が全企業を観測（base2 と同等）
    # partial_obs: 各投資家は投資ユニバース O_i の企業のみ観測
    partial_observation: bool = True


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
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
        graphs: (n_investors, n, n) 主観グラフ
        obs_masks: (n_investors, n) 観測ユニバースのマスク（True=観測可能）
        investor_df: 投資家パラメータ
    """
    n = true_w.shape[0]
    graphs = np.zeros((n_investors, n, n), dtype=float)
    obs_masks = np.zeros((n_investors, n), dtype=bool)
    rows = []

    for i in range(n_investors):
        expertise = int(rng.integers(0, sectors.max() + 1))
        base_keep = rng.uniform(0.15, 0.55)
        expert_bonus = rng.uniform(0.20, 0.40)
        false_edge_prob = rng.uniform(0.002, 0.015)
        risk_tolerance = rng.lognormal(mean=-2.6, sigma=0.35)
        trend_weight = rng.normal(0.28, 0.12)
        value_weight = rng.normal(1.00, 0.25)
        uncertainty_aversion = rng.uniform(0.15, 0.75)
        rate_sensitivity = rng.uniform(0.02, 0.18)
        temperature = rng.uniform(2.0, 6.0)
        loss_asymmetry = rng.uniform(1.0, 1.9)
        belief_phi = np.clip(rng.normal(0.82, 0.05), 0.65, 0.92)
        belief_rho_s = np.clip(rng.normal(0.13, 0.04), 0.04, 0.22)
        belief_rho_h = np.clip(rng.normal(0.30, 0.08), 0.10, 0.55)
        obs_var = rng.lognormal(mean=np.log(0.055**2), sigma=0.45)
        proc_var = rng.lognormal(mean=np.log(0.022**2), sigma=0.45)

        keep_prob = np.full((n, n), base_keep)
        expert_mask = (sectors[:, None] == expertise) | (sectors[None, :] == expertise)
        keep_prob = np.where(expert_mask, np.minimum(0.95, keep_prob + expert_bonus), keep_prob)

        keep = rng.random((n, n)) < keep_prob
        subjective = np.where(keep & (true_w > 0), true_w, 0.0)

        false_edges = (rng.random((n, n)) < false_edge_prob) & (true_w == 0)
        subjective = np.where(false_edges, rng.uniform(0.02, 0.15, size=(n, n)), subjective)
        np.fill_diagonal(subjective, 0.0)
        subjective = row_normalize(subjective)
        graphs[i] = subjective

        # 観測ユニバース O_i:
        #   Step 1: 専門セクター企業（必ず観測）
        #   Step 2: 主観グラフのエッジ重みが大きい上位 top_k 社を追加
        # これにより投資家ごとに明確に異なる観測ユニバースが生まれる
        expert_sector_mask = (sectors == expertise)
        n_expert = expert_sector_mask.sum()

        # グラフ上の「影響力スコア」= 行和 + 列和（どれだけ重要な企業か）
        graph_importance = subjective.sum(axis=1) + subjective.sum(axis=0)
        graph_importance[expert_sector_mask] = 0.0  # 専門セクターは既に含む

        # 非専門セクターから上位 top_k 社を観測対象に追加
        top_k = max(0, 20 - n_expert)   # 専門セクターと合わせて最大20社程度
        if top_k > 0 and graph_importance.sum() > 0:
            top_firms = np.argpartition(graph_importance, -top_k)[-top_k:]
            top_mask = np.zeros(n, dtype=bool)
            top_mask[top_firms] = True
        else:
            top_mask = np.zeros(n, dtype=bool)

        obs_mask_i = expert_sector_mask | top_mask
        obs_masks[i] = obs_mask_i

        rows.append(
            {
                "investor_id": i,
                "expertise_sector": expertise,
                "edge_keep_base": base_keep,
                "risk_tolerance": risk_tolerance,
                "trend_weight": trend_weight,
                "value_weight": value_weight,
                "uncertainty_aversion": uncertainty_aversion,
                "rate_sensitivity": rate_sensitivity,
                "temperature": temperature,
                "loss_asymmetry": loss_asymmetry,
                "belief_phi": belief_phi,
                "belief_rho_s": belief_rho_s,
                "belief_rho_h": belief_rho_h,
                "obs_var": obs_var,
                "proc_var": proc_var,
                "recognized_edges": int((subjective > 0).sum()),
                "n_observed_firms": int(obs_mask_i.sum()),
            }
        )

    return graphs, obs_masks, pd.DataFrame(rows)


def simulate_market(
    output_df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    n = config.n_firms
    t_max = config.n_days

    if len(output_df) < t_max + 2:
        raise ValueError("output.csv is shorter than requested n_days")

    historical_tail = output_df.tail(t_max).reset_index(drop=True)
    last_row = output_df.iloc[-1]
    initial_sp = float(config.initial_sp500_abs or last_row["sp500_abs"])
    initial_dgs10 = float(config.initial_dgs10_abs or historical_tail.iloc[0]["DGS10_abs"])

    dates = pd.bdate_range(pd.to_datetime(last_row["Date"]) + pd.offsets.BDay(1), periods=t_max)

    true_w = make_ba_graph(n, config.ba_m, rng)
    sectors = rng.integers(0, config.n_sectors, size=n)
    subjective_graphs, obs_masks, investor_df = make_subjective_graphs(
        true_w, sectors, config.n_investors, rng
    )

    shares = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    firm_prices = np.full(n, initial_sp) * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    market_caps = firm_prices * shares

    s = rng.normal(0.0, 0.03, size=n)
    h = rng.normal(0.0, 0.02, size=n)
    s_prev = s.copy()

    # 各投資家が持つ「最後に観測した y」（部分観測のとき未観測企業は前期値のまま残る）
    y_known = np.zeros((config.n_investors, n), dtype=float)

    cash = np.full(config.n_investors, 1.0, dtype=float)
    holdings = rng.lognormal(mean=-5.0, sigma=0.4, size=(config.n_investors, n))

    firm_vol = np.full(n, config.idio_vol)
    market_var = config.market_vol ** 2
    prev_common_noise = 0.0
    sp_abs = initial_sp
    records = []
    firm_records = []

    params = investor_df.to_dict("records")

    for t in range(t_max):
        dgs10_abs = float(historical_tail.loc[t, "DGS10_abs"])
        dgs10_change = float(historical_tail.loc[t, "DGS10"])
        rf_level = dgs10_abs / 100.0

        # --- 真の状態遷移 ---
        rare = rng.random(n) < config.rare_shock_prob
        eta_s = rng.normal(0.0, config.eta_s_sigma, size=n)
        eta_s += rare * rng.standard_t(df=3, size=n) * config.rare_shock_sigma
        s = config.phi_s * s + config.rho_s * (true_w @ s) + eta_s

        eta_h = rng.normal(0.0, config.eta_h_sigma, size=n)
        h = config.phi_h * h + config.rho_h * (true_w @ h) + eta_h

        # 全企業の真の観測値（投資家には O_i の分しか届かない）
        y_full = s + 0.35 * (s - s_prev) + rng.normal(0.0, config.obs_sigma, size=n)

        buy_value = np.zeros(n, dtype=float)
        sell_value = np.zeros(n, dtype=float)

        momentum = np.zeros(n) if t == 0 else firm_return

        for i, p in enumerate(params):
            if config.partial_observation:
                # 投資家 i が観測できるのは O_i の企業のみ
                # 未観測企業は前期の y_known[i] をそのまま使う（陳腐化した情報）
                mask = obs_masks[i]
                y_known[i] = np.where(mask, y_full, y_known[i])
                y_i = y_known[i]
            else:
                # 全観測（base2 と同等）
                y_i = y_full

            # 非公開成分の推定（グラフ経由）
            est_h = p["belief_rho_h"] * (subjective_graphs[i] @ y_i)
            total_est = y_i + est_h

            # 来期予測
            pred_next = p["belief_phi"] * total_est + p["belief_rho_s"] * (subjective_graphs[i] @ total_est)

            delta = pred_next - y_i
            uncertainty = np.sqrt(p["obs_var"] + p["proc_var"])
            score = (
                p["value_weight"] * pred_next
                + p["trend_weight"] * delta
                - p["uncertainty_aversion"] * uncertainty
                - p["rate_sensitivity"] * rf_level
                + 0.25 * momentum
            )

            z_buy = np.exp(np.clip(p["temperature"] * score, -20, 20))
            z_sell = np.exp(np.clip(-p["loss_asymmetry"] * p["temperature"] * score, -20, 20))
            denom = z_buy + z_sell + 1.0
            p_buy = z_buy / denom
            p_sell = z_sell / denom

            actions = rng.random(n)
            buy_mask = actions < p_buy
            sell_mask = (actions >= p_buy) & (actions < p_buy + p_sell)

            conviction = np.minimum(1.0, np.abs(score) / 0.12)
            size_frac = p["risk_tolerance"] * (0.25 + conviction)
            size_frac *= rng.lognormal(mean=0.0, sigma=0.45, size=n)
            size_frac = np.clip(size_frac, 0.0002, 0.080)

            buy_orders = buy_mask * cash[i] * size_frac
            sell_orders = sell_mask * holdings[i] * firm_prices * size_frac

            total_buy = buy_orders.sum()
            if total_buy > cash[i]:
                buy_orders *= cash[i] / (total_buy + 1e-12)

            buy_value += buy_orders
            sell_value += sell_orders

            cash[i] += sell_orders.sum() - buy_orders.sum()
            holdings[i] += buy_orders / firm_prices
            holdings[i] -= sell_orders / firm_prices
            holdings[i] = np.maximum(holdings[i], 0.0)

        imbalance = (buy_value - sell_value) / (buy_value + sell_value + 1e-9)

        # --- 価格形成（注文不均衡のみ）---
        market_stress = float(np.mean(np.abs(imbalance)))
        downside_stress = float(np.maximum(-np.mean(imbalance), 0.0))
        base_var = (config.market_vol ** 2) * (1.0 + 6.0 * market_stress + 2.0 * downside_stress)
        market_var = (
            (1.0 - config.market_garch_alpha - config.market_garch_beta) * base_var
            + config.market_garch_alpha * prev_common_noise ** 2
            + config.market_garch_beta * market_var
            + config.leverage_vol_beta * max(-prev_common_noise, 0.0) ** 2
        )
        market_var = min(max(market_var, 1e-8), 0.012 ** 2)
        market_vol = float(np.sqrt(market_var))
        common_noise = config.common_shock_beta * rng.standard_t(df=5) * market_vol

        firm_vol = (
            config.vol_persistence * firm_vol
            + (1.0 - config.vol_persistence) * config.idio_vol
        )
        noise = rng.standard_t(df=5, size=n) * firm_vol

        firm_return = (
            config.price_impact * imbalance
            + common_noise
            + noise
        )
        firm_return = np.clip(firm_return, -0.18, 0.18)

        firm_prices = firm_prices * (1.0 + firm_return)
        firm_prices = np.maximum(firm_prices, 1e-3)
        market_caps = firm_prices * shares
        weights = market_caps / market_caps.sum()
        sp_ret = float(np.sum(weights * firm_return))
        sp_abs = float(sp_abs * (1.0 + sp_ret))

        records.append(
            {
                "path_id": 0,
                "Date": dates[t].strftime("%Y-%m-%d"),
                "sp500_abs": sp_abs,
                "DGS10_abs": dgs10_abs if t > 0 else initial_dgs10,
                "sp500": sp_ret,
                "DGS10": dgs10_change,
            }
        )

        if t in {0, 1, 2, 20, 60, 252, 504, 756, 1008, t_max - 1}:
            for j in range(n):
                firm_records.append(
                    {
                        "day": t,
                        "firm_id": j,
                        "sector": int(sectors[j]),
                        "s_state": float(s[j]),
                        "h_state": float(h[j]),
                        "observation": float(y_full[j]),
                        "price": float(firm_prices[j]),
                        "return": float(firm_return[j]),
                        "market_weight": float(weights[j]),
                        "imbalance": float(imbalance[j]),
                    }
                )

        prev_common_noise = float(common_noise)
        s_prev = s.copy()

    firms_df = pd.DataFrame(
        {
            "firm_id": np.arange(n),
            "sector": sectors,
            "initial_market_cap_weight": market_caps / market_caps.sum(),
            "true_degree": (true_w > 0).sum(axis=1),
        }
    )

    config_dict = {k: getattr(config, k) for k in Config.__dataclass_fields__}

    return pd.DataFrame(records), firms_df, investor_df, {
        "config": config_dict,
        "firm_snapshots": pd.DataFrame(firm_records),
    }
