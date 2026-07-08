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
    true_phi: float = 0.82
    true_rho: float = 0.13
    process_sigma: float = 0.020
    obs_sigma: float = 0.045
    rare_shock_prob: float = 0.018
    rare_shock_sigma: float = 0.12
    price_impact: float = 0.0060
    latent_price_beta: float = 0.0022
    idio_vol: float = 0.0075
    market_vol: float = 0.0060
    common_shock_beta: float = 1.00
    market_garch_alpha: float = 0.080
    market_garch_beta: float = 0.90
    leverage_vol_beta: float = 0.015
    vol_persistence: float = 0.94
    vol_latent_beta: float = 0.35
    initial_sp500_abs: float | None = None
    initial_dgs10_abs: float | None = None
    dynamic_observation_uncertainty: bool = False
    obs_abs_state_beta: float = 2.0
    obs_abs_delta_beta: float = 3.0
    neighbor_sell_pressure: bool = False
    neighbor_sell_weight_scale: float = 1.4
    exponential_memory: bool = False
    firm_memory_decay: float = 0.94
    firm_memory_state_weight: float = 0.7
    firm_memory_delta_weight: float = 1.2
    firm_memory_shock_weight: float = 1.8
    firm_memory_obs_beta: float = 1.2
    firm_memory_impact_beta: float = 2.5
    firm_memory_vol_beta: float = 0.8
    firm_memory_process_beta: float = 0.0
    firm_memory_network_beta: float = 0.0
    investor_memory_decay: float = 0.92
    investor_memory_surprise_weight: float = 1.0
    investor_memory_var_weight: float = 0.5
    investor_memory_size_beta: float = 1.5
    memory_cap: float = 5.0


def make_ba_graph(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Small Barabasi-Albert graph without networkx."""
    if m < 1 or m >= n:
        raise ValueError("ba_m must satisfy 1 <= m < n_firms")

    adj = np.zeros((n, n), dtype=float)
    degrees = np.zeros(n, dtype=float)

    # Start from a clique of size m + 1.
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
) -> Tuple[np.ndarray, pd.DataFrame]:
    n = true_w.shape[0]
    graphs = np.zeros((n_investors, n, n), dtype=float)
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
        neighbor_sell_weight = rng.uniform(0.4, 1.8)
        belief_phi = np.clip(rng.normal(0.80, 0.05), 0.65, 0.92)
        belief_rho = np.clip(rng.normal(0.12, 0.04), 0.04, 0.22)
        obs_var = rng.lognormal(mean=np.log(0.055**2), sigma=0.45)
        proc_var = rng.lognormal(mean=np.log(0.022**2), sigma=0.45)

        keep_prob = np.full((n, n), base_keep)
        expert_mask = (sectors[:, None] == expertise) | (sectors[None, :] == expertise)
        keep_prob = np.where(expert_mask, np.minimum(0.95, keep_prob + expert_bonus), keep_prob)

        keep = rng.random((n, n)) < keep_prob
        subjective = np.where(keep, true_w, 0.0)

        # Add a few weak false beliefs; this makes "wrong causal structure" explicit.
        false_edges = (rng.random((n, n)) < false_edge_prob) & (true_w == 0)
        subjective = np.where(false_edges, rng.uniform(0.02, 0.15, size=(n, n)), subjective)
        np.fill_diagonal(subjective, 0.0)
        subjective = row_normalize(subjective)
        graphs[i] = subjective

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
                "neighbor_sell_weight": neighbor_sell_weight,
                "belief_phi": belief_phi,
                "belief_rho": belief_rho,
                "obs_var": obs_var,
                "proc_var": proc_var,
                "recognized_edges": int((subjective > 0).sum()),
            }
        )

    return graphs, pd.DataFrame(rows)


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
    subjective_graphs, investor_df = make_subjective_graphs(
        true_w, sectors, config.n_investors, rng
    )

    firm_quality = rng.normal(0.0, 0.25, size=n)
    shares = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    firm_prices = np.full(n, initial_sp) * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    market_caps = firm_prices * shares

    x = rng.normal(0.0, 0.03, size=n)
    x_prev = x.copy()
    obs = x + rng.normal(0.0, config.obs_sigma, size=n)

    estimates = np.zeros((config.n_investors, n), dtype=float)
    prev_estimates = np.zeros_like(estimates)
    variances = np.full((config.n_investors, n), 0.08**2, dtype=float)
    cash = np.full(config.n_investors, 1.0, dtype=float)
    holdings = rng.lognormal(mean=-5.0, sigma=0.4, size=(config.n_investors, n))

    firm_uncertainty = np.zeros(n, dtype=float)
    investor_uncertainty = np.zeros((config.n_investors, n), dtype=float)

    firm_vol = np.full(n, config.idio_vol)
    market_vol = config.market_vol
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

        # True latent propagation.
        eta_scale = config.process_sigma
        rare_scale = config.rare_shock_sigma
        if config.exponential_memory and config.firm_memory_process_beta > 0:
            eta_multiplier = 1.0 + config.firm_memory_process_beta * firm_uncertainty
            eta_multiplier = np.clip(eta_multiplier, 1.0, 6.0)
            eta_scale = config.process_sigma * eta_multiplier
            rare_scale = config.rare_shock_sigma * eta_multiplier
        rare = rng.random(n) < config.rare_shock_prob
        eta = rng.normal(0.0, eta_scale, size=n)
        eta += rare * rng.standard_t(df=3, size=n) * rare_scale
        x = config.true_phi * x + config.true_rho * (true_w @ x) + eta
        delta_x_true = x - x_prev
        if config.exponential_memory:
            shock_signal = (
                config.firm_memory_state_weight * np.abs(x) / 0.05
                + config.firm_memory_delta_weight * np.abs(delta_x_true) / 0.03
                + config.firm_memory_shock_weight * np.abs(eta) / (config.process_sigma + 1e-9)
            )
            shock_signal = np.clip(shock_signal, 0.0, config.memory_cap)
            neighbor_uncertainty = true_w @ firm_uncertainty
            firm_uncertainty = (
                config.firm_memory_decay * firm_uncertainty
                + (1.0 - config.firm_memory_decay) * shock_signal
                + config.firm_memory_network_beta * neighbor_uncertainty
            )
            firm_uncertainty = np.clip(firm_uncertainty, 0.0, config.memory_cap)

        obs_var_factor = np.ones(n, dtype=float)
        if config.dynamic_observation_uncertainty:
            obs_var_factor = 1.0 + config.obs_abs_state_beta * np.abs(x) + config.obs_abs_delta_beta * np.abs(delta_x_true)
            obs_var_factor = np.clip(obs_var_factor, 1.0, 8.0)
        if config.exponential_memory:
            obs_var_factor = obs_var_factor * (1.0 + config.firm_memory_obs_beta * firm_uncertainty)
            obs_var_factor = np.clip(obs_var_factor, 1.0, 20.0)
        y = x + 0.35 * delta_x_true + rng.normal(0.0, config.obs_sigma * np.sqrt(obs_var_factor), size=n)

        buy_value = np.zeros(n, dtype=float)
        sell_value = np.zeros(n, dtype=float)

        momentum = np.zeros(n) if t == 0 else firm_return

        for i, p in enumerate(params):
            a_i = p["belief_phi"] * np.eye(n) + p["belief_rho"] * subjective_graphs[i]
            pred = a_i @ estimates[i]
            pred_var = p["belief_phi"] ** 2 * variances[i]
            pred_var += p["belief_rho"] ** 2 * (subjective_graphs[i] ** 2 @ variances[i])
            pred_var += p["proc_var"]

            effective_obs_var = p["obs_var"] * obs_var_factor
            gain = pred_var / (pred_var + effective_obs_var)
            updated = pred + gain * (y - pred)
            updated_var = (1.0 - gain) * pred_var

            if config.exponential_memory:
                surprise_signal = np.abs(y - pred) / (config.obs_sigma * np.sqrt(obs_var_factor) + 1e-9)
                belief_uncertainty = np.sqrt(np.maximum(updated_var, 0.0)) / 0.08
                inv_signal = (
                    config.investor_memory_surprise_weight * surprise_signal
                    + config.investor_memory_var_weight * belief_uncertainty
                )
                inv_signal = np.clip(inv_signal, 0.0, config.memory_cap)
                investor_uncertainty[i] = (
                    config.investor_memory_decay * investor_uncertainty[i]
                    + (1.0 - config.investor_memory_decay) * inv_signal
                )
                investor_uncertainty[i] = np.clip(investor_uncertainty[i], 0.0, config.memory_cap)

            delta = updated - estimates[i]
            prev_estimates[i] = estimates[i]
            estimates[i] = updated
            variances[i] = updated_var

            score = (
                p["value_weight"] * updated
                + p["trend_weight"] * delta
                - p["uncertainty_aversion"] * np.sqrt(np.maximum(updated_var, 0.0))
                - p["rate_sensitivity"] * rf_level
                + 0.25 * momentum
            )

            z_buy = np.exp(np.clip(p["temperature"] * score, -20, 20))
            neighbor_bad = np.zeros(n, dtype=float)
            if config.neighbor_sell_pressure:
                neighbor_bad = subjective_graphs[i] @ np.maximum(-updated, 0.0)
            sell_logit = -p["loss_asymmetry"] * p["temperature"] * score
            sell_logit += config.neighbor_sell_weight_scale * p["neighbor_sell_weight"] * neighbor_bad
            z_sell = np.exp(np.clip(sell_logit, -20, 20))
            denom = z_buy + z_sell + 1.0
            p_buy = z_buy / denom
            p_sell = z_sell / denom

            actions = rng.random(n)
            buy_mask = actions < p_buy
            sell_mask = (actions >= p_buy) & (actions < p_buy + p_sell)

            conviction = np.minimum(1.0, np.abs(score) / 0.12)
            size_frac = p["risk_tolerance"] * (0.25 + conviction)
            size_frac *= rng.lognormal(mean=0.0, sigma=0.45, size=n)
            if config.exponential_memory:
                size_frac *= (1.0 + config.investor_memory_size_beta * investor_uncertainty[i])
            size_frac = np.clip(size_frac, 0.0002, 0.160 if config.exponential_memory else 0.080)

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

        # Volatility clusters when latent stress and order imbalance are large.
        market_stress = float(np.mean(np.abs(x)) + 0.6 * abs(np.mean(imbalance)))
        downside_stress = float(np.maximum(-np.mean(x), 0.0))
        base_var = (config.market_vol ** 2) * (1.0 + 10.0 * market_stress + 3.0 * downside_stress)
        market_var = (
            (1.0 - config.market_garch_alpha - config.market_garch_beta) * base_var
            + config.market_garch_alpha * prev_common_noise ** 2
            + config.market_garch_beta * market_var
            + config.leverage_vol_beta * max(-prev_common_noise, 0.0) ** 2
        )
        market_var = min(max(market_var, 1e-8), 0.012 ** 2)
        market_vol = float(np.sqrt(market_var))
        common_noise = config.common_shock_beta * rng.standard_t(df=5) * market_vol

        firm_memory_multiplier = 1.0
        if config.exponential_memory:
            firm_memory_multiplier = 1.0 + config.firm_memory_vol_beta * firm_uncertainty

        firm_vol = (
            config.vol_persistence * firm_vol
            + (1.0 - config.vol_persistence)
            * config.idio_vol
            * (1.0 + config.vol_latent_beta * np.abs(x) / (np.std(x) + 1e-9))
            * firm_memory_multiplier
        )
        noise = rng.standard_t(df=5, size=n) * firm_vol
        impact_multiplier = 1.0
        if config.exponential_memory:
            impact_multiplier = 1.0 + config.firm_memory_impact_beta * firm_uncertainty
            impact_multiplier = np.clip(impact_multiplier, 1.0, 8.0)

        firm_return = (
            config.price_impact * impact_multiplier * imbalance
            + config.latent_price_beta * x
            - 0.0018 * np.maximum(-x, 0.0)
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
                        "latent_state": float(x[j]),
                        "observation": float(y[j]),
                        "price": float(firm_prices[j]),
                        "return": float(firm_return[j]),
                        "market_weight": float(weights[j]),
                        "buy_value": float(buy_value[j]),
                        "sell_value": float(sell_value[j]),
                        "imbalance": float(imbalance[j]),
                        "firm_uncertainty": float(firm_uncertainty[j]),
                        "obs_var_factor": float(obs_var_factor[j]),
                    }
                )

        prev_common_noise = float(common_noise)
        x_prev = x.copy()

    firms_df = pd.DataFrame(
        {
            "firm_id": np.arange(n),
            "sector": sectors,
            "initial_market_cap_weight": market_caps / market_caps.sum(),
            "true_degree": (true_w > 0).sum(axis=1),
            "quality": firm_quality,
        }
    )

    config_dict = {
        k: getattr(config, k)
        for k in Config.__dataclass_fields__
    }

    return pd.DataFrame(records), firms_df, investor_df, {
        "config": config_dict,
        "firm_snapshots": pd.DataFrame(firm_records),
    }

