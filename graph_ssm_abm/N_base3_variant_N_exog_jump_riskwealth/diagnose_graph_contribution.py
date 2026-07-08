
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import Config, make_ba_graph, make_subjective_graphs


def n19_config() -> Config:
    return Config(
        vol_sensitivity_mean=0.80,
        vol_sensitivity_std=0.80,
        wealth_sigma=1.20,
        wealth_vol_corr=1.20,
        participation_vol_power=1.8,
        price_impact=0.080,
        impact_activity_scale=3.0,
        impact_activity_clip=4.0,
        exog_common_sigma=0.0060,
        exog_common_jump_prob=0.001,
        exog_common_jump_sigma=0.015,
        realized_vol_lambda=0.985,
    )


def safe_ratio(num, den):
    return float(np.nanmean(np.abs(num) / (np.abs(den) + 1e-12)))


def q(arr, qs=(0.1, 0.5, 0.9, 0.99)):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return {f"q{int(x*100):02d}": float(np.quantile(a, x)) for x in qs}


def main():
    config = n19_config()
    output_df = pd.read_csv("output.csv")
    rng = np.random.default_rng(config.seed)
    n = config.n_firms
    t_max = config.n_days
    n_pub, n_sec, n_priv = config.n_pub, config.n_sec, config.n_priv
    d = n_pub + n_sec + n_priv

    historical_tail = output_df.tail(t_max).reset_index(drop=True)

    true_w = make_ba_graph(n, config.ba_m, rng)
    sectors = rng.integers(0, config.n_sectors, size=n)
    subjective_graphs, investor_df = make_subjective_graphs(
        true_w, sectors, config.n_investors, rng,
        vol_sensitivity_mean=config.vol_sensitivity_mean,
        vol_sensitivity_std=config.vol_sensitivity_std,
    )

    # Advance RNG exactly like simulate_market for wealth and firm setup.
    vol_s = investor_df["vol_sensitivity"].to_numpy(dtype=float)
    vol_s_z = (vol_s - vol_s.mean()) / (vol_s.std() + 1e-12)
    wealth_log = rng.normal(0.0, config.wealth_sigma, size=config.n_investors)
    wealth_log += config.wealth_vol_corr * vol_s_z
    wealth_factor = np.exp(wealth_log)
    wealth_factor = np.clip(wealth_factor, config.wealth_clip_min, config.wealth_clip_max)
    wealth_factor = wealth_factor / wealth_factor.mean()

    degree = (true_w > 0).sum(axis=1).astype(float)
    base_shares = rng.pareto(a=config.mktcap_pareto_a, size=n) + 1.0
    degree_factor = (degree / degree.mean()) ** config.mktcap_degree_power
    shares = base_shares * degree_factor
    shares = shares / shares.mean()

    last_row = output_df.iloc[-1]
    initial_sp = float(config.initial_sp500_abs or last_row["sp500_abs"])
    firm_prices = np.full(n, initial_sp) * rng.lognormal(mean=0.0, sigma=0.12, size=n)

    phi = np.array(config.phi_dims)
    rho = np.array(config.rho_dims)
    eta = np.array(config.eta_dims)
    dim_w = np.array(config.dim_weights)
    priv_weight_sum = float(dim_w[n_pub+n_sec:].sum())

    x = rng.normal(0.0, 0.020, size=(n, d))
    x_prev = x.copy()
    cash = wealth_factor.astype(float).copy()
    holdings = rng.lognormal(mean=-5.0, sigma=0.4, size=(config.n_investors, n)) * wealth_factor[:, None]

    firm_vol = np.full(n, config.idio_vol)
    prev_market_vol_t = config.market_vol
    realized_var_ewma = config.market_vol ** 2
    vol_trade_ewma_fast = 0.0
    vol_trade_ewma_slow = 0.0
    vol_trade_ewma_initialized = False
    sp_ret_prev = 0.0

    params = investor_df.to_dict("records")
    rows = []
    daily_rows = []

    for t in range(t_max):
        dgs10_abs = float(historical_tail.loc[t, "DGS10_abs"])
        rf_level = dgs10_abs / 100.0

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

        buy_value = np.zeros(n)
        sell_value = np.zeros(n)

        # diagnostics accumulators per day
        day_score_diff = []
        day_score_abs = []
        day_score_nograph_abs = []
        day_score_graph_abs = []
        day_action_prob = []
        day_action_prob_nograph = []
        day_pdiff = []
        day_weighted_pdiff = []
        day_priv_ratio = []
        day_pred_ratio = []
        day_score_graph_ratio = []
        day_cross_inv_std = []
        total_est_stack = []
        total_est_nograph_stack = []

        for i, p in enumerate(params):
            expertise = int(p["expertise_sector"])
            expert_mask = (sectors == expertise)

            est_pub = y_pub @ dim_w[:n_pub]
            y_i_sec = y_sec * expert_mask[:, None]
            est_sec = y_i_sec @ dim_w[n_pub:n_pub+n_sec]

            y_pub_scalar = y_pub.mean(axis=1)
            neighbor_signal = subjective_graphs[i] @ y_pub_scalar
            est_priv = priv_weight_sum * p["belief_rho_h"] * neighbor_signal
            total_est = est_pub + est_sec + est_priv
            total_est_no_priv = est_pub + est_sec

            graph_pred_term = p["belief_rho_s"] * (subjective_graphs[i] @ total_est)
            self_pred_term = p["belief_phi"] * total_est
            pred_next = self_pred_term + graph_pred_term
            pred_next_nograph = p["belief_phi"] * total_est_no_priv

            delta = pred_next - total_est
            delta_nograph = pred_next_nograph - total_est_no_priv
            uncertainty = np.sqrt(p["obs_var"] + p["proc_var"])
            non_graph_const = -p["uncertainty_aversion"] * uncertainty - p["rate_sensitivity"] * rf_level + 0.25 * sp_ret_prev
            score = p["value_weight"] * pred_next + p["trend_weight"] * delta + non_graph_const
            score_nograph = p["value_weight"] * pred_next_nograph + p["trend_weight"] * delta_nograph + non_graph_const

            # Approximate direct graph score component by subtraction.
            score_graph_component = score - score_nograph

            vol_ratio = prev_market_vol_t / config.market_vol
            vol_factor = float(np.clip(1.0 + p["vol_sensitivity"] * (vol_ratio - 1.0), 0.05, 4.0))
            participation_factor = float(np.clip(vol_factor ** config.participation_vol_power, 0.10, 5.0))

            def probs(sc):
                z_buy = np.exp(np.clip(p["temperature"] * sc, -20, 20)) * participation_factor
                z_sell = np.exp(np.clip(-p["loss_asymmetry"] * p["temperature"] * sc, -20, 20)) * participation_factor
                denom = z_buy + z_sell + 1.0
                return z_buy / denom, z_sell / denom

            p_buy, p_sell = probs(score)
            p_buy0, p_sell0 = probs(score_nograph)

            day_priv_ratio.append(safe_ratio(est_priv, total_est_no_priv))
            day_pred_ratio.append(safe_ratio(graph_pred_term, self_pred_term))
            day_score_graph_ratio.append(safe_ratio(score_graph_component, score_nograph))
            day_score_diff.append(float(np.mean(np.abs(score - score_nograph))))
            day_score_abs.append(float(np.mean(np.abs(score))))
            day_score_nograph_abs.append(float(np.mean(np.abs(score_nograph))))
            day_score_graph_abs.append(float(np.mean(np.abs(score_graph_component))))
            day_action_prob.append(float(np.mean(p_buy + p_sell)))
            day_action_prob_nograph.append(float(np.mean(p_buy0 + p_sell0)))
            pdiff = np.mean(np.abs(p_buy - p_buy0) + np.abs(p_sell - p_sell0))
            day_pdiff.append(float(pdiff))
            day_weighted_pdiff.append(float(wealth_factor[i] * pdiff))
            total_est_stack.append(total_est)
            total_est_nograph_stack.append(total_est_no_priv)

            # Keep RNG path close to simulate_market by sampling actions and updating wealth.
            actions = rng.random(n)
            buy_mask = actions < p_buy
            sell_mask = (actions >= p_buy) & (actions < p_buy + p_sell)
            conviction = np.minimum(1.0, np.abs(score) / 0.12)
            size_frac = p["risk_tolerance"] * vol_factor * (0.25 + conviction)
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

        total_est_arr = np.stack(total_est_stack, axis=0)
        total_est_no_graph_arr = np.stack(total_est_nograph_stack, axis=0)
        investor_mean_graph = total_est_arr.mean(axis=0)
        investor_mean_nograph = total_est_no_graph_arr.mean(axis=0)
        cross_inv_std_graph = total_est_arr.std(axis=0).mean()
        cross_inv_std_nograph = total_est_no_graph_arr.std(axis=0).mean()
        mean_shift = np.mean(np.abs(investor_mean_graph - investor_mean_nograph))
        mean_level = np.mean(np.abs(investor_mean_nograph)) + 1e-12

        imbalance = (buy_value - sell_value) / (buy_value + sell_value + 1e-9)
        total_trade = float(buy_value.sum() + sell_value.sum())
        if vol_trade_ewma_initialized:
            volume_ratio = vol_trade_ewma_fast / max(vol_trade_ewma_slow, 1e-12)
        else:
            volume_ratio = 1.0

        common_noise = rng.normal(0.0, config.exog_common_sigma)
        if rng.random() < config.exog_common_jump_prob:
            common_noise += rng.standard_t(df=3) * config.exog_common_jump_sigma
        common_noise = float(np.clip(config.common_shock_beta * common_noise, -config.exog_common_clip, config.exog_common_clip))
        firm_vol = config.vol_persistence * firm_vol + (1.0 - config.vol_persistence) * config.idio_vol
        noise = rng.standard_t(df=5, size=n) * firm_vol
        impact_factor = float(np.clip(1.0 + config.impact_activity_scale * max(0.0, volume_ratio - 1.0), 0.25, config.impact_activity_clip))
        firm_return = config.price_impact * impact_factor * imbalance + common_noise + noise
        firm_return = np.clip(firm_return, -0.18, 0.18)
        firm_prices = np.maximum(firm_prices * (1.0 + firm_return), 1e-3)
        market_caps = firm_prices * shares
        weights = market_caps / market_caps.sum()
        sp_ret = float(np.sum(weights * firm_return))

        lam_rv = config.realized_vol_lambda
        realized_var_ewma = lam_rv * realized_var_ewma + (1.0 - lam_rv) * (sp_ret ** 2)
        prev_market_vol_t = float(np.sqrt(max(realized_var_ewma, 1e-10)))

        if not vol_trade_ewma_initialized:
            vol_trade_ewma_fast = vol_trade_ewma_slow = total_trade
            vol_trade_ewma_initialized = True
        else:
            lam_f = config.vol_activity_ewma_lambda
            lam_s = max(0.99, config.vol_activity_ewma_lambda)
            vol_trade_ewma_fast = lam_f * vol_trade_ewma_fast + (1.0 - lam_f) * total_trade
            vol_trade_ewma_slow = lam_s * vol_trade_ewma_slow + (1.0 - lam_s) * total_trade

        x_prev = x.copy()
        sp_ret_prev = sp_ret

        daily_rows.append({
            "day": t,
            "priv_over_pubsec": float(np.mean(day_priv_ratio)),
            "pred_graph_over_self": float(np.mean(day_pred_ratio)),
            "score_graph_over_nograph": float(np.mean(day_score_graph_ratio)),
            "mean_abs_score": float(np.mean(day_score_abs)),
            "mean_abs_score_nograph": float(np.mean(day_score_nograph_abs)),
            "mean_abs_score_graph_component": float(np.mean(day_score_graph_abs)),
            "mean_abs_score_diff": float(np.mean(day_score_diff)),
            "mean_action_prob": float(np.mean(day_action_prob)),
            "mean_action_prob_nograph": float(np.mean(day_action_prob_nograph)),
            "mean_prob_l1_diff": float(np.mean(day_pdiff)),
            "wealth_weighted_prob_l1_diff": float(np.mean(day_weighted_pdiff)),
            "mean_est_shift_ratio_after_investor_avg": float(mean_shift / mean_level),
            "cross_inv_std_graph": float(cross_inv_std_graph),
            "cross_inv_std_nograph": float(cross_inv_std_nograph),
            "sp_ret": sp_ret,
        })

    daily = pd.DataFrame(daily_rows)
    summary = {}
    for col in daily.columns:
        if col == "day":
            continue
        vals = daily[col].to_numpy(dtype=float)
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            **q(vals),
        }

    out_dir = Path("graph_ssm_abm/N_base3_variant_N_exog_jump_riskwealth/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "N19_graph_contribution_daily.csv", index=False)
    with open(out_dir / "N19_graph_contribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
