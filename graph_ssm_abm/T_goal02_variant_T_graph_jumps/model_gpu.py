"""
P_base5_variant_P_leverage/model_gpu.py

model.py の PyTorch GPU 版。
投資家ループ (60人 × 1260日) を torch.bmm などのバッチ演算に置き換え、
RTX 5000 での高速実行を実現する。

主な変更点:
  - belief_state: (n_investors, n_firms, d) の GPU テンソル
  - subjective_graphs: (n_investors, n_firms, n_firms) の GPU テンソル
  - 投資家ループを bmm / 行列演算に置き換え
  - 乱数生成を torch.rand / torch.randn (GPU) で高速化

CPU 版 model.py と同じ Config・同じ出力形式を共有する。
"""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from model import Config, make_ba_graph, row_normalize  # CPU 版から共用


def _make_subjective_graphs_gpu(
    true_w: np.ndarray,
    sectors: np.ndarray,
    n_investors: int,
    rng: np.random.Generator,
    vol_sensitivity_mean: float,
    vol_sensitivity_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """主観グラフを生成し GPU テンソルで返す。投資家属性は DataFrame で返す。"""
    n = true_w.shape[0]
    graphs_np = np.zeros((n_investors, n, n), dtype=np.float32)
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
        graphs_np[i] = subjective.astype(np.float32)

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

    graphs_t = torch.from_numpy(graphs_np).to(device)
    return graphs_t, pd.DataFrame(rows)


def simulate_market_gpu(
    output_df: pd.DataFrame,
    config: Config,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """GPU 加速版シミュレーション。CPU 版 simulate_market と同じ出力を返す。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU乱数状態をseedでリセット → 実行順序に依存しない再現性を保証
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

    rng = np.random.default_rng(config.seed)
    # graph event専用RNG。R03本体の乱数列を変えず、イベントだけを重ねる。
    event_rng = np.random.default_rng(config.seed + 7_654_321)
    n = config.n_firms
    t_max = config.n_days
    n_pub  = config.n_pub
    n_sec  = config.n_sec
    n_priv = config.n_priv
    n_inv  = config.n_investors
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

    # ---- グラフ生成 ----
    ba_reference_w = make_ba_graph(n, config.ba_m, rng)
    if config.graph_topology == "zero":
        true_w = np.zeros((n, n), dtype=float)
    else:
        true_w = ba_reference_w.copy()

    sectors = rng.integers(0, config.n_sectors, size=n)
    subjective_graphs, investor_df = _make_subjective_graphs_gpu(
        true_w, sectors, n_inv, rng,
        vol_sensitivity_mean=config.vol_sensitivity_mean,
        vol_sensitivity_std=config.vol_sensitivity_std,
        device=device,
    )
    if config.graph_topology == "zero":
        subjective_graphs = torch.zeros_like(subjective_graphs)
        investor_df["recognized_edges"] = 0
    elif config.subjective_graph_mode == "perfect":
        true_w_t = torch.from_numpy(true_w.astype(np.float32)).to(device)
        subjective_graphs = true_w_t.unsqueeze(0).expand(n_inv, -1, -1).clone()
        investor_df["recognized_edges"] = int((true_w > 0).sum())

    # graph_quality (CPU で計算)
    if np.mean(np.abs(true_w)) > 0:
        denom_w = np.mean(np.abs(true_w)) + 1e-12
        subj_np = subjective_graphs.cpu().numpy()
        gq = [float(np.clip(1.0 - np.mean(np.abs(subj_np[i] - true_w)) / denom_w, 0, 1))
              for i in range(n_inv)]
        investor_df["graph_quality"] = gq
    else:
        investor_df["graph_quality"] = 0.0

    investor_df["update_style"] = np.clip(
        rng.lognormal(mean=0.0, sigma=config.update_style_sigma, size=n_inv), 0.5, 1.5
    )

    # ---- 投資家属性 → GPU テンソル化 ----
    vol_s = investor_df["vol_sensitivity"].to_numpy(dtype=np.float32)
    vol_s_z = (vol_s - vol_s.mean()) / (vol_s.std() + 1e-12)
    wealth_log = rng.normal(0.0, config.wealth_sigma, size=n_inv).astype(np.float32)
    wealth_log += config.wealth_vol_corr * vol_s_z
    wealth_factor = np.clip(np.exp(wealth_log), config.wealth_clip_min, config.wealth_clip_max)
    wealth_factor = (wealth_factor / wealth_factor.mean()).astype(np.float32)
    investor_df["wealth_factor"] = wealth_factor

    # 時価総額 (CPU)
    if config.mktcap_degree_reference == "true":
        deg_mc = (true_w > 0).sum(axis=1).astype(float)
    else:
        deg_mc = (ba_reference_w > 0).sum(axis=1).astype(float)
    base_shares = (rng.pareto(a=config.mktcap_pareto_a, size=n) + 1.0).astype(np.float32)
    deg_factor = (deg_mc / max(deg_mc.mean(), 1e-12)) ** config.mktcap_degree_power if deg_mc.mean() > 0 else np.ones(n)
    shares_np = (base_shares * deg_factor).astype(np.float32)
    shares_np = shares_np / shares_np.mean()

    firm_prices_np = (np.full(n, initial_sp) * rng.lognormal(0.0, 0.12, size=n)).astype(np.float32)

    # GPU テンソルに変換
    def T(a): return torch.from_numpy(a).to(device)

    shares       = T(shares_np)          # (n,)
    firm_prices  = T(firm_prices_np)     # (n,)
    # 長期安定化用の企業別ファンダメンタル価値。初期価格近傍から開始し、ゆっくり成長する。
    fundamental_np = (firm_prices_np * rng.lognormal(0.0, 0.05, size=n)).astype(np.float32)
    fundamental_prices = T(fundamental_np)
    market_caps  = firm_prices * shares  # (n,)

    phi_dims = T(np.array(config.phi_dims, dtype=np.float32))    # (d,)
    rho_dims = T(np.array(config.rho_dims, dtype=np.float32))    # (d,)
    eta_dims = T(np.array(config.eta_dims, dtype=np.float32))    # (d,)
    dim_w    = T(np.array(config.dim_weights, dtype=np.float32)) # (d,)
    true_w_t = T(true_w.astype(np.float32))                      # (n,n)

    # グラフジャンプ用のショック源サンプリング確率。高次数企業ほどイベント源になりやすくできる。
    deg_event = (true_w > 0).sum(axis=1).astype(float)
    if deg_event.sum() > 0:
        src_prob = (deg_event + 1e-3) ** config.graph_jump_degree_power
        src_prob = src_prob / src_prob.sum()
    else:
        src_prob = np.full(n, 1.0 / n)
    graph_event_residual = torch.zeros(n, device=device)

    # 投資家パラメータ (n_inv,) ベクトル
    def iv(col, dtype=torch.float32):
        return torch.tensor(investor_df[col].to_numpy(), dtype=dtype, device=device)

    vol_sens_t     = iv("vol_sensitivity")
    risk_tol_t     = iv("risk_tolerance")
    trend_wt       = iv("trend_weight")
    value_wt       = iv("value_weight")
    unc_aversion   = iv("uncertainty_aversion")
    rate_sens      = iv("rate_sensitivity")
    temperature_t  = iv("temperature")
    loss_asym      = iv("loss_asymmetry")
    belief_phi_t   = iv("belief_phi")         # (n_inv,)
    belief_rho_s_t = iv("belief_rho_s")       # (n_inv,) for pub/sec dims
    belief_rho_h_t = iv("belief_rho_h")       # (n_inv,) for priv dims

    # float32 stability: ensure phi + rho < 0.97 for each dim to prevent divergence.
    # CPU version uses float64 and avoids float overflow within 1260 steps (max ~10^99);
    # float32 overflows at ~480 steps when phi+rho ≈ 1.20.
    # Fix: clamp each rho independently so phi + rho ≤ 0.97.
    _max_rho_s = (0.97 - belief_phi_t).clamp(min=0.0)
    _max_rho_h = (0.97 - belief_phi_t).clamp(min=0.0)
    belief_rho_s_t = torch.minimum(belief_rho_s_t, _max_rho_s)
    belief_rho_h_t = torch.minimum(belief_rho_h_t, _max_rho_h)

    obs_var_t      = iv("obs_var")
    proc_var_t     = iv("proc_var")
    graph_quality_t = iv("graph_quality")
    update_style_t = iv("update_style")
    wealth_t       = T(wealth_factor)          # (n_inv,)

    # 専門セクターマスク: (n_inv, n) bool tensor
    expertise_t = iv("expertise_sector", dtype=torch.long)
    sectors_t   = torch.from_numpy(sectors).to(device)                 # (n,)
    expert_mask = (expertise_t.unsqueeze(1) == sectors_t.unsqueeze(0)) # (n_inv, n) bool

    # α: 各投資家×企業の更新率 (n_inv, n) — pubはスカラーでも OK、secは専門セクターだけ高い
    alpha_pub_base = config.alpha_pub_base
    alpha_pub_nonexp = config.alpha_pub_nonexpert_scale
    alpha_pub_arr = update_style_t.unsqueeze(1) * alpha_pub_base * torch.where(
        expert_mask,
        torch.ones(n_inv, n, device=device),
        torch.full((n_inv, n), alpha_pub_nonexp, device=device),
    )
    alpha_pub_arr = torch.clamp(alpha_pub_arr, 0.0, config.alpha_max)  # (n_inv, n)

    alpha_sec_arr = update_style_t.unsqueeze(1) * torch.where(
        expert_mask,
        torch.full((n_inv, n), config.alpha_sec_expert, device=device),
        torch.full((n_inv, n), config.alpha_sec_nonexpert, device=device),
    )
    alpha_sec_arr = torch.clamp(alpha_sec_arr, 0.0, config.alpha_max)  # (n_inv, n)

    alpha_priv_arr = (
        config.alpha_priv_base
        * update_style_t.unsqueeze(1)
        * graph_quality_t.unsqueeze(1)
        * torch.where(
            expert_mask,
            torch.ones(n_inv, n, device=device),
            torch.full((n_inv, n), config.alpha_priv_nonexpert_scale, device=device),
        )
    )
    alpha_priv_arr = torch.clamp(alpha_priv_arr, 0.0, config.alpha_max)  # (n_inv, n)

    # private anchor anchor_specs
    anchor_specs = [
        [(0, 0.60), (3, 0.40)],
        [(1, 0.50), (6, 0.50)],
        [(8, 0.50), (9, 0.50)],
        [(4, 0.50), (10, 0.50)],
        [(2, 0.50), (3, 0.50)],
        [(11, 0.34), (12, 0.33), (13, 0.33)],
    ]
    rho_h_dim = T(np.array(config.rho_h_dims, dtype=np.float32))  # (n_priv,)

    # 信念状態と保有
    belief_state = torch.randn(n_inv, n, d, device=device) * 0.010  # (n_inv, n, d)
    x = torch.randn(n, d, device=device) * 0.020
    x_prev = x.clone()
    cash = wealth_t.clone()                                         # (n_inv,)
    holdings = torch.distributions.LogNormal(-5.0, 0.4).sample((n_inv, n)).to(device)
    holdings = holdings * wealth_t.unsqueeze(1)                     # (n_inv, n)

    firm_vol = torch.full((n,), config.idio_vol, device=device)
    market_var        = config.market_vol ** 2
    prev_common_noise = 0.0
    prev_market_vol_t = config.market_vol
    realized_var_ewma = config.market_vol ** 2
    vol_trade_ewma_fast = 0.0
    vol_trade_ewma_slow = 0.0
    vol_trade_ewma_initialized = False
    sp_abs = initial_sp
    market_fundamental_abs = float(initial_sp)

    down_var_ewma = config.exog_common_sigma ** 2
    prev_sp_ret = 0.0

    # belief_rho_s 拡張: pub/sec 次元には rho_s, priv 次元には rho_h
    rho_per_dim = torch.where(
        torch.arange(d, device=device) < (n_pub + n_sec),
        belief_rho_s_t.unsqueeze(1).expand(n_inv, d),
        belief_rho_h_t.unsqueeze(1).expand(n_inv, d),
    )  # (n_inv, d)

    records      = []
    firm_records = []

    for t in range(t_max):
        dgs10_abs    = float(historical_tail.loc[t, "DGS10_abs"])
        dgs10_change = float(historical_tail.loc[t, "DGS10"])
        rf_level     = dgs10_abs / 100.0

        # ---- 真の状態遷移 (CPU numpy で生成、GPU へ転送) ----
        rare_mask_np = (rng.random(n) < config.rare_shock_prob)
        x_np = x.cpu().numpy()
        x_prev_np = x_prev.cpu().numpy()
        true_w_np = true_w

        new_x_np = np.empty_like(x_np)
        for k in range(d):
            noise_k = rng.normal(0.0, float(eta_dims[k].item()), size=n)
            if k == 0:
                noise_k += rare_mask_np * rng.standard_t(df=3, size=n) * config.rare_shock_sigma
            new_x_np[:, k] = float(phi_dims[k].item()) * x_np[:, k] + float(rho_dims[k].item()) * (true_w_np @ x_np[:, k]) + noise_k
        x = T(new_x_np.astype(np.float32))

        # ---- 観測生成 (GPU) ----
        obs_noise_pub = torch.randn(n, n_pub, device=device) * config.obs_sigma_pub
        mom_factors = torch.tensor([0.35 if k == 0 else 0.15 for k in range(n_pub)],
                                   dtype=torch.float32, device=device)
        y_pub = x[:, :n_pub] + mom_factors * (x[:, :n_pub] - x_prev[:, :n_pub]) + obs_noise_pub  # (n, n_pub)

        obs_noise_sec = torch.randn(n, n_sec, device=device) * config.obs_sigma_sec
        y_sec = x[:, n_pub:n_pub+n_sec] + obs_noise_sec  # (n, n_sec)

        y_obs = torch.cat([y_pub, y_sec], dim=1)  # (n, n_pub+n_sec)

        # private anchor: (n, n_priv)
        anchors = torch.zeros(n, n_priv, device=device)
        for kp, entries in enumerate(anchor_specs):
            for dim_idx, weight in entries:
                anchors[:, kp] += weight * y_obs[:, dim_idx]

        momentum = records[-1]["sp500"] if t > 0 else 0.0

        # =====================================================
        # 投資家ループ → GPU バッチ演算
        # =====================================================

        # --- 予測ステップ: pred_state[i,j,k] = phi_i * belief[i,j,k] + rho_i * (W_i @ belief[i,:,k]) ---
        # bmm: (n_inv, n, n) @ (n_inv, n, d) = (n_inv, n, d)
        if config.use_graph:
            graph_term = torch.bmm(subjective_graphs, belief_state)  # (n_inv, n, d)
        else:
            graph_term = torch.zeros_like(belief_state)

        # rho_per_dim: (n_inv, d), expand to (n_inv, 1, d) for broadcast over n_firms
        pred_state = (
            belief_phi_t[:, None, None] * belief_state
            + rho_per_dim[:, None, :] * graph_term
        )  # (n_inv, n, d)

        updated = pred_state.clone()

        # --- 観測更新: public dims ---
        # alpha_pub_arr: (n_inv, n), y_pub: (n, n_pub)
        ap = alpha_pub_arr.unsqueeze(-1)  # (n_inv, n, 1)
        y_pub_b = y_pub.unsqueeze(0)      # (1, n, n_pub)
        updated[:, :, :n_pub] = (1.0 - ap) * pred_state[:, :, :n_pub] + ap * y_pub_b

        # --- 観測更新: sector dims ---
        as_ = alpha_sec_arr.unsqueeze(-1)  # (n_inv, n, 1)
        y_sec_b = y_sec.unsqueeze(0)       # (1, n, n_sec)
        updated[:, :, n_pub:n_pub+n_sec] = (
            (1.0 - as_) * pred_state[:, :, n_pub:n_pub+n_sec] + as_ * y_sec_b
        )

        # --- 観測更新: private dims (主観グラフ経由) ---
        if config.use_graph:
            apriv = alpha_priv_arr.unsqueeze(-1)  # (n_inv, n, 1)
            # pseudo_obs[i, j, kp] = pa_scale * rho_h_i * rho_h_dim[kp] * (W_i @ anchor[:, kp])
            # (W_i @ anchor): bmm (n_inv, n, n) @ (n, n_priv).unsqueeze(0) → (n_inv, n, n_priv)
            anchors_b = anchors.unsqueeze(0).expand(n_inv, -1, -1)  # (n_inv, n, n_priv)
            pseudo_obs = (
                config.private_anchor_scale
                * belief_rho_h_t[:, None, None]
                * rho_h_dim[None, None, :]
                * torch.bmm(subjective_graphs, anchors_b)
            )  # (n_inv, n, n_priv)
            sl = slice(n_pub + n_sec, d)
            updated[:, :, sl] = (1.0 - apriv) * pred_state[:, :, sl] + apriv * pseudo_obs

        belief_state = updated  # (n_inv, n, d)

        # --- total_est と1期先予測 ---
        total_est = (updated * dim_w[None, None, :]).sum(dim=-1)  # (n_inv, n)

        if config.use_graph:
            graph_term_next = torch.bmm(subjective_graphs, updated)
        else:
            graph_term_next = torch.zeros_like(updated)
        pred_next_state = (
            belief_phi_t[:, None, None] * updated
            + rho_per_dim[:, None, :] * graph_term_next
        )
        pred_next = (pred_next_state * dim_w[None, None, :]).sum(dim=-1)  # (n_inv, n)

        # --- スコア ---
        delta = pred_next - total_est  # (n_inv, n)
        uncertainty = torch.sqrt(obs_var_t + proc_var_t).unsqueeze(1)  # (n_inv, 1)
        score = (
            value_wt[:, None] * pred_next
            + trend_wt[:, None] * delta
            - unc_aversion[:, None] * uncertainty
            - rate_sens[:, None] * rf_level
            + config.momentum_score_weight * momentum
            + config.market_risk_premium_score
        )  # (n_inv, n)
        if config.score_centering > 0.0:
            # 投資家ごとに銘柄間の相対魅力度を見る成分を入れる。
            # 全銘柄共通の uncertainty/rate ペナルティが長期の一方的売り圧になるのを防ぐ。
            score = score - config.score_centering * score.mean(dim=1, keepdim=True)

        # --- vol_sensitivity ---
        vol_ratio = prev_market_vol_t / config.market_vol
        vol_factor = torch.clamp(
            1.0 + vol_sens_t * (vol_ratio - 1.0), 0.05, 4.0
        )  # (n_inv,)
        participation_factor = torch.clamp(
            vol_factor ** config.participation_vol_power, 0.10, 5.0
        )  # (n_inv,)

        z_buy  = torch.exp(torch.clamp( temperature_t[:, None] * score, -20, 20)) * participation_factor[:, None]
        z_sell = torch.exp(torch.clamp(-loss_asym[:, None] * temperature_t[:, None] * score, -20, 20)) * participation_factor[:, None]

        # 3a. stop-loss (リスク回避型のみ)
        if config.stoploss_scale > 0.0 and prev_sp_ret < 0.0:
            ra_mask = (vol_sens_t < 0.0)  # (n_inv,)
            if ra_mask.any():
                loss_fear = 1.0 + config.stoploss_scale * abs(prev_sp_ret) / (config.market_vol + 1e-10)
                loss_fear_clamped = min(loss_fear, 5.0)
                z_sell[ra_mask] = z_sell[ra_mask] * loss_fear_clamped

        # 3b. market-wide fear
        if config.stoploss_universal_scale > 0.0 and prev_sp_ret < -config.stoploss_universal_threshold:
            fear_mult = float(np.clip(
                1.0 + config.stoploss_universal_scale * abs(prev_sp_ret) / (config.market_vol + 1e-10),
                1.0, 6.0
            ))
            z_sell = z_sell * fear_mult

        denom  = z_buy + z_sell + 1.0
        p_buy  = z_buy  / denom  # (n_inv, n)
        p_sell = z_sell / denom  # (n_inv, n)

        actions   = torch.rand(n_inv, n, device=device)
        buy_mask  = actions < p_buy
        sell_mask = (actions >= p_buy) & (actions < p_buy + p_sell)

        conviction = torch.minimum(torch.ones_like(score), torch.abs(score) / 0.12)
        size_frac  = risk_tol_t[:, None] * vol_factor[:, None] * (0.25 + conviction)
        size_frac  = size_frac * torch.exp(torch.randn_like(size_frac) * 0.45)
        size_frac  = torch.clamp(size_frac, 0.0002, 0.080)

        buy_orders  = buy_mask.float()  * cash.unsqueeze(1) * size_frac           # (n_inv, n)
        sell_orders = sell_mask.float() * holdings * firm_prices.unsqueeze(0) * size_frac

        # キャッシュ制約
        total_buy = buy_orders.sum(dim=1)  # (n_inv,)
        scale = torch.where(total_buy > cash, cash / (total_buy + 1e-12), torch.ones_like(cash))
        buy_orders = buy_orders * scale.unsqueeze(1)

        buy_value  = buy_orders.sum(dim=0)   # (n,)
        sell_value = sell_orders.sum(dim=0)  # (n,)

        cash     = torch.clamp(cash + sell_orders.sum(dim=1) - buy_orders.sum(dim=1), min=0.0)
        holdings = holdings + buy_orders / firm_prices.unsqueeze(0)
        holdings = holdings - sell_orders / firm_prices.unsqueeze(0)
        holdings = torch.clamp(holdings, min=0.0)

        # 長期turnover/rebalancing。
        # 60年で同一投資家の資金制約・保有集中が固定化して長期ボラレジームを作るのを防ぐ。
        if config.portfolio_rebalance_rate > 0.0:
            rb = float(config.portfolio_rebalance_rate)
            current_weights = market_caps / torch.clamp(market_caps.sum(), min=1e-12)
            hold_value = holdings * firm_prices.unsqueeze(0)
            wealth_now = cash + hold_value.sum(dim=1)
            target_cash = wealth_now * config.portfolio_cash_target
            target_hold_value = (wealth_now - target_cash).unsqueeze(1) * current_weights.unsqueeze(0)
            new_hold_value = (1.0 - rb) * hold_value + rb * target_hold_value
            holdings = torch.clamp(new_hold_value / firm_prices.unsqueeze(0), min=0.0)
            cash = torch.clamp((1.0 - rb) * cash + rb * target_cash, min=0.0)

        # ---- 価格形成 ----
        imbalance   = (buy_value - sell_value) / (buy_value + sell_value + 1e-9)  # (n,)
        total_trade = float((buy_value + sell_value).sum().item())

        # volume_ratio
        if vol_trade_ewma_initialized:
            volume_ratio = vol_trade_ewma_fast / max(vol_trade_ewma_slow, 1e-12)
        else:
            volume_ratio = 1.0

        # 機構1: GJR-GARCH
        if config.gjr_scale > 0.0:
            if config.gjr_centered:
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
            -config.exog_common_clip, config.exog_common_clip
        ))
        market_vol_t = prev_market_vol_t

        firm_vol = (
            config.vol_persistence * firm_vol
            + (1.0 - config.vol_persistence) * config.idio_vol
        )
        noise = torch.from_numpy(
            rng.standard_t(df=5, size=n).astype(np.float32)
        ).to(device) * firm_vol

        # グラフベースイベントジャンプ。
        # 局所ショック源 z を W, W^2, ... で伝播させ、企業ごとの exposure を作る。
        graph_event_jump = torch.zeros(n, device=device)
        if config.graph_jump_prob > 0.0:
            graph_event_residual = config.graph_jump_residual_decay * graph_event_residual
            if event_rng.random() < config.graph_jump_prob:
                ksrc = max(1, min(int(config.graph_jump_sources), n))
                src = event_rng.choice(n, size=ksrc, replace=False, p=src_prob)
                z_np = np.zeros(n, dtype=np.float32)
                z_np[src] = event_rng.lognormal(mean=0.0, sigma=0.35, size=ksrc).astype(np.float32)
                exposure = torch.from_numpy(z_np).to(device)
                prop = exposure.clone()
                for _hop in range(max(0, int(config.graph_jump_hops))):
                    prop = true_w_t @ prop
                    exposure = exposure + (config.graph_jump_spread ** (_hop + 1)) * prop
                if config.graph_jump_normalize == "max1":
                    exposure = exposure / torch.clamp(exposure.max(), min=1e-8)
                else:
                    nz = exposure > 1e-8
                    if bool(nz.any()):
                        exposure = exposure / torch.clamp(exposure[nz].mean(), min=1e-8)
                    exposure = torch.clamp(exposure, 0.0, 6.0)
                mag = abs(event_rng.standard_t(df=config.graph_jump_df)) * config.graph_jump_sigma
                sign = -1.0 if event_rng.random() < config.graph_jump_neg_prob else 1.0
                signed = float(np.clip(sign * mag, -config.graph_jump_clip, config.graph_jump_clip))
                if config.graph_jump_mode == "abs_noise":
                    local_sign = torch.sign(torch.randn(n, device=device) + 0.20 * sign)
                    graph_event_jump = torch.clamp(local_sign * abs(signed) * exposure, -config.graph_jump_clip, config.graph_jump_clip)
                else:
                    graph_event_jump = torch.clamp(signed * exposure, -config.graph_jump_clip, config.graph_jump_clip)
                if config.graph_jump_residual_scale > 0.0:
                    graph_event_residual = graph_event_residual + config.graph_jump_residual_scale * torch.abs(graph_event_jump)
            if config.graph_jump_mode == "residual" and config.graph_jump_residual_scale > 0.0:
                graph_event_jump = graph_event_jump + torch.randn(n, device=device) * graph_event_residual

        # 機構2: 非対称価格インパクト
        if config.asym_pi_scale > 0.0:
            if config.asym_pi_centered:
                baseline_dv = realized_var_ewma / 2.0 + 1e-12
                pi_excess = max(down_var_ewma / baseline_dv - 1.0, 0.0)
                asym_factor = 1.0 + config.asym_pi_scale * pi_excess
            else:
                asym_factor = 1.0 + config.asym_pi_scale * down_var_ewma / (config.market_vol ** 2 + 1e-12)
        else:
            asym_factor = 1.0

        impact_factor_raw = asym_factor * (1.0 + config.impact_activity_scale * max(0.0, volume_ratio - 1.0))

        # 機構4: 非対称クラッシュ
        if config.asym_crash_sell_only:
            sv = sell_value.sum().item()
            bv = buy_value.sum().item()
            if sv / (bv + 1e-9) > 1.1:
                crash_exc = max(0.0, volume_ratio - config.impact_crash_threshold)
                impact_factor_raw += config.impact_crash_scale * (crash_exc ** config.impact_crash_power)
        else:
            crash_exc = max(0.0, volume_ratio - config.impact_crash_threshold)
            impact_factor_raw += config.impact_crash_scale * (crash_exc ** config.impact_crash_power)

        impact_factor = float(np.clip(impact_factor_raw, 0.25, config.impact_activity_clip))

        # 長期安定化: ファンダメンタル価値を更新し、価格との乖離に応じた弱い平均回帰を追加。
        if config.fundamental_strength > 0.0:
            state_value_signal = (x * dim_w[None, :]).sum(dim=1)
            fundamental_growth = (
                config.fundamental_drift
                + config.fundamental_state_sensitivity * torch.tanh(state_value_signal)
                + torch.randn(n, device=device) * config.fundamental_noise
            )
            fundamental_growth = torch.clamp(fundamental_growth, -0.01, 0.01)
            fundamental_prices = fundamental_prices * torch.exp(fundamental_growth)
            fundamental_prices = torch.clamp(fundamental_prices, min=1e-3, max=1e9)
            value_gap = torch.log(fundamental_prices / torch.clamp(firm_prices, min=1e-3))
            fundamental_return = config.fundamental_strength * torch.tanh(value_gap / config.fundamental_gap_scale)
            fundamental_return = torch.clamp(fundamental_return, -config.fundamental_clip, config.fundamental_clip)
        else:
            fundamental_return = torch.zeros(n, device=device)

        if config.market_anchor_strength > 0.0:
            market_fundamental_abs *= float(np.exp(config.market_anchor_drift))
            market_gap = np.log(max(market_fundamental_abs, 1e-12) / max(sp_abs, 1e-12))
            market_anchor_return = config.market_anchor_strength * np.tanh(market_gap / config.market_anchor_gap_scale)
            market_anchor_return = float(np.clip(market_anchor_return, -config.market_anchor_clip, config.market_anchor_clip))
        else:
            market_anchor_return = 0.0

        firm_return = config.price_impact * impact_factor * imbalance + common_noise + graph_event_jump + noise + fundamental_return + market_anchor_return
        firm_return = torch.clamp(firm_return, -config.firm_return_clip, config.firm_return_clip)

        firm_prices = firm_prices * (1.0 + firm_return)
        firm_prices = torch.clamp(firm_prices, min=1e-3, max=1e7)
        market_caps = firm_prices * shares
        weights     = market_caps / market_caps.sum()
        sp_ret_raw  = float((weights * firm_return).sum().item())
        # exog_drift: 価格水準安定化のための外生ドリフト。
        # モデル内部の動態 (down_var_ewma, prev_sp_ret) は生リターンで更新し、
        # 記録・価格水準計算のみ補正済みリターンを使う。
        sp_ret      = sp_ret_raw + config.exog_drift
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
            fw = weights.cpu().numpy()
            fi = imbalance.cpu().numpy()
            fp = firm_prices.cpu().numpy()
            xn = x.cpu().numpy()
            for j in range(n):
                rec = {"day": t, "firm_id": j, "sector": int(sectors[j])}
                for k in range(d):
                    rec[f"x{k}"] = float(xn[j, k])
                rec.update({
                    "price": float(fp[j]),
                    "return": float(firm_return[j].item()),
                    "market_weight": float(fw[j]),
                    "imbalance": float(fi[j]),
                    "market_vol_t": market_vol_t,
                    "down_var_ewma": down_var_ewma,
                })
                firm_records.append(rec)

        prev_common_noise = float(common_noise)
        lam_rv = config.realized_vol_lambda
        # 内部動態 (ボラ推定・恐怖メカニズム) は生リターンで更新
        realized_var_ewma = lam_rv * realized_var_ewma + (1.0 - lam_rv) * (sp_ret_raw ** 2)
        prev_market_vol_t = float(np.sqrt(max(realized_var_ewma, 1e-10)))
        x_prev = x.clone()

        if not vol_trade_ewma_initialized:
            vol_trade_ewma_fast = vol_trade_ewma_slow = total_trade
            vol_trade_ewma_initialized = True
        else:
            lam_f = config.vol_activity_ewma_lambda
            lam_s = max(0.99, lam_f)
            vol_trade_ewma_fast = lam_f * vol_trade_ewma_fast + (1.0 - lam_f) * total_trade
            vol_trade_ewma_slow = lam_s * vol_trade_ewma_slow + (1.0 - lam_s) * total_trade

        neg_ret = max(-sp_ret_raw, 0.0)
        down_var_ewma = (
            config.down_ewma_decay * down_var_ewma
            + (1.0 - config.down_ewma_decay) * neg_ret ** 2
        )
        prev_sp_ret = sp_ret_raw

    firms_df = pd.DataFrame({
        "firm_id": np.arange(n),
        "sector": sectors,
        "initial_market_cap_weight": (market_caps / market_caps.sum()).cpu().numpy(),
        "true_degree": (true_w > 0).sum(axis=1) if config.graph_topology != "zero" else np.zeros(n, dtype=int),
    })

    config_dict = {k: getattr(config, k) for k in Config.__dataclass_fields__}

    return (
        pd.DataFrame(records),
        firms_df,
        investor_df,
        {"config": config_dict, "firm_snapshots": pd.DataFrame(firm_records)},
    )
