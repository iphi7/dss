"""
L_experiment_L_mktcap/run.py

時価総額の偏在化（Pareto 分布 + BA次数相関）の効果を検証。
モデルは K_base2_variant_K_multidim (d=20) を基に shares 生成を変更。

Usage:
    python3 graph_ssm_abm/L_experiment_L_mktcap/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def run_variant(hist: pd.DataFrame, config: Config, label: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    generated, firms, investors, aux = simulate_market(hist, config)
    generated.to_csv(outdir / "generated_paths.csv", index=False)
    firms.to_csv(outdir / "firms.csv", index=False)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)
    print(f"  done: {label}")
    return summarize_stylized_facts(generated, label)


def main() -> None:
    hist = pd.read_csv("output.csv")
    n_days = 1260
    real_tail = hist.tail(n_days).reset_index(drop=True)
    out_base = Path("graph_ssm_abm/L_experiment_L_mktcap/results")

    rows = [summarize_stylized_facts(real_tail, "real_tail")]

    # --- Round 1: Pareto分布・次数相関の感度 ---
    cfg_base = Config(mktcap_pareto_a=999.0, mktcap_degree_power=0.0)
    rows.append(run_variant(hist, cfg_base, "base3_uniform_mktcap", out_base / "base3_uniform"))

    cfg_pd = Config(mktcap_pareto_a=1.2, mktcap_degree_power=0.7)
    rows.append(run_variant(hist, cfg_pd, "pareto_a1.2+deg0.7", out_base / "pareto_deg0.7"))

    # --- Round 2: price_impact 強化 + ノイズ低減 (imbalance チャネルを強調) ---
    # 狙い: price_impact を 2x、ノイズを下げて std を維持しつつ信号強化
    cfg_pi_base = Config(
        mktcap_pareto_a=999.0, mktcap_degree_power=0.0,
        price_impact=0.018, market_vol=0.004, idio_vol=0.005,
    )
    rows.append(run_variant(hist, cfg_pi_base, "strong_pi_uniform", out_base / "strong_pi_uniform"))

    cfg_pi_pd = Config(
        mktcap_pareto_a=1.2, mktcap_degree_power=0.7,
        price_impact=0.018, market_vol=0.004, idio_vol=0.005,
    )
    rows.append(run_variant(hist, cfg_pi_pd, "strong_pi+pareto+deg", out_base / "strong_pi_pareto_deg"))

    # --- Round 3: さらに強化 ---
    cfg_pi_pd2 = Config(
        mktcap_pareto_a=1.0, mktcap_degree_power=0.9,
        price_impact=0.025, market_vol=0.004, idio_vol=0.005,
    )
    rows.append(run_variant(hist, cfg_pi_pd2, "stronger_pi+pareto_a1+deg0.9", out_base / "stronger_pi_pareto"))

    # --- Round 3b: GARCH 安定化 (stress scale を下げて base_var の変動を抑制) ---
    # market_stress が base_var を大きく動かすと、GARCH の実効的持続性が低下する
    cfg_garch_stable = Config(
        mktcap_pareto_a=1.2, mktcap_degree_power=0.7,
        price_impact=0.018, market_vol=0.005, idio_vol=0.004,
        market_garch_alpha=0.06, market_garch_beta=0.93,
        garch_stress_scale=2.0, garch_down_scale=0.5,
    )
    rows.append(run_variant(hist, cfg_garch_stable, "garch_stable+pareto", out_base / "garch_stable_pareto"))

    # --- Round 3c: GARCH 安定化 + price_impact なし (純粋にGARCHの貢献を見る) ---
    cfg_garch_only = Config(
        mktcap_pareto_a=1.2, mktcap_degree_power=0.7,
        price_impact=0.008, market_vol=0.005, idio_vol=0.004,
        market_garch_alpha=0.06, market_garch_beta=0.93,
        garch_stress_scale=2.0, garch_down_scale=0.5,
    )
    rows.append(run_variant(hist, cfg_garch_only, "garch_stable_base_pi", out_base / "garch_stable_base_pi"))

    # --- Round 4: 実現ボラ (EWMA of sp_ret²) を base_var に使用 ---
    cfg_rv = Config(
        mktcap_pareto_a=1.2, mktcap_degree_power=0.7,
        price_impact=0.018, idio_vol=0.005,
        market_garch_alpha=0.07, market_garch_beta=0.92,
        use_realized_vol=True, realized_vol_ewma_lambda=0.94,
    )
    rows.append(run_variant(hist, cfg_rv, "realized_vol+pareto+pi", out_base / "realized_vol_pareto_pi"))

    # --- Round 5: 外部注入実現ボラ (rolling 63日分散, フィードバックなし) ---
    # EWMA 方式は t₅ 分布の E[z²]=5/3 により発散していた。
    # 外部注入方式: 実データ S&P500 rolling 63日分散を base_var として使う。
    cfg_rv5 = Config(
        mktcap_pareto_a=1.2, mktcap_degree_power=0.7,
        price_impact=0.018,
        market_garch_alpha=0.07, market_garch_beta=0.92,
        use_realized_vol=True,
    )
    rows.append(run_variant(hist, cfg_rv5, "rv_rolling63+pareto+pi", out_base / "rv_rolling63_pareto_pi"))

    # --- Round 5b: 同上 + base price_impact ---
    cfg_rv5b = Config(
        mktcap_pareto_a=1.2, mktcap_degree_power=0.7,
        market_garch_alpha=0.07, market_garch_beta=0.92,
        use_realized_vol=True,
    )
    rows.append(run_variant(hist, cfg_rv5b, "rv_rolling63+pareto_base_pi", out_base / "rv_rolling63_pareto_base_pi"))

    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/L_experiment_L_mktcap/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print("\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
