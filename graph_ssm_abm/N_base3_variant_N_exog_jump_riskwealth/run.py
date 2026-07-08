"""
N_base3_variant_N_exog_jump_riskwealth/run.py

c_t を市場活動依存の GARCH 共通ボラから切り離し、
「小さな通常共通ノイズ + 稀な外生ジャンプ」にする。
ボラクラは、リスク選好型の大口投資家が高実現ボラ時に
参加確率と注文量を上げるミクロ行動から生むことを狙う。
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
    investors.to_csv(outdir / "investors.csv", index=False)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)
    print(f"  done: {label}")
    return summarize_stylized_facts(generated, label)


def main() -> None:
    hist = pd.read_csv("output.csv")
    n_days = 1260
    real_tail = hist.tail(n_days).reset_index(drop=True)
    out_base = Path("graph_ssm_abm/N_base3_variant_N_exog_jump_riskwealth/results")

    rows = [summarize_stylized_facts(real_tail, "real_tail")]

    # N0: 外生ジャンプ c_t のみ。リスク選好反応・資産相関・参加確率反応を切る。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.0, vol_sensitivity_std=0.0,
        wealth_sigma=0.0, wealth_vol_corr=0.0,
        participation_vol_power=0.0,
        price_impact=0.008,
        exog_common_sigma=0.0015, exog_common_jump_prob=0.010, exog_common_jump_sigma=0.030,
    ), "N0_exog_jump_only", out_base / "N0_exog_jump_only"))

    # N1: リスク選好はあるが、資産規模との相関は切る。参加確率と注文量は反応。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        wealth_sigma=0.0, wealth_vol_corr=0.0,
        participation_vol_power=1.0,
        price_impact=0.008,
        exog_common_sigma=0.0015, exog_common_jump_prob=0.010, exog_common_jump_sigma=0.030,
    ), "N1_riskpref_no_wealth", out_base / "N1_riskpref_no_wealth"))

    # N2: リスク選好型が大口になりやすい。N の基本案。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        wealth_sigma=1.00, wealth_vol_corr=0.80,
        participation_vol_power=1.0,
        price_impact=0.008,
        exog_common_sigma=0.0015, exog_common_jump_prob=0.010, exog_common_jump_sigma=0.030,
    ), "N2_riskwealth", out_base / "N2_riskwealth"))

    # N3: 高ボラ時の参加確率反応を強める。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.60, vol_sensitivity_std=0.70,
        wealth_sigma=1.10, wealth_vol_corr=1.00,
        participation_vol_power=1.5,
        price_impact=0.008,
        exog_common_sigma=0.0015, exog_common_jump_prob=0.010, exog_common_jump_sigma=0.030,
    ), "N3_stronger_participation", out_base / "N3_stronger_participation"))

    # N4: price impact を強め、投資家行動が価格に出やすくする。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.60, vol_sensitivity_std=0.70,
        wealth_sigma=1.10, wealth_vol_corr=1.00,
        participation_vol_power=1.5,
        price_impact=0.016,
        exog_common_sigma=0.0015, exog_common_jump_prob=0.010, exog_common_jump_sigma=0.030,
    ), "N4_strong_participation_pi", out_base / "N4_strong_participation_pi"))

    # N5: 外生ジャンプをより稀・大きめにする。c_t はボラクラでなく起点ショックとして扱う。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.60, vol_sensitivity_std=0.70,
        wealth_sigma=1.10, wealth_vol_corr=1.00,
        participation_vol_power=1.5,
        price_impact=0.016,
        exog_common_sigma=0.0010, exog_common_jump_prob=0.006, exog_common_jump_sigma=0.050,
    ), "N5_rare_larger_jump", out_base / "N5_rare_larger_jump"))

    # N6: 実現ボラ EWMA を遅くし、投資家の高ボラ認識を長引かせる。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.60, vol_sensitivity_std=0.70,
        wealth_sigma=1.10, wealth_vol_corr=1.00,
        participation_vol_power=1.5,
        price_impact=0.016,
        exog_common_sigma=0.0010, exog_common_jump_prob=0.006, exog_common_jump_sigma=0.050,
        realized_vol_lambda=0.975,
    ), "N6_slow_realized_vol", out_base / "N6_slow_realized_vol"))

    # N7: リスク選好型大口投資家をさらに強くする。ただし過剰尖度化の確認用。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.85, vol_sensitivity_std=0.80,
        wealth_sigma=1.25, wealth_vol_corr=1.30,
        participation_vol_power=1.8,
        price_impact=0.016,
        exog_common_sigma=0.0010, exog_common_jump_prob=0.006, exog_common_jump_sigma=0.050,
        realized_vol_lambda=0.975,
    ), "N7_aggressive_riskwealth", out_base / "N7_aggressive_riskwealth"))

    # N8: N7 が強すぎる場合に、price impact を少し戻すバランス案。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.85, vol_sensitivity_std=0.80,
        wealth_sigma=1.25, wealth_vol_corr=1.30,
        participation_vol_power=1.8,
        price_impact=0.012,
        exog_common_sigma=0.0010, exog_common_jump_prob=0.006, exog_common_jump_sigma=0.050,
        realized_vol_lambda=0.975,
    ), "N8_aggressive_balanced_pi", out_base / "N8_aggressive_balanced_pi"))


    # N9: ジャンプを控えめにし、通常共通ノイズを増やす。過大kurtosisの緩和。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.70, vol_sensitivity_std=0.80,
        wealth_sigma=1.25, wealth_vol_corr=1.30,
        participation_vol_power=1.8,
        price_impact=0.030,
        exog_common_sigma=0.0040, exog_common_jump_prob=0.003, exog_common_jump_sigma=0.025,
        realized_vol_lambda=0.975,
    ), "N9_smoother_common_pi03", out_base / "N9_smoother_common_pi03"))

    # N10: price impact をさらに強め、分散を投資家行動側から出す。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.70, vol_sensitivity_std=0.80,
        wealth_sigma=1.25, wealth_vol_corr=1.30,
        participation_vol_power=1.8,
        price_impact=0.060,
        exog_common_sigma=0.0040, exog_common_jump_prob=0.003, exog_common_jump_sigma=0.025,
        realized_vol_lambda=0.975,
    ), "N10_smoother_common_pi06", out_base / "N10_smoother_common_pi06"))

    # N11: c_t 通常成分はやや大きく、price impact は中程度。std と kurt のバランス案。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.70, vol_sensitivity_std=0.80,
        wealth_sigma=1.25, wealth_vol_corr=1.30,
        participation_vol_power=1.8,
        price_impact=0.040,
        exog_common_sigma=0.0060, exog_common_jump_prob=0.002, exog_common_jump_sigma=0.020,
        realized_vol_lambda=0.975,
    ), "N11_common06_pi04", out_base / "N11_common06_pi04"))

    # N12: 高ボラ認識をかなり長く残す。absacf5 改善狙い。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.80, vol_sensitivity_std=0.90,
        wealth_sigma=1.30, wealth_vol_corr=1.40,
        participation_vol_power=2.0,
        price_impact=0.050,
        exog_common_sigma=0.0050, exog_common_jump_prob=0.002, exog_common_jump_sigma=0.020,
        realized_vol_lambda=0.990,
    ), "N12_slowvol_pi05", out_base / "N12_slowvol_pi05"))

    # N13: 外生共通ノイズを控えめに保ち、行動側を最大限強める。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=1.00, vol_sensitivity_std=0.90,
        wealth_sigma=1.40, wealth_vol_corr=1.60,
        participation_vol_power=2.2,
        price_impact=0.080,
        exog_common_sigma=0.0030, exog_common_jump_prob=0.002, exog_common_jump_sigma=0.020,
        realized_vol_lambda=0.990,
    ), "N13_behavior_dominant", out_base / "N13_behavior_dominant"))


    # N14: 取引量は c_t ではなく注文インパクトを増幅する。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.80, vol_sensitivity_std=0.90,
        wealth_sigma=1.30, wealth_vol_corr=1.40,
        participation_vol_power=2.0,
        price_impact=0.040,
        impact_activity_scale=2.0,
        impact_activity_clip=4.0,
        exog_common_sigma=0.0050, exog_common_jump_prob=0.002, exog_common_jump_sigma=0.020,
        realized_vol_lambda=0.985,
    ), "N14_activity_impact", out_base / "N14_activity_impact"))

    # N15: 注文インパクト増幅を強める。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.85, vol_sensitivity_std=0.90,
        wealth_sigma=1.35, wealth_vol_corr=1.50,
        participation_vol_power=2.0,
        price_impact=0.040,
        impact_activity_scale=4.0,
        impact_activity_clip=5.0,
        exog_common_sigma=0.0050, exog_common_jump_prob=0.002, exog_common_jump_sigma=0.020,
        realized_vol_lambda=0.985,
    ), "N15_strong_activity_impact", out_base / "N15_strong_activity_impact"))

    # N16: price impact も強める。std/ACF改善狙い。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.85, vol_sensitivity_std=0.90,
        wealth_sigma=1.35, wealth_vol_corr=1.50,
        participation_vol_power=2.0,
        price_impact=0.060,
        impact_activity_scale=4.0,
        impact_activity_clip=5.0,
        exog_common_sigma=0.0040, exog_common_jump_prob=0.002, exog_common_jump_sigma=0.020,
        realized_vol_lambda=0.985,
    ), "N16_strong_activity_pi06", out_base / "N16_strong_activity_pi06"))

    # N17: 過剰尖度を避けるため c_t をほぼ通常ノイズにし、行動側を主にする。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.90, vol_sensitivity_std=0.90,
        wealth_sigma=1.35, wealth_vol_corr=1.60,
        participation_vol_power=2.1,
        price_impact=0.060,
        impact_activity_scale=4.0,
        impact_activity_clip=5.0,
        exog_common_sigma=0.0030, exog_common_jump_prob=0.001, exog_common_jump_sigma=0.015,
        realized_vol_lambda=0.990,
    ), "N17_behavior_main", out_base / "N17_behavior_main"))

    # N18: N17 の分散不足対策として price impact をさらに上げる。
    rows.append(run_variant(hist, Config(
        vol_sensitivity_mean=0.90, vol_sensitivity_std=0.90,
        wealth_sigma=1.35, wealth_vol_corr=1.60,
        participation_vol_power=2.1,
        price_impact=0.080,
        impact_activity_scale=4.0,
        impact_activity_clip=5.0,
        exog_common_sigma=0.0030, exog_common_jump_prob=0.001, exog_common_jump_sigma=0.015,
        realized_vol_lambda=0.990,
    ), "N18_behavior_main_pi08", out_base / "N18_behavior_main_pi08"))

    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/N_base3_variant_N_exog_jump_riskwealth/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "sqacf1_sp500",
            "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print("\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
