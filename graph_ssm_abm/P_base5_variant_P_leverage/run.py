"""
P_base5_variant_P_leverage/run.py

O19 をベースに leverage effect を再現するための複数機構を検証する。

実験設計:
  P0:  O19 完全再現（ベースライン）
  P1:  GJR mild   (gjr_scale=2.0)
  P2:  GJR strong (gjr_scale=5.0)
  P3:  GJR very strong (gjr_scale=10.0)
  P4:  非対称 PI mild   (asym_pi_scale=1.0)
  P5:  非対称 PI strong (asym_pi_scale=3.0)
  P6:  stop-loss mild   (stoploss_scale=1.0)
  P7:  stop-loss strong (stoploss_scale=3.0)
  P8:  非対称クラッシュ (asym_crash_sell_only=True)
  P9:  GJR mild + 非対称 PI mild
  P10: GJR mild + stop-loss mild
  P11: 非対称 PI + stop-loss
  P12: GJR strong + 非対称クラッシュ
  P13: 全機構 mild combination
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def o19_config(**overrides) -> Config:
    """O19 exact parameters as base."""
    params = dict(
        price_impact=0.050,
        exog_common_sigma=0.0040,
        exog_common_jump_prob=0.006,
        exog_common_jump_sigma=0.035,
        exog_common_clip=0.100,
        realized_vol_lambda=0.985,
        vol_sensitivity_mean=0.80,
        vol_sensitivity_std=0.80,
        wealth_sigma=1.20,
        wealth_vol_corr=1.20,
        participation_vol_power=1.80,
        impact_activity_scale=2.50,
        impact_activity_clip=6.00,
        impact_crash_threshold=1.20,
        impact_crash_scale=4.00,
        impact_crash_power=2.00,
    )
    params.update(overrides)
    return Config(**params)


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
    out_base = Path("graph_ssm_abm/P_base5_variant_P_leverage/results")

    rows = [summarize_stylized_facts(real_tail, "real_tail")]

    # --- P0: O19 ベースライン (leverage 機構なし) ---
    rows.append(run_variant(hist, o19_config(), "P0_O19_baseline", out_base / "P0_O19_baseline"))

    # --- 機構1: GJR-GARCH (c_t sigma を下落後に増幅) ---
    rows.append(run_variant(hist, o19_config(
        gjr_scale=2.0,
    ), "P1_gjr_mild", out_base / "P1_gjr_mild"))

    rows.append(run_variant(hist, o19_config(
        gjr_scale=5.0,
    ), "P2_gjr_strong", out_base / "P2_gjr_strong"))

    rows.append(run_variant(hist, o19_config(
        gjr_scale=10.0,
    ), "P3_gjr_vstrong", out_base / "P3_gjr_vstrong"))

    # --- 機構2: 非対称価格インパクト ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.0,
    ), "P4_asym_pi_mild", out_base / "P4_asym_pi_mild"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0,
    ), "P5_asym_pi_strong", out_base / "P5_asym_pi_strong"))

    # --- 機構3: stop-loss ---
    rows.append(run_variant(hist, o19_config(
        stoploss_scale=1.0,
    ), "P6_stoploss_mild", out_base / "P6_stoploss_mild"))

    rows.append(run_variant(hist, o19_config(
        stoploss_scale=3.0,
    ), "P7_stoploss_strong", out_base / "P7_stoploss_strong"))

    rows.append(run_variant(hist, o19_config(
        stoploss_scale=6.0,
    ), "P8_stoploss_vstrong", out_base / "P8_stoploss_vstrong"))

    # --- 機構4: 非対称クラッシュ (売り超過時のみ) ---
    rows.append(run_variant(hist, o19_config(
        asym_crash_sell_only=True,
    ), "P9_asym_crash", out_base / "P9_asym_crash"))

    # --- 組み合わせ: 上位候補を保存して後で組み合わせ ---
    rows.append(run_variant(hist, o19_config(
        gjr_scale=2.0,
        asym_pi_scale=1.0,
    ), "P10_gjr_asym_pi", out_base / "P10_gjr_asym_pi"))

    rows.append(run_variant(hist, o19_config(
        gjr_scale=2.0,
        stoploss_scale=1.0,
    ), "P11_gjr_stoploss", out_base / "P11_gjr_stoploss"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.0,
        stoploss_scale=1.0,
    ), "P12_asym_pi_stoploss", out_base / "P12_asym_pi_stoploss"))

    rows.append(run_variant(hist, o19_config(
        gjr_scale=2.0,
        asym_pi_scale=1.0,
        stoploss_scale=1.0,
    ), "P13_all_mild", out_base / "P13_all_mild"))

    # ================================================================
    # 第2ラウンド: centered / recalibrated 版
    # 問題: 第1ラウンドで GJR/asym_pi が std を4倍以上に膨張させた。
    # 原因: down_var_ewma が平常時でも market_vol^2 の2-3倍あるため、
    #       非対称係数が「平常時でも」 price_impact を増幅してしまう。
    # 解決: (A) centered=True → realized_var/2 との比率の超過分のみ増幅
    #       (B) recalibrated → base price_impact を下げて補正
    #       (C) market-wide fear stop-loss
    # ================================================================

    # --- (A) centered asym_pi: 平常時に増幅なし ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.0, asym_pi_centered=True,
    ), "P14_asym_pi_c1", out_base / "P14_asym_pi_c1"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
    ), "P15_asym_pi_c2", out_base / "P15_asym_pi_c2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=4.0, asym_pi_centered=True,
    ), "P16_asym_pi_c4", out_base / "P16_asym_pi_c4"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=8.0, asym_pi_centered=True,
    ), "P17_asym_pi_c8", out_base / "P17_asym_pi_c8"))

    # --- (A) centered gjr ---
    rows.append(run_variant(hist, o19_config(
        gjr_scale=2.0, gjr_centered=True,
    ), "P18_gjr_c2", out_base / "P18_gjr_c2"))

    rows.append(run_variant(hist, o19_config(
        gjr_scale=5.0, gjr_centered=True,
    ), "P19_gjr_c5", out_base / "P19_gjr_c5"))

    # --- (B) recalibrated: base price_impact を下げて平均 impact を維持 ---
    # 第1ラウンドで asym_pi=1.0(uncentered) のとき avg factor≈3.4
    # → base pi を 0.050/3.4≈0.015 に下げると平均 impact が O19 と同等
    rows.append(run_variant(hist, o19_config(
        price_impact=0.015,
        asym_pi_scale=1.0, asym_pi_centered=False,
    ), "P20_recalib_pi015", out_base / "P20_recalib_pi015"))

    rows.append(run_variant(hist, o19_config(
        price_impact=0.020,
        asym_pi_scale=1.0, asym_pi_centered=False,
    ), "P21_recalib_pi020", out_base / "P21_recalib_pi020"))

    # --- (C) market-wide fear stop-loss ---
    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=1.0, stoploss_universal_threshold=0.005,
    ), "P22_mktfear_s1", out_base / "P22_mktfear_s1"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=3.0, stoploss_universal_threshold=0.005,
    ), "P23_mktfear_s3", out_base / "P23_mktfear_s3"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=5.0, stoploss_universal_threshold=0.005,
    ), "P24_mktfear_s5", out_base / "P24_mktfear_s5"))

    # --- 組み合わせ: centered asym_pi + centered gjr ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
        gjr_scale=2.0, gjr_centered=True,
    ), "P25_c_asym_gjr", out_base / "P25_c_asym_gjr"))

    # --- 組み合わせ: centered asym_pi + market-wide fear ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
        stoploss_universal_scale=2.0, stoploss_universal_threshold=0.005,
    ), "P26_c_asym_mktfear", out_base / "P26_c_asym_mktfear"))

    # --- recalibrated + market-wide fear ---
    rows.append(run_variant(hist, o19_config(
        price_impact=0.015,
        asym_pi_scale=1.0, asym_pi_centered=False,
        stoploss_universal_scale=2.0, stoploss_universal_threshold=0.005,
    ), "P27_recalib_mktfear", out_base / "P27_recalib_mktfear"))

    # ================================================================
    # 第3ラウンド: centered asym_pi × クラッシュ強度の調整
    #
    # 第2ラウンドの知見:
    #   - centered GJR は c_t が対称なので leverage 効果なし
    #   - centered asym_pi scale=2 → leverage=-0.039 (91%) だが absacf5=0.477 (過剰)
    #   - market-wide fear scale=1 → leverage=-0.057 (超過) だが kurtosis=21.7 (過剰)
    #   - 問題: O19 の crash_scale=4 + asym_pi が重なって ACF が2倍以上になる
    #
    # 方針:
    #   (A) asym_pi_centered + crash_scale を下げて ACF を制御
    #   (B) asym_pi_centered scale を適度に + crash も適度に
    #   (C) mktfear で leverage 出しつつ jump を小さくして kurtosis 制御
    # ================================================================

    # --- (A) crash 削減で ACF を制御 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
        impact_crash_scale=0.0,  # クラッシュ無効
    ), "P28_c2_nocrash", out_base / "P28_c2_nocrash"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
        impact_crash_scale=1.0,  # クラッシュ半減
    ), "P29_c2_crash1", out_base / "P29_c2_crash1"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
        impact_crash_scale=2.0,  # クラッシュ半分
    ), "P30_c2_crash2", out_base / "P30_c2_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
        impact_activity_scale=1.5,  # activity 弱め
        impact_crash_scale=2.0,
    ), "P31_c2_act15_crash2", out_base / "P31_c2_act15_crash2"))

    # --- (B) スケール + クラッシュの適正水準探索 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.5, asym_pi_centered=True,
        impact_crash_scale=2.0,
    ), "P32_c15_crash2", out_base / "P32_c15_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.0, asym_pi_centered=True,
        impact_crash_scale=2.0,
    ), "P33_c1_crash2", out_base / "P33_c1_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.5, asym_pi_centered=True,
        impact_crash_scale=3.0,
    ), "P34_c15_crash3", out_base / "P34_c15_crash3"))

    # --- (C) mktfear + jump 縮小 (kurtosis 制御) ---
    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=1.0, stoploss_universal_threshold=0.005,
        exog_common_jump_sigma=0.022,  # jump 縮小
    ), "P35_mktfear1_smalljump", out_base / "P35_mktfear1_smalljump"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.7, stoploss_universal_threshold=0.005,
    ), "P36_mktfear07", out_base / "P36_mktfear07"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.5, stoploss_universal_threshold=0.005,
    ), "P37_mktfear05", out_base / "P37_mktfear05"))

    # --- (D) 有望組み合わせ: asym_pi_centered 適正 + mktfear 軽度 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.5, asym_pi_centered=True,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.5, stoploss_universal_threshold=0.005,
    ), "P38_c15_crash2_mktfear05", out_base / "P38_c15_crash2_mktfear05"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.3, stoploss_universal_threshold=0.005,
    ), "P39_c2_crash2_mktfear03", out_base / "P39_c2_crash2_mktfear03"))

    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/P_base5_variant_P_leverage/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "sqacf1_sp500",
            "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print("\n=== P leverage comparison ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
