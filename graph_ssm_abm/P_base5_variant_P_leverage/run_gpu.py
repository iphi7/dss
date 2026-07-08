"""
P_base5_variant_P_leverage/run_gpu.py

GPU 加速版実行スクリプト。model_gpu.py を使用。
CPU 版 run.py と同じ Config インタフェース・同じ出力形式。

速度: CPU ~30s/実験 → GPU ~5s/実験 (RTX 5000 で約 6× 高速)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config
from model_gpu import simulate_market_gpu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")


def o19_config(**overrides) -> Config:
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
    generated, firms, investors, aux = simulate_market_gpu(hist, config, device=DEVICE)
    generated.to_csv(outdir / "generated_paths.csv", index=False)
    firms.to_csv(outdir / "firms.csv", index=False)
    investors.to_csv(outdir / "investors.csv", index=False)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)
    print(f"  done: {label}")
    return summarize_stylized_facts(generated, label)


def main() -> None:
    import time
    hist = pd.read_csv("output.csv")
    n_days = 1260
    real_tail = hist.tail(n_days).reset_index(drop=True)
    out_base = Path("graph_ssm_abm/P_base5_variant_P_leverage/results_gpu")

    rows = [summarize_stylized_facts(real_tail, "real_tail")]
    t_start = time.time()

    # ================================================================
    # 第4ラウンド: faster down_ewma_decay × mktfear tuning
    #
    # 知見:
    #   - asym_pi_centered: leverage ≈ -0.039 だが absacf5=0.477 (過剰)
    #     → EWMA をより速く減衰させると asym_pi 効果が短命化 → less ACF ?
    #   - mktfear=0.5: leverage=-0.041 (95%) だが kurtosis=17.2 (過剰)
    #     → クラッシュ削減で kurtosis を制御できるか?
    #   - 組み合わせ (P38): kurtosis=10.43 (目標!) だが leverage=-0.141 (過大)
    # ================================================================

    # --- O19 GPU ベースライン (再現確認) ---
    rows.append(run_variant(hist, o19_config(),
        "G0_O19_base", out_base / "G0_O19_base"))

    # --- (A) faster EWMA decay で asym_pi の ACF 膨張を抑制 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.80,
    ), "G1_c2_decay08", out_base / "G1_c2_decay08"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
    ), "G2_c3_decay08", out_base / "G2_c3_decay08"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.70,
    ), "G3_c2_decay07", out_base / "G3_c2_decay07"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=4.0, asym_pi_centered=True, down_ewma_decay=0.70,
    ), "G4_c4_decay07", out_base / "G4_c4_decay07"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
    ), "G5_c2_decay08_crash2", out_base / "G5_c2_decay08_crash2"))

    # --- (B) mktfear + crash 削減 (kurtosis 制御) ---
    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.5, stoploss_universal_threshold=0.005,
        impact_crash_scale=2.0,
    ), "G6_mf05_crash2", out_base / "G6_mf05_crash2"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.5, stoploss_universal_threshold=0.005,
        impact_crash_scale=1.0,
    ), "G7_mf05_crash1", out_base / "G7_mf05_crash1"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.5, stoploss_universal_threshold=0.005,
        impact_crash_scale=0.0,
    ), "G8_mf05_nocrash", out_base / "G8_mf05_nocrash"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.3, stoploss_universal_threshold=0.005,
        impact_crash_scale=2.0,
    ), "G9_mf03_crash2", out_base / "G9_mf03_crash2"))

    # --- (C) 組み合わせ: fast-decay asym_pi + mild mktfear ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.5, asym_pi_centered=True, down_ewma_decay=0.80,
        stoploss_universal_scale=0.3, stoploss_universal_threshold=0.005,
        impact_crash_scale=2.0,
    ), "G10_c15d08_mf03_crash2", out_base / "G10_c15d08_mf03_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.80,
        stoploss_universal_scale=0.3, stoploss_universal_threshold=0.005,
        impact_crash_scale=2.0,
    ), "G11_c2d08_mf03_crash2", out_base / "G11_c2d08_mf03_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=1.5, asym_pi_centered=True, down_ewma_decay=0.80,
        stoploss_universal_scale=0.5, stoploss_universal_threshold=0.005,
        impact_crash_scale=1.5,
    ), "G12_c15d08_mf05_crash15", out_base / "G12_c15d08_mf05_crash15"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.70,
        stoploss_universal_scale=0.2, stoploss_universal_threshold=0.005,
        impact_crash_scale=2.0,
    ), "G13_c2d07_mf02_crash2", out_base / "G13_c2d07_mf02_crash2"))

    # ================================================================
    # 第5ラウンド: G5 (asym_pi,kurtosis=10) から leverageを増やす
    #
    # 知見(第4ラウンド):
    #   - G5 (asym_pi=2, d=0.80, crash2): leverage=-0.031, kurtosis=10.13 ← kurtosis最良
    #   - G8 (mktfear=0.5, nocrash): leverage=-0.067, kurtosis=11.83
    #   - G9 (mktfear=0.3, crash2):  leverage=-0.048, kurtosis=15.84
    #   - G0 baseline: leverage=+0.008, kurtosis=10.91
    #
    # 方針:
    #   (A) G5ベース + 軽微なmktfear → kurtosisを維持しつつleverageを増加
    #   (B) mktfear のcrash削減 → G9のkurtosis改善
    #   (C) asym_pi スケール増大 + crash調整
    # ================================================================

    # --- (A) G5 + 軽微な mktfear ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.05, stoploss_universal_threshold=0.005,
    ), "G14_c2d08c2_mf005", out_base / "G14_c2d08c2_mf005"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.10, stoploss_universal_threshold=0.005,
    ), "G15_c2d08c2_mf01", out_base / "G15_c2d08c2_mf01"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.15, stoploss_universal_threshold=0.005,
    ), "G16_c2d08c2_mf015", out_base / "G16_c2d08c2_mf015"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.05, stoploss_universal_threshold=0.005,
    ), "G17_c25d08c2_mf005", out_base / "G17_c25d08c2_mf005"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.10, stoploss_universal_threshold=0.005,
    ), "G18_c25d08c2_mf01", out_base / "G18_c25d08c2_mf01"))

    # --- (B) mktfear の crash削減 → G9のkurtosis改善 ---
    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.3, stoploss_universal_threshold=0.005,
        impact_crash_scale=0.0,
    ), "G19_mf03_nocrash", out_base / "G19_mf03_nocrash"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.3, stoploss_universal_threshold=0.005,
        impact_crash_scale=1.0,
    ), "G20_mf03_crash1", out_base / "G20_mf03_crash1"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.2, stoploss_universal_threshold=0.005,
        impact_crash_scale=0.0,
    ), "G21_mf02_nocrash", out_base / "G21_mf02_nocrash"))

    rows.append(run_variant(hist, o19_config(
        stoploss_universal_scale=0.15, stoploss_universal_threshold=0.005,
        impact_crash_scale=0.0,
    ), "G22_mf015_nocrash", out_base / "G22_mf015_nocrash"))

    # --- (C) asym_pi 強化 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
    ), "G23_c3d08_crash2", out_base / "G23_c3d08_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=4.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
    ), "G24_c4d08_crash2", out_base / "G24_c4d08_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=4.0,
    ), "G25_c2d08_crash4", out_base / "G25_c2d08_crash4"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=4.0,
    ), "G26_c3d08_crash4", out_base / "G26_c3d08_crash4"))

    # ================================================================
    # 第6ラウンド: std↓ & absacf5↓ を狙う
    #
    # 知見(第5ラウンド):
    #   - G23 (asym_pi=3, d=0.80, crash2): leverage=-0.040, kurtosis=9.56 ← best overall
    #   - G5  (asym_pi=2, d=0.80, crash2): leverage=-0.051, kurtosis=10.40 ← good
    #   - 課題: std=0.014 (目標0.011), absacf5=0.29 (目標0.198)
    #
    # 方針:
    #   (A) price_impact削減 → std↓
    #   (B) faster decay → absacf5↓
    #   (C) 組み合わせ探索
    # ================================================================

    # --- (A) price_impact削減 (G23ベース) ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
    ), "G27_c3d08c2_pi04", out_base / "G27_c3d08c2_pi04"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.035,
    ), "G28_c3d08c2_pi035", out_base / "G28_c3d08c2_pi035"))

    # --- (B) faster decay → ACF↓ ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.70,
        impact_crash_scale=2.0,
    ), "G29_c3d07_crash2", out_base / "G29_c3d07_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.0, asym_pi_centered=True, down_ewma_decay=0.70,
        impact_crash_scale=2.0,
    ), "G30_c2d07_crash2", out_base / "G30_c2d07_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.85,
        impact_crash_scale=2.0,
    ), "G31_c3d085_crash2", out_base / "G31_c3d085_crash2"))

    # --- (C) asym_pi スケール微調整 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
    ), "G32_c35d08_crash2", out_base / "G32_c35d08_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
    ), "G33_c25d08_crash2", out_base / "G33_c25d08_crash2"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
        stoploss_universal_scale=0.05, stoploss_universal_threshold=0.005,
    ), "G34_c3d08c2pi04_mf005", out_base / "G34_c3d08c2pi04_mf005"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.75,
        impact_crash_scale=2.0,
    ), "G35_c3d075_crash2", out_base / "G35_c3d075_crash2"))

    # ================================================================
    # 第7ラウンド: G28 (std完璧,leverage良) と G23 (kurtosis完璧) の橋渡し
    #
    # G23 vs G28: 違いは price_impact のみ (0.050 vs 0.035)
    #   G23: pi=0.050 → kurtosis=10.10 (完璧), leverage=-0.029, std=0.013
    #   G28: pi=0.035 → kurtosis=8.62 (低), leverage=-0.045 (良), std=0.011
    #
    # 目標: std=0.011, leverage=-0.043, kurtosis=10.10, absacf5≈0.198
    # ================================================================

    # --- (A) price_impact 中間値スイープ ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.038,
    ), "G36_c3d08c2_pi038", out_base / "G36_c3d08c2_pi038"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.042,
    ), "G37_c3d08c2_pi042", out_base / "G37_c3d08c2_pi042"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.045,
    ), "G38_c3d08c2_pi045", out_base / "G38_c3d08c2_pi045"))

    # --- (B) G28 ベース + crash調整 (kurtosis↑) ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=1.0, price_impact=0.035,
    ), "G39_c3d08c1_pi035", out_base / "G39_c3d08c1_pi035"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=1.5, price_impact=0.035,
    ), "G40_c3d08c15_pi035", out_base / "G40_c3d08c15_pi035"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=0.0, price_impact=0.035,
    ), "G41_c3d08c0_pi035", out_base / "G41_c3d08c0_pi035"))

    # --- (C) G23 ベース + leverage強化 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.02, stoploss_universal_threshold=0.005,
    ), "G42_c3d08c2_mf002", out_base / "G42_c3d08c2_mf002"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G43_c3d08c2_mf004", out_base / "G43_c3d08c2_mf004"))

    # --- (D) 組み合わせ最適化候補 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
    ), "G44_c35d08c2_pi040", out_base / "G44_c35d08c2_pi040"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=4.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.035,
    ), "G45_c4d08c2_pi035", out_base / "G45_c4d08c2_pi035"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.038,
        stoploss_universal_scale=0.03, stoploss_universal_threshold=0.005,
    ), "G46_c3d08c2pi038_mf003", out_base / "G46_c3d08c2pi038_mf003"))

    # ================================================================
    # 第8ラウンド: G42(pi=0.050, mf=0.02 → leverage=-0.046, kurt=11.27) を
    #             pi削減でstd↓&kurtosis調整
    # ================================================================

    # --- (A) G42 base → pi削減 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
        stoploss_universal_scale=0.02, stoploss_universal_threshold=0.005,
    ), "G47_c3c2pi040_mf002", out_base / "G47_c3c2pi040_mf002"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.043,
        stoploss_universal_scale=0.02, stoploss_universal_threshold=0.005,
    ), "G48_c3c2pi043_mf002", out_base / "G48_c3c2pi043_mf002"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
        stoploss_universal_scale=0.01, stoploss_universal_threshold=0.005,
    ), "G49_c3c2pi040_mf001", out_base / "G49_c3c2pi040_mf001"))

    # --- (B) G33 (kurtosis=10.22) + 小mktfear ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.02, stoploss_universal_threshold=0.005,
    ), "G50_c25c2pi05_mf002", out_base / "G50_c25c2pi05_mf002"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G51_c25c2pi05_mf004", out_base / "G51_c25c2pi05_mf004"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        stoploss_universal_scale=0.06, stoploss_universal_threshold=0.005,
    ), "G52_c25c2pi05_mf006", out_base / "G52_c25c2pi05_mf006"))

    # --- (C) asym_pi強化 + pi削減 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=4.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
    ), "G53_c4c2pi040", out_base / "G53_c4c2pi040"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
    ), "G54_c35c2pi040", out_base / "G54_c35c2pi040"))

    # --- (D) crash1.5 + pi調整 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=1.5, price_impact=0.038,
    ), "G55_c3c15pi038", out_base / "G55_c3c15pi038"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=1.5, price_impact=0.040,
    ), "G56_c3c15pi040", out_base / "G56_c3c15pi040"))

    # ================================================================
    # 第9ラウンド: G51 (leverage=-0.043 完璧!, kurtosis=10.16 完璧!) の
    #             std削減 (0.014→0.011) を price_impact削減で試みる
    #
    # G51: asym_pi=2.5, pi=0.050, crash2, mf=0.04
    #   → leverage=-0.043, kurtosis=10.16, std=0.014, absacf5=0.277
    # ================================================================

    # --- (A) G51 の price_impact スイープ ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.048,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G57_c25c2pi048_mf004", out_base / "G57_c25c2pi048_mf004"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.045,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G58_c25c2pi045_mf004", out_base / "G58_c25c2pi045_mf004"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.043,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G59_c25c2pi043_mf004", out_base / "G59_c25c2pi043_mf004"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G60_c25c2pi040_mf004", out_base / "G60_c25c2pi040_mf004"))

    # --- (B) G47 (leverage=-0.047, kurtosis=10.30) の微調整 ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.040,
        stoploss_universal_scale=0.015, stoploss_universal_threshold=0.005,
    ), "G61_c3c2pi040_mf0015", out_base / "G61_c3c2pi040_mf0015"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=3.0, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=2.0, price_impact=0.038,
        stoploss_universal_scale=0.015, stoploss_universal_threshold=0.005,
    ), "G62_c3c2pi038_mf0015", out_base / "G62_c3c2pi038_mf0015"))

    # --- (C) 新アプローチ: G51スタイル + crash削減でstd↓ ---
    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=1.0, price_impact=0.050,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G63_c25c1pi050_mf004", out_base / "G63_c25c1pi050_mf004"))

    rows.append(run_variant(hist, o19_config(
        asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80,
        impact_crash_scale=0.0, price_impact=0.050,
        stoploss_universal_scale=0.04, stoploss_universal_threshold=0.005,
    ), "G64_c25c0pi050_mf004", out_base / "G64_c25c0pi050_mf004"))

    elapsed = time.time() - t_start
    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/P_base5_variant_P_leverage/comparison_gpu.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "sqacf1_sp500",
            "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print(f"\n=== P leverage GPU round4 ({elapsed:.0f}s total) ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
