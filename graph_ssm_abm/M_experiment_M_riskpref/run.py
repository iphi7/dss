"""
M_experiment_M_riskpref/run.py

リスク選好によるボラティリティクラスタリングの検証。
・c_t の外生化: garch_stress_scale=0 → market_stress を GARCH から切り離す
・vol_sensitivity: 投資家がボラ水準に応じてポジションサイズを変える
  positive → 高ボラ時に積極売買 (リスク選好)
  negative → 高ボラ時に縮小 (リスク回避)
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
    out_base = Path("graph_ssm_abm/M_experiment_M_riskpref/results")

    rows = [summarize_stylized_facts(real_tail, "real_tail")]

    # --- R0: ベースライン比較 (base3 相当, vol_sensitivity なし) ---
    # garch_stress_scale=6.0 (従来) / vol_sensitivity_mean=0 → ボラクラは market_stress のみ
    cfg_base_stress = Config(
        vol_sensitivity_mean=0.0, vol_sensitivity_std=0.0,
        garch_stress_scale=6.0, garch_down_scale=2.0,
        market_garch_alpha=0.080, market_garch_beta=0.900,
    )
    rows.append(run_variant(hist, cfg_base_stress, "R0_base_stress_only", out_base / "R0_base_stress"))

    # --- R1: c_t 外生化のみ (vol_sensitivity なし) ---
    # garch_stress_scale=0 → c_t が固定分散の外乱になる
    # ボラクラは GARCH β=0.88 の持続性のみ → どれだけ落ちるか確認
    cfg_ct_exog = Config(
        vol_sensitivity_mean=0.0, vol_sensitivity_std=0.0,
        garch_stress_scale=0.0, garch_down_scale=0.0,
    )
    rows.append(run_variant(hist, cfg_ct_exog, "R1_ct_exog_novs", out_base / "R1_ct_exog_novs"))

    # --- R2: vol_sensitivity のみ (c_t は外生化) ---
    # リスク選好チャネルの純効果を見る
    cfg_vs_only = Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
    )
    rows.append(run_variant(hist, cfg_vs_only, "R2_vs_only", out_base / "R2_vs_only"))

    # --- R3: vol_sensitivity 強め (mean=0.8) ---
    cfg_vs_strong = Config(
        vol_sensitivity_mean=0.80, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
    )
    rows.append(run_variant(hist, cfg_vs_strong, "R3_vs_strong", out_base / "R3_vs_strong"))

    # --- R4: vol_sensitivity + price_impact 強化 (L で有効だった) ---
    cfg_vs_pi = Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
        price_impact=0.018,
    )
    rows.append(run_variant(hist, cfg_vs_pi, "R4_vs+pi", out_base / "R4_vs_pi"))

    # --- R5: vol_sensitivity + price_impact + down_scale 残す ---
    cfg_vs_pi_down = Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=1.0,
        price_impact=0.018,
    )
    rows.append(run_variant(hist, cfg_vs_pi_down, "R5_vs+pi+down", out_base / "R5_vs_pi_down"))

    # --- R6: vol_sensitivity 強め + price_impact ---
    cfg_vs_strong_pi = Config(
        vol_sensitivity_mean=0.80, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
        price_impact=0.018,
    )
    rows.append(run_variant(hist, cfg_vs_strong_pi, "R6_vs_strong+pi", out_base / "R6_vs_strong_pi"))

    # --- R7: vol_activity チャネル (取引量→GARCH) のみ, vol_sensitivity なし ---
    # baseline として R1 設定に取引量チャネルを追加
    cfg_va_only = Config(
        vol_sensitivity_mean=0.0, vol_sensitivity_std=0.0,
        garch_stress_scale=0.0, garch_down_scale=0.0,
        vol_activity_scale=2.0,
    )
    rows.append(run_variant(hist, cfg_va_only, "R7_va_only(scale=2)", out_base / "R7_va_only"))

    # --- R8: vol_sensitivity + vol_activity 両方 ---
    cfg_vs_va = Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
        vol_activity_scale=2.0,
    )
    rows.append(run_variant(hist, cfg_vs_va, "R8_vs+va(scale=2)", out_base / "R8_vs_va"))

    # --- R9: vol_sensitivity 強め + vol_activity 強め ---
    cfg_vs_strong_va = Config(
        vol_sensitivity_mean=0.80, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
        vol_activity_scale=4.0,
    )
    rows.append(run_variant(hist, cfg_vs_strong_va, "R9_vs_strong+va(scale=4)", out_base / "R9_vs_strong_va"))

    # --- R10: 最良候補: vol_sensitivity + vol_activity + price_impact ---
    cfg_best = Config(
        vol_sensitivity_mean=0.80, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
        vol_activity_scale=4.0,
        price_impact=0.018,
    )
    rows.append(run_variant(hist, cfg_best, "R10_vs_strong+va+pi", out_base / "R10_best"))

    # --- R11: fast EWMA を遅くして spike の持続期間を伸ばす (λ=0.97, 半減期~23日) ---
    # absacf5 (lag 5 = 1週間) への影響を見る
    cfg_va_slow = Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=0.0,
        vol_activity_scale=2.0, vol_activity_ewma_lambda=0.97,
    )
    rows.append(run_variant(hist, cfg_va_slow, "R11_vs+va_slowfast(lam=0.97)", out_base / "R11_va_slow"))

    # --- R12: R8 + garch_down_scale 追加 (leverage 改善目的) ---
    cfg_r8_down = Config(
        vol_sensitivity_mean=0.40, vol_sensitivity_std=0.60,
        garch_stress_scale=0.0, garch_down_scale=1.5,
        vol_activity_scale=2.0,
    )
    rows.append(run_variant(hist, cfg_r8_down, "R12_R8+down_scale=1.5", out_base / "R12_r8_down"))

    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/M_experiment_M_riskpref/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print("\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
