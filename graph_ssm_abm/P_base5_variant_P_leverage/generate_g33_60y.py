"""
G33 パラメータで60年分（全実データ期間）の生成パスを作成するスクリプト。

設定:
  - asym_pi_scale=2.5, asym_pi_centered=True, down_ewma_decay=0.80
  - impact_crash_scale=2.0
  - n_days = len(output.csv) - 2 = 14882 (≈ 60年)
  - initial_sp500_abs = 92.18 (1966-01-03 実データ初期値)
  - DGS10 系列は output.csv 全期間を使用

出力: graph_ssm_abm/P_base5_variant_P_leverage/results_gpu/G33_60y/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config
from model_gpu import simulate_market_gpu

import csv


def load_csv(path: str):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    import pandas as pd
    return pd.DataFrame(rows).astype({
        "sp500_abs": float,
        "DGS10_abs": float,
        "sp500": float,
        "DGS10": float,
    })


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    hist = load_csv("output.csv")
    n_total = len(hist)
    n_days  = n_total - 2   # 長さチェック (n_total < n_days + 2) を回避

    initial_sp   = float(hist.iloc[0]["sp500_abs"])   # 92.18 (1966-01-03)
    initial_dgs10 = float(hist.iloc[0]["DGS10_abs"])  # 4.63

    print(f"total data rows : {n_total}")
    print(f"n_days (sim)    : {n_days}")
    print(f"initial_sp500   : {initial_sp}")
    print(f"initial_dgs10   : {initial_dgs10}")

    config = Config(
        # G33 パラメータ
        asym_pi_scale=2.5,
        asym_pi_centered=True,
        down_ewma_decay=0.80,
        impact_crash_scale=2.0,
        # O19 基本パラメータ
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
        impact_crash_power=2.00,
        # 期間・初期値
        n_days=n_days,
        initial_sp500_abs=initial_sp,
        initial_dgs10_abs=initial_dgs10,
    )

    outdir = Path("graph_ssm_abm/P_base5_variant_P_leverage/results_gpu/G33_60y")
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    generated, firms, investors, aux = simulate_market_gpu(hist, config, device=device)
    elapsed = time.time() - t0
    print(f"simulation done: {elapsed:.1f}s")

    generated.to_csv(outdir / "generated_paths.csv", index=False)
    firms.to_csv(outdir / "firms.csv", index=False)
    investors.to_csv(outdir / "investors.csv", index=False)
    with open(outdir / "config.json", "w") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)

    stats = summarize_stylized_facts(generated, "G33_60y")

    # 目標値との比較表示
    target = {
        "std_sp500": 0.0107,
        "kurt_sp500": 10.10,
        "absacf5_sp500": 0.198,
        "leverage_sp500_lag1_20": -0.043,
    }
    print("\n=== G33 (60年生成) vs 目標 ===")
    print(f"{'指標':30s} {'生成':>10s} {'目標':>10s} {'比率':>8s}")
    print("-" * 62)
    for key, tgt in target.items():
        val = stats.get(key, float("nan"))
        ratio = val / tgt if tgt != 0 else float("nan")
        print(f"{key:30s} {val:>10.4f} {tgt:>10.4f} {ratio:>8.3f}")

    # 実データ60年分との比較
    real_stats = summarize_stylized_facts(hist, "real_60y")
    print("\n=== 実データ60年分 統計量 ===")
    for key in target:
        val = real_stats.get(key, float("nan"))
        print(f"  {key}: {val:.4f}")

    print(f"\n出力先: {outdir}")
    print(f"generated_paths.csv: {len(generated)} 行")


if __name__ == "__main__":
    main()
