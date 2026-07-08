"""
multi_seed/run.py
=================
アプローチ3: 複数シードによる短期統計の安定性検証

G33 (1260日版) を seed=1〜12 で各1回実行し、統計量の分布を確認する。
オーバーフィッティングの懸念: seed=42 の1回の結果だけで
「G33が目標に最も近い」と結論づけた。複数seedで検証する。

期待:
  - もし統計量のばらつきが小さい → G33は安定して良い統計を生成
  - もしばらつきが大きい → seed=42 がたまたま良かっただけ（過適合）
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import summarize_stylized_facts
from model import Config
from model_gpu import simulate_market_gpu


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = Path(__file__).parent
REPO = Path(__file__).parent.parent.parent.parent

SEEDS = list(range(1, 13))   # 12 seeds
N_DAYS = 1260


def g33_config(seed: int) -> Config:
    return Config(
        asym_pi_scale=2.5,
        asym_pi_centered=True,
        down_ewma_decay=0.80,
        impact_crash_scale=2.0,
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
        n_days=N_DAYS,
        seed=seed,
    )


def main():
    print(f"device: {DEVICE}")
    hist = pd.read_csv(REPO / "output.csv")

    target = {
        "std_sp500": 0.0107,
        "kurt_sp500": 10.10,
        "absacf5_sp500": 0.198,
        "leverage_sp500_lag1_20": -0.043,
    }

    # 実データ末尾1260日の統計
    real_tail = hist.tail(N_DAYS).reset_index(drop=True)
    real_stats = summarize_stylized_facts(real_tail, "real_tail")
    print(f"\n実データ末尾{N_DAYS}日: std={real_stats.get('std_sp500',float('nan')):.4f}"
          f"  kurt={real_stats.get('kurt_sp500',float('nan')):.2f}"
          f"  absacf5={real_stats.get('absacf5_sp500',float('nan')):.3f}"
          f"  leverage={real_stats.get('leverage_sp500_lag1_20',float('nan')):.4f}")

    outdir = BASE / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    header = (f"{'seed':>5s} {'std':>8s} {'kurt':>8s} {'absacf5':>9s} "
              f"{'leverage':>10s} {'time':>6s}")
    print(f"\n{header}")
    print("-" * len(header))

    for seed in SEEDS:
        config = g33_config(seed)
        t0 = time.time()
        generated, _, _, _ = simulate_market_gpu(hist, config, device=DEVICE)
        elapsed = time.time() - t0

        stats = summarize_stylized_facts(generated, f"G33_seed{seed}")
        std_v    = stats.get("std_sp500", float("nan"))
        kurt_v   = stats.get("kurt_sp500", float("nan"))
        acf_v    = stats.get("absacf5_sp500", float("nan"))
        lev_v    = stats.get("leverage_sp500_lag1_20", float("nan"))

        print(f"{seed:>5d} {std_v:>8.4f} {kurt_v:>8.2f} {acf_v:>9.3f} {lev_v:>10.4f} {elapsed:>5.1f}s")

        row = {"seed": seed, "elapsed": elapsed}
        row.update({k: stats.get(k, float("nan")) for k in target})
        results.append(row)

    # 統計サマリ
    print(f"\n=== G33 複数seed統計量分布 ===")
    print(f"  {'指標':30s} {'mean':>8s} {'std_dev':>9s} {'min':>8s} {'max':>8s} {'目標':>8s}")
    print(f"  {'-'*75}")
    for k, tgt in target.items():
        vals = [r[k] for r in results if not math.isnan(r[k])]
        if vals:
            mu = sum(vals)/len(vals)
            sigma = math.sqrt(sum((v-mu)**2 for v in vals)/max(len(vals)-1,1))
            print(f"  {k:30s} {mu:>8.4f} {sigma:>9.4f} {min(vals):>8.4f} {max(vals):>8.4f} {tgt:>8.4f}")
            print(f"  {'(目標との差)':30s} {mu-tgt:>+8.4f} {'±':>4s}{sigma:.4f}")

    # seed=42 (既存の結果) の統計 (比較用)
    print(f"\n[参考] seed=42 (既存結果)")
    config42 = g33_config(42)
    t0 = time.time()
    gen42, _, _, _ = simulate_market_gpu(hist, config42, device=DEVICE)
    elapsed = time.time() - t0
    s42 = summarize_stylized_facts(gen42, "G33_seed42")
    for k, tgt in target.items():
        v = s42.get(k, float("nan"))
        print(f"  {k:30s} {v:>8.4f}  (目標: {tgt:>8.4f})")

    # CSV保存
    fieldnames = ["seed", "elapsed"] + list(target.keys())
    with open(outdir / "seed_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
        # seed=42 も追記
        row42 = {"seed": 42, "elapsed": elapsed}
        row42.update({k: s42.get(k, float("nan")) for k in target})
        w.writerow(row42)

    print(f"\n結果保存先: {outdir}")


if __name__ == "__main__":
    main()
