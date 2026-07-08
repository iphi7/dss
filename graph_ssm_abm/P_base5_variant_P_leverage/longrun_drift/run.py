"""
longrun_drift/run.py
====================
アプローチ1: ドリフト補正による60年シミュレーション

問題: asym_pi_centered が持続的な負のドリフト（約-0.001/日）を生成し、
     60年では価格が0に崩壊する。

対処: exog_drift パラメータで日次リターンに正のドリフトを加算し、
     実際のSP500水準（92→7501、年率+7%）と整合させる。

     必要補正 = (実際の年率成長) + (モデル固有負ドリフト打消し)
     実データ  : ln(7501/92) / 14882日 ≈ +0.000296/日
     モデル推定: 約 -0.00103/日
     合計補正  : 約 +0.00133/日

試験値: drift = 0.0 (補正なし), +0.00066, +0.00133, +0.00200
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import summarize_stylized_facts
from model import Config
from model_gpu import simulate_market_gpu

import pandas as pd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = Path(__file__).parent
REPO = Path(__file__).parent.parent.parent.parent  # /home/u00121


def g33_config(**overrides) -> Config:
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
        **overrides,
    )


def load_hist():
    hist = pd.read_csv(REPO / "output.csv")
    return hist


def run_single(hist: pd.DataFrame, drift: float, label: str) -> dict:
    n_total = len(hist)
    n_days  = n_total - 2
    initial_sp    = float(hist.iloc[0]["sp500_abs"])
    initial_dgs10 = float(hist.iloc[0]["DGS10_abs"])

    config = g33_config(
        n_days=n_days,
        initial_sp500_abs=initial_sp,
        initial_dgs10_abs=initial_dgs10,
        exog_drift=drift,
    )

    outdir = BASE / "results" / label
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    generated, firms, investors, aux = simulate_market_gpu(hist, config, device=DEVICE)
    elapsed = time.time() - t0

    generated.to_csv(outdir / "generated_paths.csv", index=False)
    with open(outdir / "config.json", "w") as f:
        json.dump(aux["config"], f, indent=2)

    sp_abs = generated["sp500_abs"].tolist()
    sp_rets = generated["sp500"].tolist()

    # 価格軌跡サマリ
    span = len(sp_abs)
    checkpoints = [0, span//4, span//2, 3*span//4, span-1]
    dates = generated["Date"].tolist()
    traj = [(dates[i], sp_abs[i]) for i in checkpoints]

    # 全期間スタイライズドファクト
    stats_full = summarize_stylized_facts(generated, label)

    # 先頭・末尾1260日のスタイライズドファクト
    stats_head = summarize_stylized_facts(generated.head(1260).reset_index(drop=True), f"{label}_head1260")
    stats_tail = summarize_stylized_facts(generated.tail(1260).reset_index(drop=True), f"{label}_tail1260")

    print(f"\n{'='*60}")
    print(f"drift={drift:+.5f}  ({elapsed:.1f}s)")
    print(f"  価格軌跡: SP500  {traj[0][0]}={traj[0][1]:.1f} → "
          f"{traj[2][0]}={traj[2][1]:.1f} → {traj[4][0]}={traj[4][1]:.1f}")
    print(f"  最小SP500: {min(sp_abs):.3f}  最大: {max(sp_abs):.1f}")

    target = {"std_sp500": 0.0107, "kurt_sp500": 10.10,
              "absacf5_sp500": 0.198, "leverage_sp500_lag1_20": -0.043}
    print(f"  {'指標':30s} {'全期間':>8s} {'先頭1260':>9s} {'末尾1260':>9s} {'目標':>8s}")
    print(f"  {'-'*70}")
    for k, tgt in target.items():
        v_full = stats_full.get(k, float("nan"))
        v_head = stats_head.get(k, float("nan"))
        v_tail = stats_tail.get(k, float("nan"))
        print(f"  {k:30s} {v_full:>8.4f} {v_head:>9.4f} {v_tail:>9.4f} {tgt:>8.4f}")

    return {
        "label": label, "drift": drift,
        "sp_start": sp_abs[0], "sp_mid": sp_abs[span//2], "sp_end": sp_abs[-1],
        **{f"full_{k}": stats_full.get(k, float("nan")) for k in target},
        **{f"head_{k}": stats_head.get(k, float("nan")) for k in target},
        **{f"tail_{k}": stats_tail.get(k, float("nan")) for k in target},
    }


def main():
    print(f"device: {DEVICE}")
    hist = load_hist()

    # 実データの年率成長
    sp_start = float(hist.iloc[0]["sp500_abs"])
    sp_end   = float(hist.iloc[-1]["sp500_abs"])
    n_days   = len(hist) - 2
    target_daily_drift = math.log(sp_end / sp_start) / n_days
    print(f"実データ SP500: {sp_start:.2f} → {sp_end:.2f}  ({n_days}日)")
    print(f"  目標日次ドリフト: {target_daily_drift:+.6f}/日")

    drift_values = [
        (0.0,       "drift_0000"),
        (0.00066,   "drift_0066"),
        (target_daily_drift + 0.00103, "drift_corrected"),   # モデル負ドリフト打消し+実株式premium
        (0.00200,   "drift_0200"),
    ]

    results = []
    for drift, label in drift_values:
        row = run_single(hist, drift, label)
        results.append(row)

    # サマリ CSV
    keys = list(results[0].keys())
    with open(BASE / "results" / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    print(f"\n結果保存先: {BASE/'results'}")


if __name__ == "__main__":
    main()
