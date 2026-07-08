"""
F_variant_F_no_kalman/run.py

アブレーション: カルマンフィルタの有無がスタイライズドファクトに与える影響を検証する。

- base: 通常のカルマン更新（use_kalman=True）
- no_kalman: 観測値をそのまま推定値として使用（use_kalman=False）

Usage:
    python3 graph_ssm_abm/F_variant_F_no_kalman/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def run_variant(output_df: pd.DataFrame, use_kalman: bool, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    config = Config(use_kalman=use_kalman)
    generated, firms, investors, aux = simulate_market(output_df, config)
    generated.to_csv(outdir / "generated_paths.csv", index=False)
    investors.to_csv(outdir / "investors.csv", index=False)
    firms.to_csv(outdir / "firms.csv", index=False)
    aux["firm_snapshots"].to_csv(outdir / "firm_snapshots.csv", index=False)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)
    return generated


def main() -> None:
    hist = pd.read_csv("output.csv")
    n_days = 1260

    real_tail = hist.tail(n_days).reset_index(drop=True)

    base_dir     = Path("graph_ssm_abm/F_variant_F_no_kalman/results/base")
    nokal_dir    = Path("graph_ssm_abm/F_variant_F_no_kalman/results/no_kalman")

    print("=== base (カルマンあり) を実行中 ===")
    gen_base = run_variant(hist, use_kalman=True, outdir=base_dir)
    print("=== no_kalman (観測値をそのまま使用) を実行中 ===")
    gen_nokal = run_variant(hist, use_kalman=False, outdir=nokal_dir)

    rows = [
        summarize_stylized_facts(real_tail,  "real_tail"),
        summarize_stylized_facts(gen_base,   "base (kalman)"),
        summarize_stylized_facts(gen_nokal,  "no_kalman"),
    ]
    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/F_variant_F_no_kalman/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print(f"\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
