"""
E_variant_E_perfect_graph/run.py

アブレーション: 全投資家の主観グラフを真のグラフ W に完全一致させたとき
スタイライズドファクトがどう変化するかを検証する。

Usage:
    python3 graph_ssm_abm/E_variant_E_perfect_graph/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# スタンドアロン実行のためパスを追加
sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def run_variant(output_df: pd.DataFrame, perfect: bool, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    config = Config(perfect_graph=perfect)
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

    base_dir   = Path("graph_ssm_abm/E_variant_E_perfect_graph/results/base")
    perfect_dir = Path("graph_ssm_abm/E_variant_E_perfect_graph/results/perfect_graph")

    print("=== base (通常の主観グラフ) を実行中 ===")
    gen_base = run_variant(hist, perfect=False, outdir=base_dir)
    print("=== perfect_graph (全員が真のグラフを知っている) を実行中 ===")
    gen_perfect = run_variant(hist, perfect=True, outdir=perfect_dir)

    rows = [
        summarize_stylized_facts(real_tail, "real_tail"),
        summarize_stylized_facts(gen_base,    "base"),
        summarize_stylized_facts(gen_perfect, "perfect_graph"),
    ]
    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/E_variant_E_perfect_graph/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\n=== 比較結果 ===")
    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
