"""
H_variant_H_pub_priv/run.py

設計変更:
- 潜在状態を公開成分 s（観測可能）と非公開成分 h（観測不可）に分離
    y_j = s_j + noise  （h は直接見えない）
- 投資家は W_i @ y 経由でのみ h を推定
    est_h = belief_rho_h * (W_i @ y)
    pred_next = belief_phi * (y + est_h) + belief_rho_s * (W_i @ (y + est_h))
- 価格形成から真の x の直接介入を除去（純粋に注文不均衡ベース）

比較:
- no_graph: 投資家がグラフを無視（h 推定不能） → pred_next = phi * y
- with_graph: 投資家がグラフ経由で h を推定 → pred_next に W_i の個性が入る

Usage:
    python3 graph_ssm_abm/B_base1_H_variant_H_pub_priv/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def run_variant(output_df: pd.DataFrame, use_graph: bool, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    config = Config(use_graph=use_graph)
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

    no_graph_dir  = Path("graph_ssm_abm/B_base1_H_variant_H_pub_priv/results/no_graph")
    with_graph_dir = Path("graph_ssm_abm/B_base1_H_variant_H_pub_priv/results/with_graph")

    print("=== no_graph (グラフなし・h 推定不能) を実行中 ===")
    gen_no_graph  = run_variant(hist, use_graph=False, outdir=no_graph_dir)
    print("=== with_graph (グラフあり・h を W_i 経由で推定) を実行中 ===")
    gen_with_graph = run_variant(hist, use_graph=True,  outdir=with_graph_dir)

    rows = [
        summarize_stylized_facts(real_tail,      "real_tail"),
        summarize_stylized_facts(gen_no_graph,   "no_graph"),
        summarize_stylized_facts(gen_with_graph, "with_graph"),
    ]
    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/B_base1_H_variant_H_pub_priv/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print(f"\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
