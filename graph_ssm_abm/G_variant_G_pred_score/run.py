"""
G_variant_G_pred_score/run.py

変更点:
- カルマンフィルタを廃止（観測 y をそのまま現在推定に使う）
- 取引スコアを来期予測ベースに変更
    pred_next = belief_phi * y + belief_rho * (W_i @ y)
    score = f(pred_next)
- W_i の個性がスコアに直接反映される

比較:
- base_no_graph: カルマンなし・現在観測値ベーススコア（グラフ不使用）
- pred_score:    カルマンなし・来期予測ベーススコア（グラフ使用）

Usage:
    python3 graph_ssm_abm/G_variant_G_pred_score/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def run_variant(output_df: pd.DataFrame, use_pred_score: bool, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    config = Config(use_pred_score=use_pred_score)
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

    base_dir = Path("graph_ssm_abm/G_variant_G_pred_score/results/base_no_graph")
    pred_dir = Path("graph_ssm_abm/G_variant_G_pred_score/results/pred_score")

    print("=== base (現在観測値スコア・グラフ不使用) を実行中 ===")
    gen_base = run_variant(hist, use_pred_score=False, outdir=base_dir)
    print("=== pred_score (来期予測スコア・グラフ使用) を実行中 ===")
    gen_pred = run_variant(hist, use_pred_score=True, outdir=pred_dir)

    rows = [
        summarize_stylized_facts(real_tail, "real_tail"),
        summarize_stylized_facts(gen_base,  "base_no_graph"),
        summarize_stylized_facts(gen_pred,  "pred_score"),
    ]
    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/G_variant_G_pred_score/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print(f"\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
