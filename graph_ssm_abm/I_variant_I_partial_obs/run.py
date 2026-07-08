"""
I_variant_I_partial_obs/run.py

設計変更:
- 各投資家は投資ユニバース O_i の企業のみ観測できる
    O_i = 専門セクター企業 ∪ 主観グラフ W_i でエッジが残っている企業
- O_i 外の企業は前期の観測値（陳腐化した情報）を使い続ける
- ハブ企業のショック伝播に「情報伝播の時差」が生まれることを期待する

比較:
- full_obs:    全投資家が全企業を同時観測（base2 と同等）
- partial_obs: 各投資家は O_i のみ観測

Usage:
    python3 graph_ssm_abm/I_variant_I_partial_obs/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def run_variant(output_df: pd.DataFrame, partial_observation: bool, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    config = Config(partial_observation=partial_observation)
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

    full_dir    = Path("graph_ssm_abm/I_variant_I_partial_obs/results/full_obs")
    partial_dir = Path("graph_ssm_abm/I_variant_I_partial_obs/results/partial_obs")

    print("=== full_obs (全企業を全員が同時観測) を実行中 ===")
    gen_full    = run_variant(hist, partial_observation=False, outdir=full_dir)
    print("=== partial_obs (投資ユニバース O_i のみ観測) を実行中 ===")
    gen_partial = run_variant(hist, partial_observation=True,  outdir=partial_dir)

    rows = [
        summarize_stylized_facts(real_tail,   "real_tail"),
        summarize_stylized_facts(gen_full,    "full_obs"),
        summarize_stylized_facts(gen_partial, "partial_obs"),
    ]
    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/I_variant_I_partial_obs/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print(f"\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")

    # 観測ユニバースの統計を表示
    inv = pd.read_csv(partial_dir / "investors.csv")
    print(f"\n=== 観測ユニバース統計 (partial_obs) ===")
    print(f"観測企業数 mean: {inv['n_observed_firms'].mean():.1f} / 80")
    print(f"観測企業数 min: {inv['n_observed_firms'].min()}, max: {inv['n_observed_firms'].max()}")


if __name__ == "__main__":
    main()
