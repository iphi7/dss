"""
J_variant_J_scale/run.py

n_firms, n_investors, ba_m をそれぞれ変化させ、スタイライズドファクトへの影響を検証する。
モデルは B_base1_H_variant_H_pub_priv を使用。パラメータを1つずつ変化させ、他はベース値に固定。

    n_firms:     40 / 80(base) / 160
    n_investors: 30 / 60(base) / 120
    ba_m:         2 /  3(base) /   5

Usage:
    python3 graph_ssm_abm/J_variant_J_scale/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE2_DIR = Path(__file__).parent.parent / "B_base1_H_variant_H_pub_priv"
sys.path.insert(0, str(BASE2_DIR))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def run_config(hist: pd.DataFrame, config: Config, label: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    generated, firms, investors, aux = simulate_market(hist, config)
    generated.to_csv(outdir / "generated_paths.csv", index=False)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)
    print(f"  done: {label}")
    return summarize_stylized_facts(generated, label)


def main() -> None:
    hist = pd.read_csv("output.csv")
    n_days = 1260
    real_tail = hist.tail(n_days).reset_index(drop=True)

    out_base = Path("graph_ssm_abm/J_variant_J_scale/results")
    rows = [summarize_stylized_facts(real_tail, "real_tail")]

    # ベース
    print("=== ベース ===")
    rows.append(run_config(hist, Config(seed=42, n_days=n_days), "base(n80_i60_m3)", out_base / "base"))

    # n_firms の変化
    print("=== n_firms ===")
    for nf in [40, 160]:
        cfg = Config(seed=42, n_days=n_days, n_firms=nf)
        rows.append(run_config(hist, cfg, f"n_firms={nf}", out_base / f"n_firms_{nf}"))

    # n_investors の変化
    print("=== n_investors ===")
    for ni in [30, 120]:
        cfg = Config(seed=42, n_days=n_days, n_investors=ni)
        rows.append(run_config(hist, cfg, f"n_inv={ni}", out_base / f"n_investors_{ni}"))

    # ba_m の変化
    print("=== ba_m ===")
    for m in [2, 5]:
        cfg = Config(seed=42, n_days=n_days, ba_m=m)
        rows.append(run_config(hist, cfg, f"ba_m={m}", out_base / f"ba_m_{m}"))

    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/J_variant_J_scale/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print("\n=== 比較結果 ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
