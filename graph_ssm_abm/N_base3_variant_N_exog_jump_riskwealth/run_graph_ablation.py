"""
N19 graph ablation.

Compare:
- N19_baseline_partial_graph: BA true graph + subjective missing/noisy investor graphs
- N19_zero_graph: zero true graph + zero investor graphs
- N19_perfect_investor_graph: BA true graph + all investors know the true graph
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def n19_config(**overrides) -> Config:
    params = dict(
        vol_sensitivity_mean=0.80,
        vol_sensitivity_std=0.80,
        wealth_sigma=1.20,
        wealth_vol_corr=1.20,
        participation_vol_power=1.8,
        price_impact=0.080,
        impact_activity_scale=3.0,
        impact_activity_clip=4.0,
        exog_common_sigma=0.0060,
        exog_common_jump_prob=0.001,
        exog_common_jump_sigma=0.015,
        realized_vol_lambda=0.985,
    )
    params.update(overrides)
    return Config(**params)


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
    out_base = Path("graph_ssm_abm/N_base3_variant_N_exog_jump_riskwealth/graph_ablation_results")

    rows = [summarize_stylized_facts(real_tail, "real_tail")]
    rows.append(run_variant(
        hist,
        n19_config(graph_topology="ba", subjective_graph_mode="partial", use_graph=True),
        "N19_baseline_partial_graph",
        out_base / "N19_baseline_partial_graph",
    ))
    rows.append(run_variant(
        hist,
        n19_config(graph_topology="zero", subjective_graph_mode="partial", use_graph=True, mktcap_degree_reference="ba_reference"),
        "N19_zero_graph",
        out_base / "N19_zero_graph",
    ))
    rows.append(run_variant(
        hist,
        n19_config(graph_topology="ba", subjective_graph_mode="perfect", use_graph=True),
        "N19_perfect_investor_graph",
        out_base / "N19_perfect_investor_graph",
    ))

    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/N_base3_variant_N_exog_jump_riskwealth/graph_ablation_summary.csv")
    summary.to_csv(out_csv, index=False)
    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "sqacf1_sp500",
            "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print("\n=== N19 graph ablation ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
