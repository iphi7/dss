from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import summarize_stylized_facts
from .model import Config, simulate_market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Graph-SSM ABM MVP.")
    parser.add_argument("--input", default="output.csv", help="Historical CSV with output.csv format.")
    parser.add_argument("--outdir", default="graph_ssm_abm/results", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=1260)
    parser.add_argument("--firms", type=int, default=80)
    parser.add_argument("--investors", type=int, default=60)
    parser.add_argument("--variant", choices=["base", "A", "B", "C", "C_strong", "C2", "C2_strong", "C2_balanced", "C2_scaled"], default="base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(args.input)
    config = Config(
        seed=args.seed,
        n_days=args.days,
        n_firms=args.firms,
        n_investors=args.investors,
    )
    if args.variant == "A":
        config.dynamic_observation_uncertainty = True
    elif args.variant == "B":
        config.neighbor_sell_pressure = True
    elif args.variant == "C":
        config.exponential_memory = True
    elif args.variant == "C_strong":
        config.exponential_memory = True
        config.firm_memory_decay = 0.97
        config.firm_memory_obs_beta = 1.8
        config.firm_memory_impact_beta = 4.0
        config.firm_memory_vol_beta = 1.4
        config.investor_memory_decay = 0.95
        config.investor_memory_size_beta = 2.3
        config.memory_cap = 6.0
    elif args.variant == "C2":
        config.exponential_memory = True
        config.firm_memory_decay = 0.96
        config.firm_memory_obs_beta = 1.2
        config.firm_memory_impact_beta = 3.0
        config.firm_memory_vol_beta = 1.0
        config.firm_memory_process_beta = 0.45
        config.firm_memory_network_beta = 0.015
        config.investor_memory_decay = 0.94
        config.investor_memory_size_beta = 1.8
        config.memory_cap = 5.0
    elif args.variant == "C2_strong":
        config.exponential_memory = True
        config.firm_memory_decay = 0.975
        config.firm_memory_obs_beta = 1.6
        config.firm_memory_impact_beta = 4.5
        config.firm_memory_vol_beta = 1.6
        config.firm_memory_process_beta = 0.85
        config.firm_memory_network_beta = 0.025
        config.investor_memory_decay = 0.96
        config.investor_memory_size_beta = 2.5
        config.memory_cap = 6.0
    elif args.variant == "C2_balanced":
        config.exponential_memory = True
        config.common_shock_beta = 0.60
        config.market_vol = 0.0045
        config.price_impact = 0.0045
        config.latent_price_beta = 0.0013
        config.firm_memory_decay = 0.97
        config.firm_memory_obs_beta = 1.2
        config.firm_memory_impact_beta = 2.8
        config.firm_memory_vol_beta = 0.9
        config.firm_memory_process_beta = 0.35
        config.firm_memory_network_beta = 0.020
        config.investor_memory_decay = 0.95
        config.investor_memory_size_beta = 1.0
        config.memory_cap = 5.0
    elif args.variant == "C2_scaled":
        config.exponential_memory = True
        config.idio_vol = 0.0040
        config.common_shock_beta = 0.50
        config.market_vol = 0.0035
        config.price_impact = 0.0030
        config.latent_price_beta = 0.0008
        config.firm_memory_decay = 0.975
        config.firm_memory_obs_beta = 1.6
        config.firm_memory_impact_beta = 4.5
        config.firm_memory_vol_beta = 1.6
        config.firm_memory_process_beta = 0.85
        config.firm_memory_network_beta = 0.025
        config.investor_memory_decay = 0.96
        config.investor_memory_size_beta = 2.0
        config.memory_cap = 6.0
    generated, firms, investors, aux = simulate_market(hist, config)

    generated.to_csv(outdir / "generated_paths.csv", index=False)
    generated[["Date", "sp500_abs", "DGS10_abs", "sp500", "DGS10"]].to_csv(
        outdir / "generated_path_output_format.csv", index=False
    )
    firms.to_csv(outdir / "firms.csv", index=False)
    investors.to_csv(outdir / "investors.csv", index=False)
    aux["firm_snapshots"].to_csv(outdir / "firm_snapshots.csv", index=False)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)

    real_tail = hist.tail(args.days).reset_index(drop=True)
    summary = pd.DataFrame(
        [
            summarize_stylized_facts(real_tail, "real_tail"),
            summarize_stylized_facts(generated, "graph_ssm_abm"),
        ]
    )
    summary.to_csv(outdir / "stylized_facts_summary.csv", index=False)

    print(f"saved: {outdir / 'generated_paths.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
