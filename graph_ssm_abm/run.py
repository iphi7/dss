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
    parser.add_argument("--variant", choices=["base", "A", "B", "C", "C_strong", "C2", "C2_strong", "C2_balanced", "C2_scaled", "D_no_market", "D_factors", "D_factors_strong", "D_factors_memory", "D_factors_tuned", "D_factors_tuned2", "D_factors_final"], default="base")
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
    elif args.variant == "D_no_market":
        config.common_shock_beta = 0.0
    elif args.variant == "D_factors":
        config.common_shock_beta = 0.0
        config.firm_common_factors = True
        config.factor_phi = 0.94
        config.factor_sigma = 0.025
        config.factor_rare_sigma = 0.080
        config.factor_market_loading = 0.28
        config.factor_sector_loading = 0.32
        config.factor_random_loading = 0.05
        config.latent_price_beta = 0.0030
    elif args.variant == "D_factors_strong":
        config.common_shock_beta = 0.0
        config.firm_common_factors = True
        config.factor_phi = 0.96
        config.factor_sigma = 0.040
        config.factor_rare_prob = 0.020
        config.factor_rare_sigma = 0.120
        config.factor_market_loading = 0.40
        config.factor_sector_loading = 0.45
        config.factor_random_loading = 0.08
        config.latent_price_beta = 0.0040
        config.price_impact = 0.0070
    elif args.variant == "D_factors_memory":
        config.common_shock_beta = 0.0
        config.firm_common_factors = True
        config.exponential_memory = True
        config.factor_phi = 0.95
        config.factor_sigma = 0.030
        config.factor_rare_prob = 0.016
        config.factor_rare_sigma = 0.100
        config.factor_market_loading = 0.34
        config.factor_sector_loading = 0.40
        config.factor_random_loading = 0.06
        config.latent_price_beta = 0.0032
        config.firm_memory_decay = 0.965
        config.firm_memory_obs_beta = 1.0
        config.firm_memory_impact_beta = 2.5
        config.firm_memory_vol_beta = 0.9
        config.firm_memory_process_beta = 0.30
        config.firm_memory_network_beta = 0.015
        config.investor_memory_decay = 0.94
        config.investor_memory_size_beta = 1.2
    elif args.variant == "D_factors_tuned":
        config.common_shock_beta = 0.0
        config.firm_common_factors = True
        config.exponential_memory = True
        config.factor_phi = 0.965
        config.factor_sigma = 0.024
        config.factor_rare_prob = 0.018
        config.factor_rare_sigma = 0.090
        config.factor_market_loading = 0.34
        config.factor_sector_loading = 0.38
        config.factor_random_loading = 0.05
        config.process_sigma = 0.016
        config.rare_shock_sigma = 0.090
        config.idio_vol = 0.0050
        config.price_impact = 0.0060
        config.latent_price_beta = 0.0032
        config.firm_memory_decay = 0.970
        config.firm_memory_obs_beta = 0.9
        config.firm_memory_impact_beta = 2.8
        config.firm_memory_vol_beta = 0.8
        config.firm_memory_process_beta = 0.25
        config.firm_memory_network_beta = 0.018
        config.investor_memory_decay = 0.95
        config.investor_memory_size_beta = 1.0
    elif args.variant == "D_factors_tuned2":
        config.common_shock_beta = 0.0
        config.firm_common_factors = True
        config.exponential_memory = True
        config.factor_phi = 0.970
        config.factor_sigma = 0.020
        config.factor_rare_prob = 0.016
        config.factor_rare_sigma = 0.075
        config.factor_market_loading = 0.30
        config.factor_sector_loading = 0.34
        config.factor_random_loading = 0.045
        config.process_sigma = 0.014
        config.rare_shock_sigma = 0.080
        config.idio_vol = 0.0035
        config.price_impact = 0.0045
        config.latent_price_beta = 0.0025
        config.firm_memory_decay = 0.975
        config.firm_memory_obs_beta = 0.9
        config.firm_memory_impact_beta = 2.5
        config.firm_memory_vol_beta = 0.7
        config.firm_memory_process_beta = 0.22
        config.firm_memory_network_beta = 0.018
        config.investor_memory_decay = 0.955
        config.investor_memory_size_beta = 0.8
    elif args.variant == "D_factors_final":
        config.common_shock_beta = 0.0
        config.firm_common_factors = True
        config.exponential_memory = True
        config.factor_phi = 0.970
        config.factor_sigma = 0.024
        config.factor_rare_prob = 0.016
        config.factor_rare_sigma = 0.085
        config.factor_market_loading = 0.32
        config.factor_sector_loading = 0.36
        config.factor_random_loading = 0.050
        config.process_sigma = 0.015
        config.rare_shock_sigma = 0.085
        config.idio_vol = 0.0045
        config.price_impact = 0.0052
        config.latent_price_beta = 0.0028
        config.firm_memory_decay = 0.975
        config.firm_memory_obs_beta = 0.9
        config.firm_memory_impact_beta = 2.6
        config.firm_memory_vol_beta = 0.8
        config.firm_memory_process_beta = 0.24
        config.firm_memory_network_beta = 0.018
        config.investor_memory_decay = 0.955
        config.investor_memory_size_beta = 0.9

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
