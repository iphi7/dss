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
    parser.add_argument("--variant", choices=["base", "A", "B"], default="base")
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
