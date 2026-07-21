from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description='Create quick static overview plots for market_visualizer logs.')
    ap.add_argument('--result-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    out = args.output or (args.result_dir / 'market_overview.png')
    paths = pd.read_csv(args.result_dir / 'generated_paths.csv')
    inv = pd.read_csv(args.result_dir / 'investor_states.csv')
    orders = pd.read_csv(args.result_dir / 'orders.csv')

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    axes[0].plot(paths['Date'], paths['sp500_abs'], color='black')
    axes[0].set_title('SP500 level')
    axes[1].plot(paths['Date'], paths['DGS10_abs'], color='darkorange')
    axes[1].set_title('DGS10 level')
    if not inv.empty:
        wealth = inv.groupby('day')['wealth'].agg(['mean', 'median', 'max'])
        axes[2].plot(wealth.index, wealth['mean'], label='mean')
        axes[2].plot(wealth.index, wealth['median'], label='median')
        axes[2].plot(wealth.index, wealth['max'], label='max')
        axes[2].set_title('Investor wealth')
        axes[2].legend()
    if not orders.empty:
        flow = orders.pivot_table(index='day', columns='side', values='value', aggfunc='sum').fillna(0)
        axes[3].plot(flow.index, flow['buy'] if 'buy' in flow else 0, color='green', label='buy')
        axes[3].plot(flow.index, flow['sell'] if 'sell' in flow else 0, color='red', label='sell')
        axes[3].set_title('Top-order flow')
        axes[3].legend()
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
