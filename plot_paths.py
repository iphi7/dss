"""
plot_paths.py – 生成パス全体の可視化

任意のモデルの generated_paths.csv を読み込み、パスの全体像（水準・リターン・分布）を
PNG に出力する。compare_paths.py が実績との短期比較に特化しているのに対し、
このスクリプトは生成パス全体（drift・分散・外れ値）の把握を目的とする。

使い方:
  python plot_paths.py --gen sigcwgan/generated_paths.csv
  python plot_paths.py --gen timegrad_sfv2/generated_paths.csv --out paths_sfv2.png
  python plot_paths.py --gen sigcwgan/generated_paths.csv --since 2020-01
  python plot_paths.py --gen sigcwgan/generated_paths.csv --since 2010-06
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_CSV  = "output.csv"


def main(args):
    df_real = pd.read_csv(args.real, parse_dates=["Date"]).sort_values("Date")
    df_gen  = pd.read_csv(args.gen,  parse_dates=["Date"]).sort_values(["path_id", "Date"])

    all_pids = sorted(df_gen["path_id"].unique())
    n_paths  = min(len(all_pids), args.max_paths)
    pids     = all_pids[:n_paths]

    if args.since:
        real_tail = df_real[df_real["Date"] >= pd.Timestamp(args.since)]
    else:
        real_tail = df_real.tail(252)

    # ── 統計サマリー ─────────────────────────────────────
    print(f"=== {os.path.basename(args.gen)} ===")
    print(f"paths={n_paths}  steps/path={len(df_gen[df_gen.path_id==pids[0]])}")
    print("\n-- SP500 return stats (per path) --")
    print(df_gen.groupby("path_id")["sp500"]
          .agg(mean="mean", std="std", min="min", max="max")
          .round(5).to_string())
    print("\n-- SP500 level (start → end) --")
    for pid in pids:
        sub = df_gen[df_gen.path_id == pid].sort_values("Date")
        print(f"  path {pid:2d}: {sub.sp500_abs.iloc[0]:>12,.1f} → {sub.sp500_abs.iloc[-1]:>12,.1f}")

    # ── カラーマップ ─────────────────────────────────────
    cmap   = plt.cm.tab20(np.linspace(0, 1, n_paths))
    black  = "black"
    gray   = "gray"

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    model_name = os.path.splitext(os.path.basename(args.gen))[0].replace("generated_paths", "gen")
    fig.suptitle(f"Generated Paths: {os.path.dirname(args.gen)}  (n={n_paths})", fontsize=12)

    def plot_level(ax, col, title, ylabel, log_scale=False):
        ax.plot(real_tail["Date"], real_tail[col],
                color=black, lw=1.8, label="Real (tail)", zorder=5)
        for i, pid in enumerate(pids):
            sub = df_gen[df_gen.path_id == pid]
            ax.plot(sub["Date"], sub[col], color=cmap[i], alpha=0.55, lw=0.8)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=25)

    def plot_return(ax, col, title, ylabel):
        ax.axhline(0, color=gray, lw=0.5)
        for i, pid in enumerate(pids):
            sub = df_gen[df_gen.path_id == pid]
            ax.plot(sub["Date"], sub[col], color=cmap[i], alpha=0.5, lw=0.6)
        ax.plot(real_tail["Date"], real_tail[col],
                color=black, lw=1.0, alpha=0.7, label="Real (tail)")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=25)

    # Row 0: SP500 level (linear & log)
    plot_level(axes[0, 0], "sp500_abs", "SP500 Level",          "Price")
    plot_level(axes[0, 1], "sp500_abs", "SP500 Level (log)",    "Price (log)", log_scale=True)

    # Row 1: DGS10 level & SP500 returns
    plot_level(axes[1, 0], "DGS10_abs", "DGS10 Level",          "Rate (%)")
    plot_return(axes[1, 1], "sp500",    "SP500 Daily Return",   "Return")

    # Row 2: DGS10 changes & return distribution
    plot_return(axes[2, 0], "DGS10",    "DGS10 Daily Change",   "Change (%pt)")

    ax = axes[2, 1]
    all_gen  = df_gen["sp500"].dropna().values
    all_real = df_real["sp500"].dropna().values
    clip = max(abs(np.percentile(all_real, 1)),
               abs(np.percentile(all_real, 99))) * 3
    bins = np.linspace(-clip, clip, 80)
    ax.hist(all_real, bins=bins, alpha=0.45, color=black,
            density=True, label=f"Real  μ={all_real.mean():.5f}")
    ax.hist(all_gen,  bins=bins, alpha=0.45, color="steelblue",
            density=True, label=f"Gen   μ={all_gen.mean():.5f}")
    ax.set_title("SP500 Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen",       required=True,
                        help="generated_paths.csv へのパス")
    parser.add_argument("--real",      default=REAL_CSV)
    parser.add_argument("--since",     default=None,
                        help="実データの表示開始年月（例: 2020-01）。省略時は直近252日")
    parser.add_argument("--out",       default=None,
                        help="出力 PNG（省略時: gen と同ディレクトリの paths_overview.png）")
    parser.add_argument("--max_paths", type=int, default=20)
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.gen), "paths_overview.png")

    main(args)
