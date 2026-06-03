"""
train.py – TimeGrad Multi-Scale 学習 & パス生成

使い方:
  # 学習
  python timegrad_ms/train.py train --csv output.csv --epochs 200

  # パス生成
  python timegrad_ms/train.py generate \
      --ckpt timegrad_ms/ckpt_best.pt \
      --csv output.csv \
      --n_paths 100 \
      --business_days 504 \
      --out timegrad_ms/generated_paths.csv
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from timegrad_ms.dataset import ReturnStats, build_dataset, load_returns
from timegrad_ms.model import MultiScaleTimeGradModel


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    (train_ds, val_ds, stats,
     _, dates, val_target_ranges) = build_dataset(
        args.csv,
        context_length=args.slow_ctx,   # dataset stores the full slow context
        pred_length=args.pred_length,
        stride=args.stride,
        val_method=args.val_method,
        seed=args.seed,
    )
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0)

    model = MultiScaleTimeGradModel(
        data_dim=2,
        fast_hidden=args.fast_hidden,
        slow_hidden=args.slow_hidden,
        fast_ctx_len=args.fast_ctx,
        slow_period=args.slow_period,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        step_emb_dim=args.step_emb_dim,
        diff_steps=args.diff_steps,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_fast   = sum(p.numel() for p in model.fast_rnn.parameters())
    n_slow   = sum(p.numel() for p in model.slow_rnn.parameters())
    n_eps    = sum(p.numel() for p in model.eps_net.parameters())
    print(f"parameters: {n_params:,}  "
          f"(fast_rnn={n_fast:,}  slow_rnn={n_slow:,}  eps_net={n_eps:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    val_csv_path = args.ckpt.replace(".pt", "_val_windows.csv")
    pd.DataFrame([
        {"start_idx": s, "end_idx": e,
         "start_date": dates[s], "end_date": dates[e - 1]}
        for s, e in val_target_ranges
    ]).to_csv(val_csv_path, index=False)
    print(f"val windows saved → {val_csv_path}")

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for ctx, targets in train_loader:
            ctx, targets = ctx.to(device), targets.to(device)
            loss = model.compute_loss(ctx, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        lr_sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ctx, targets in val_loader:
                ctx, targets = ctx.to(device), targets.to(device)
                val_loss += model.compute_loss(ctx, targets).item()

        tl = train_loss / len(train_loader)
        vl = val_loss / max(1, len(val_loader))
        print(f"epoch {epoch:4d}/{args.epochs}  train={tl:.4f}  val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save({
                "model":        model.state_dict(),
                "stats_mean":   stats.mean.tolist(),
                "stats_std":    stats.std.tolist(),
                "diff_steps":   args.diff_steps,
                "slow_ctx":     args.slow_ctx,
                "fast_ctx":     args.fast_ctx,
                "pred_length":  args.pred_length,
                "fast_hidden":  args.fast_hidden,
                "slow_hidden":  args.slow_hidden,
                "slow_period":  args.slow_period,
                "hidden_dim":   args.hidden_dim,
                "n_layers":     args.n_layers,
                "step_emb_dim": args.step_emb_dim,
                "dropout":      args.dropout,
            }, args.ckpt)
            print(f"  → saved {args.ckpt}")

    print("Done.")


def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = MultiScaleTimeGradModel(
        data_dim=2,
        fast_hidden=ckpt["fast_hidden"],
        slow_hidden=ckpt["slow_hidden"],
        fast_ctx_len=ckpt["fast_ctx"],
        slow_period=ckpt["slow_period"],
        hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"],
        step_emb_dim=ckpt["step_emb_dim"],
        diff_steps=ckpt["diff_steps"],
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = ReturnStats.__new__(ReturnStats)
    stats.mean = np.array(ckpt["stats_mean"], dtype=np.float32)
    stats.std  = np.array(ckpt["stats_std"],  dtype=np.float32)

    df_hist = pd.read_csv(args.csv, parse_dates=["Date"])
    slow_ctx_len = ckpt["slow_ctx"]

    returns_hist = df_hist[["sp500", "DGS10"]].values.astype(np.float32)
    ctx_norm = stats.normalize(returns_hist[-slow_ctx_len:])
    ctx_t = (torch.tensor(ctx_norm, dtype=torch.float32)
             .unsqueeze(0).expand(args.n_paths, -1, -1).contiguous().to(device))

    horizon = args.horizon
    print(f"Generating {args.n_paths} paths × {horizon} business days ...")
    gen_norm = model.generate(ctx_t, horizon)
    gen_ret  = stats.denormalize(gen_norm.cpu().numpy())

    last_date  = start_row["Date"]
    last_sp500 = float(start_row["sp500_abs"])
    if getattr(args, "start_date", None) is not None:
        print(f"  開始日: {last_date.date()}  SP500={last_sp500:.2f}")
    last_dgs10 = float(df_hist["DGS10_abs"].iloc[-1])
    bdays = pd.bdate_range(start=last_date, periods=horizon + 2)[1:]

    sp500_paths = np.zeros((args.n_paths, horizon + 1))
    dgs10_paths = np.zeros((args.n_paths, horizon + 1))
    sp500_paths[:, 0] = last_sp500
    dgs10_paths[:, 0] = last_dgs10
    for d in range(horizon):
        sp500_paths[:, d + 1] = sp500_paths[:, d] * (1 + gen_ret[:, d, 0])
        dgs10_paths[:, d + 1] = dgs10_paths[:, d] + gen_ret[:, d, 1]

    sp500_rets = np.zeros_like(sp500_paths)
    dgs10_chgs = np.zeros_like(dgs10_paths)
    sp500_rets[:, 0]  = sp500_paths[:, 0] / last_sp500 - 1
    dgs10_chgs[:, 0]  = dgs10_paths[:, 0] - last_dgs10
    sp500_rets[:, 1:] = sp500_paths[:, 1:] / sp500_paths[:, :-1] - 1
    dgs10_chgs[:, 1:] = dgs10_paths[:, 1:] - dgs10_paths[:, :-1]

    rows = []
    for i in range(args.n_paths):
        for d in range(horizon + 1):
            rows.append({
                "path_id":   i,
                "Date":      bdays[d].strftime("%Y-%m-%d"),
                "sp500_abs": sp500_paths[i, d],
                "DGS10_abs": dgs10_paths[i, d],
                "sp500":     sp500_rets[i, d],
                "DGS10":     dgs10_chgs[i, d],
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.n_paths} paths × {horizon + 1} steps → {args.out}")

    final_sp500 = sp500_paths[:, -1]
    final_dgs10 = dgs10_paths[:, -1]
    print(f"SP500 final: mean={final_sp500.mean():.1f}  "
          f"5%={np.percentile(final_sp500, 5):.1f}  "
          f"95%={np.percentile(final_sp500, 95):.1f}")
    print(f"DGS10 final: mean={final_dgs10.mean():.2f}  "
          f"5%={np.percentile(final_dgs10, 5):.2f}  "
          f"95%={np.percentile(final_dgs10, 95):.2f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("train")
    p.add_argument("--csv",         default="output.csv")
    p.add_argument("--slow_ctx",    type=int, default=252,
                   help="slow LSTM に与える過去の営業日数（長いウィンドウ）")
    p.add_argument("--fast_ctx",    type=int, default=63,
                   help="fast LSTM に与える過去の営業日数（短いウィンドウ）")
    p.add_argument("--pred_length", type=int, default=21)
    p.add_argument("--stride",      type=int, default=1)
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch",       type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--diff_steps",  type=int,   default=100)
    p.add_argument("--fast_hidden", type=int,   default=64,
                   help="fast LSTM の隠れ次元（1日スケール）")
    p.add_argument("--slow_hidden", type=int,   default=64,
                   help="slow LSTM の隠れ次元（slow_period 日スケール）")
    p.add_argument("--slow_period", type=int,   default=21,
                   help="slow LSTM の更新間隔（営業日）")
    p.add_argument("--hidden_dim",     type=int,   default=64)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--step_emb_dim",   type=int,   default=64)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--val_method",     default="chronological",
                   choices=["chronological", "random_disjoint"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--ckpt",           default="timegrad_ms/ckpt_best.pt")

    p = sub.add_parser("generate")
    p.add_argument("--ckpt",          default="timegrad_ms/ckpt_best.pt")
    p.add_argument("--csv",           default="output.csv")
    p.add_argument("--n_paths",       type=int, default=100)
    p.add_argument("--business_days", type=int, default=504, dest="horizon")
    p.add_argument("--start_date",    default=None,
                   help="生成開始日 YYYY-MM-DD。省略時はデータ末尾。")
    p.add_argument("--out",           default="timegrad_ms/generated_paths.csv")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "generate":
        generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
