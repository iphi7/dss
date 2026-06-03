"""
train.py – TimeGrad 学習 & パス生成

使い方:
  # 学習
  python timegrad/train.py train --csv output.csv --epochs 100

  # パス生成
  python timegrad/train.py generate \
      --ckpt timegrad/ckpt_best.pt \
      --csv output.csv \
      --n_paths 100 \
      --business_days 504 \
      --out timegrad/generated_paths.csv
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
from timegrad.dataset import build_dataset, load_returns, ReturnStats
from timegrad.model import TimeGradModel


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, val_ds, stats, _, dates, val_target_ranges = build_dataset(
        args.csv,
        context_length=args.context_length,
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

    model = TimeGradModel(
        data_dim=2,
        rnn_hidden=args.rnn_hidden,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        step_emb_dim=args.step_emb_dim,
        diff_steps=args.diff_steps,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # val ウィンドウのターゲット期間を CSV に保存（学習開始時に1回だけ）
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
                "model":             model.state_dict(),
                "stats_mean":        stats.mean.tolist(),
                "stats_std":         stats.std.tolist(),
                "diff_steps":        args.diff_steps,
                "context_length":    args.context_length,
                "pred_length":       args.pred_length,
                "rnn_hidden":        args.rnn_hidden,
                "hidden_dim":        args.hidden_dim,
                "n_layers":          args.n_layers,
                "step_emb_dim":      args.step_emb_dim,
                "dropout":           args.dropout,
            }, args.ckpt)
            print(f"  → saved {args.ckpt}")

    print("Done.")


# ---------------------------------------------------------------------------
# パス生成
# ---------------------------------------------------------------------------
def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = TimeGradModel(
        data_dim=2,
        rnn_hidden=ckpt["rnn_hidden"],
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

    # 実績データ読み込みと start_date 解決
    df_hist = pd.read_csv(args.csv, parse_dates=["Date"])
    context_length = ckpt["context_length"]

    if getattr(args, "start_date", None) is not None:
        mask = df_hist["Date"] >= pd.Timestamp(args.start_date)
        if not mask.any(): raise ValueError(f"start_date={args.start_date} がデータ範囲外")
        start_idx = df_hist[mask].index[0]
        ctx_end = max(start_idx, context_length)
        start_row = df_hist.loc[start_idx]
    else:
        ctx_end   = len(df_hist)
        start_row = df_hist.iloc[-1]

    returns_hist = df_hist[["sp500", "DGS10"]].values.astype(np.float32)
    ctx_raw  = returns_hist[max(0, ctx_end - context_length):ctx_end]
    if len(ctx_raw) < context_length:
        pad = np.zeros((context_length - len(ctx_raw), 2), dtype=np.float32)
        ctx_raw = np.concatenate([pad, ctx_raw], axis=0)
    ctx_norm = stats.normalize(ctx_raw)
    ctx_t = (torch.tensor(ctx_norm, dtype=torch.float32)
             .unsqueeze(0)
             .expand(args.n_paths, -1, -1)
             .contiguous()
             .to(device))                                 # (n_paths, ctx_len, 2)

    horizon = args.horizon
    print(f"Generating {args.n_paths} paths × {horizon} business days ...")
    if getattr(args, "start_date", None) is not None:
        print(f"  開始日: {start_row['Date'].date()}  SP500={float(start_row['sp500_abs']):.2f}  DGS10={float(start_row['DGS10_abs']):.4f}")
    gen_norm = model.generate(ctx_t, horizon)             # (n_paths, horizon, 2)
    gen_ret  = stats.denormalize(gen_norm.cpu().numpy())  # actual returns

    # 絶対水準へ変換
    last_date    = start_row["Date"]
    last_sp500   = float(start_row["sp500_abs"])
    last_dgs10   = float(start_row["DGS10_abs"])
    bdays = pd.bdate_range(start=last_date, periods=horizon + 2)[1:]  # horizon+1 entries

    sp500_paths = np.zeros((args.n_paths, horizon + 1))
    dgs10_paths = np.zeros((args.n_paths, horizon + 1))
    sp500_paths[:, 0] = last_sp500
    dgs10_paths[:, 0] = last_dgs10
    for d in range(horizon):
        sp500_paths[:, d + 1] = sp500_paths[:, d] * (1 + gen_ret[:, d, 0])
        dgs10_paths[:, d + 1] = dgs10_paths[:, d] + gen_ret[:, d, 1]

    # リターン・変化量（output.csv と同形式）
    sp500_rets = np.zeros_like(sp500_paths)
    dgs10_chgs = np.zeros_like(dgs10_paths)
    sp500_rets[:, 0]  = sp500_paths[:, 0] / last_sp500 - 1   # ≈ 0
    dgs10_chgs[:, 0]  = dgs10_paths[:, 0] - last_dgs10        # ≈ 0
    sp500_rets[:, 1:] = sp500_paths[:, 1:] / sp500_paths[:, :-1] - 1
    dgs10_chgs[:, 1:] = dgs10_paths[:, 1:] - dgs10_paths[:, :-1]

    # CSV 出力
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # train
    p = sub.add_parser("train")
    p.add_argument("--csv",            default="output.csv")
    p.add_argument("--context_length", type=int,   default=252,
                   help="RNN に与える過去の営業日数 (約1年)")
    p.add_argument("--pred_length",    type=int,   default=21,
                   help="1 サンプルあたりの学習予測長 (約1ヶ月)")
    p.add_argument("--stride",         type=int,   default=1)
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--batch",          type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--diff_steps",     type=int,   default=100,
                   help="DDPM の拡散ステップ数")
    p.add_argument("--rnn_hidden",     type=int,   default=64)
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--n_layers",       type=int,   default=8)
    p.add_argument("--step_emb_dim",   type=int,   default=64)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--val_method",     default="chronological",
                   choices=["chronological", "random_disjoint"],
                   help="val 分割方式: chronological=末尾固定 / random_disjoint=ターゲット非重複ランダム")
    p.add_argument("--seed",           type=int,   default=42,
                   help="random_disjoint 時のシード")
    p.add_argument("--ckpt",           default="timegrad/ckpt_best.pt")

    # generate
    p = sub.add_parser("generate")
    p.add_argument("--ckpt",          default="timegrad/ckpt_best.pt")
    p.add_argument("--csv",           default="output.csv")
    p.add_argument("--n_paths",       type=int, default=100)
    p.add_argument("--business_days", type=int, default=504, dest="horizon",
                   metavar="N", help="生成する営業日数 (252≈1年, 504≈2年)")
    p.add_argument("--start_date",    default=None,
                   help="生成開始日 YYYY-MM-DD。省略時はデータ末尾。")
    p.add_argument("--out",           default="timegrad/generated_paths.csv")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "generate":
        generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
