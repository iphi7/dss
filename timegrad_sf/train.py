"""
train.py – TimeGrad-SF 学習 & パス生成

通常の DDPM loss に加え、SFAG-style の Stylized Fact alignment loss
(L_ACF + L_Lev + L_CFVC) を線形ウォームアップで合算します。

使い方:
  # 学習
  python timegrad_sf/train.py train --csv output.csv --epochs 200

  # SF loss なしで学習（base TimeGrad と同じ損失）
  python timegrad_sf/train.py train --csv output.csv --sf_weight 0

  # パス生成
  python timegrad_sf/train.py generate \
      --ckpt timegrad_sf/ckpt_best.pt \
      --csv output.csv \
      --n_paths 100 \
      --business_days 504 \
      --out timegrad_sf/generated_paths.csv
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
from timegrad_sf.dataset import ReturnStats, build_dataset, load_returns
from timegrad_sf.model import TimeGradSFModel
from timegrad_sf.sf_loss import compute_sf_loss


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    (train_ds, val_ds, stats,
     _, dates, val_target_ranges) = build_dataset(
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

    model = TimeGradSFModel(
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

    # val ウィンドウのターゲット期間を CSV に保存
    val_csv_path = args.ckpt.replace(".pt", "_val_windows.csv")
    pd.DataFrame([
        {"start_idx": s, "end_idx": e,
         "start_date": dates[s], "end_date": dates[e - 1]}
        for s, e in val_target_ranges
    ]).to_csv(val_csv_path, index=False)
    print(f"val windows saved → {val_csv_path}")

    warmup_epochs = max(1, int(args.epochs * args.sf_warmup))

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        # Linear warmup: SF loss weight ramps 0 → sf_weight over warmup_epochs
        sf_w = args.sf_weight * min(1.0, epoch / warmup_epochs)

        model.train()
        train_loss = train_ddpm = train_sf = 0.0
        for ctx, targets in train_loader:
            ctx, targets = ctx.to(device), targets.to(device)

            if sf_w > 0:
                ddpm_loss, x0_hat = model.compute_loss(ctx, targets, return_x0=True)
                pred_full = torch.cat([ctx, x0_hat], dim=1)
                real_full = torch.cat([ctx, targets], dim=1)
                sf_loss, _, _, _, _ = compute_sf_loss(
                    pred_full, real_full,
                    w_acf=args.w_acf, w_lev=args.w_lev,
                    w_cfvc=args.w_cfvc, w_kurt=args.w_kurt)
                loss = ddpm_loss + sf_w * sf_loss
                train_sf += sf_loss.item()
            else:
                ddpm_loss = model.compute_loss(ctx, targets)
                loss = ddpm_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_ddpm += ddpm_loss.item()

        lr_sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ctx, targets in val_loader:
                ctx, targets = ctx.to(device), targets.to(device)
                val_loss += model.compute_loss(ctx, targets).item()

        tl     = train_loss / len(train_loader)
        t_ddpm = train_ddpm / len(train_loader)
        t_sf   = train_sf   / len(train_loader)
        vl     = val_loss   / max(1, len(val_loader))

        if sf_w > 0:
            print(f"epoch {epoch:4d}/{args.epochs}  "
                  f"train={tl:.4f} (ddpm={t_ddpm:.4f} sf={t_sf:.4f} w={sf_w:.3f})  "
                  f"val={vl:.4f}")
        else:
            print(f"epoch {epoch:4d}/{args.epochs}  train={tl:.4f}  val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save({
                "model":          model.state_dict(),
                "stats_mean":     stats.mean.tolist(),
                "stats_std":      stats.std.tolist(),
                "diff_steps":     args.diff_steps,
                "context_length": args.context_length,
                "pred_length":    args.pred_length,
                "rnn_hidden":     args.rnn_hidden,
                "hidden_dim":     args.hidden_dim,
                "n_layers":       args.n_layers,
                "step_emb_dim":   args.step_emb_dim,
                "dropout":        args.dropout,
            }, args.ckpt)
            print(f"  → saved {args.ckpt}")

    print("Done.")


# ---------------------------------------------------------------------------
# パス生成
# ---------------------------------------------------------------------------
def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = TimeGradSFModel(
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

    df_hist = pd.read_csv(args.csv, parse_dates=["Date"])
    context_length = ckpt["context_length"]

    returns_hist = df_hist[["sp500", "DGS10"]].values.astype(np.float32)
    if getattr(args, "start_date", None) is not None:
        mask = df_hist["Date"] >= pd.Timestamp(args.start_date)
        if not mask.any(): raise ValueError(f"start_date={args.start_date} がデータ範囲外")
        ctx_end = df_hist[mask].index[0]
        start_row = df_hist.loc[ctx_end]
        ctx_end = max(ctx_end, context_length)
    else:
        ctx_end, start_row = len(df_hist), df_hist.iloc[-1]
    ctx_raw_slice = returns_hist[max(0, ctx_end - context_length):ctx_end]
    if len(ctx_raw_slice) < context_length:
        pad = np.zeros((context_length - len(ctx_raw_slice), 2), dtype=np.float32)
        ctx_raw_slice = np.concatenate([pad, ctx_raw_slice], axis=0)
    ctx_norm = stats.normalize(ctx_raw_slice)
    ctx_t = (torch.tensor(ctx_norm, dtype=torch.float32)
             .unsqueeze(0)
             .expand(args.n_paths, -1, -1)
             .contiguous()
             .to(device))

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # train
    p = sub.add_parser("train")
    p.add_argument("--csv",            default="output.csv")
    p.add_argument("--context_length", type=int,   default=252)
    p.add_argument("--pred_length",    type=int,   default=21)
    p.add_argument("--stride",         type=int,   default=1)
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--batch",          type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--diff_steps",     type=int,   default=100)
    p.add_argument("--rnn_hidden",     type=int,   default=64)
    p.add_argument("--hidden_dim",     type=int,   default=64)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--step_emb_dim",   type=int,   default=64)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--val_method",     default="chronological",
                   choices=["chronological", "random_disjoint"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--ckpt",           default="timegrad_sf/ckpt_best.pt")
    # Stylized-fact alignment loss
    p.add_argument("--sf_weight",  type=float, default=1.0,
                   help="SF loss の全体重み λ_max（0 で無効）")
    p.add_argument("--sf_warmup",  type=float, default=0.2,
                   help="SF loss を線形ウォームアップするエポック割合")
    p.add_argument("--w_acf",      type=float, default=1.0,
                   help="L_ACF の重み λ₁")
    p.add_argument("--w_lev",      type=float, default=1.0,
                   help="L_Lev の重み λ₂")
    p.add_argument("--w_cfvc",     type=float, default=1.0,
                   help="L_CFVC の重み λ₃")
    p.add_argument("--w_kurt",     type=float, default=0.0,
                   help="L_kurt の重み（尖度、デフォルト無効）")

    # generate
    p = sub.add_parser("generate")
    p.add_argument("--ckpt",          default="timegrad_sf/ckpt_best.pt")
    p.add_argument("--csv",           default="output.csv")
    p.add_argument("--n_paths",       type=int, default=100)
    p.add_argument("--business_days", type=int, default=504, dest="horizon",
                   metavar="N", help="生成する営業日数 (252≈1年, 504≈2年)")
    p.add_argument("--start_date",    default=None,
                   help="生成開始日 YYYY-MM-DD。省略時はデータ末尾。")
    p.add_argument("--out",           default="timegrad_sf/generated_paths.csv")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "generate":
        generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
