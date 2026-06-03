"""train.py – sa_gmm 学習スクリプト"""

import argparse, math, os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sa_gmm.dataset import build_dataset
from sa_gmm.model   import GMMSignTimeGrad


def main(args):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("sa_gmm", exist_ok=True)
    ckpt_path = "sa_gmm/ckpt.pt"

    print(f"\n{'='*55}\nsa_gmm  device={device}\n{'='*55}")

    (train_ds, val_ds, r_stats, _, _,
     p_base, sigma_n, sigma_r, _) = build_dataset(
        args.csv, context_length=args.context_length,
        pred_length=args.pred_length, val_ratio=args.val_ratio,
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = GMMSignTimeGrad(
        data_dim=2, rnn_hidden=args.rnn_hidden, sign_window=30,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        step_emb_dim=args.step_emb_dim, diff_steps=args.diff_steps,
        dropout=args.dropout, sigma_n=sigma_n, sigma_r=sigma_r,
        p_base=p_base, tau=3.0,
    ).to(device)

    n = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val, no_improve = math.inf, 0

    for epoch in range(1, args.epochs + 1):
        model.train(); tr = 0.0
        for ctx_r, tgt_r in train_ld:
            ctx_r, tgt_r = ctx_r.to(device), tgt_r.to(device)
            loss = model.compute_loss(ctx_r, tgt_r)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tr += loss.item()
        sched.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for ctx_r, tgt_r in val_ld:
                ctx_r, tgt_r = ctx_r.to(device), tgt_r.to(device)
                vl += model.compute_loss(ctx_r, tgt_r).item()

        tl = tr / len(train_ld); vl = vl / max(1, len(val_ld))
        print(f"  epoch {epoch:4d}/{args.epochs}  train={tl:.4f}  val={vl:.4f}", end="")

        if vl < best_val:
            best_val, no_improve = vl, 0
            torch.save({
                "model": model.state_dict(),
                "r_stats_mean": r_stats.mean.tolist(),
                "r_stats_std":  r_stats.std.tolist(),
                "sigma_n": sigma_n.tolist(), "sigma_r": sigma_r.tolist(),
                "p_base": float(p_base), "tau": 3.0,
                "context_length": args.context_length,
                "pred_length": args.pred_length,
                "rnn_hidden": args.rnn_hidden, "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers, "step_emb_dim": args.step_emb_dim,
                "diff_steps": args.diff_steps, "dropout": args.dropout,
                "sign_window": 30,
            }, ckpt_path)
            print("  → saved")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}."); break

    print(f"  best val = {best_val:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",            default="output.csv")
    parser.add_argument("--context_length", type=int,   default=252)
    parser.add_argument("--pred_length",    type=int,   default=21)
    parser.add_argument("--val_ratio",      type=float, default=0.1)
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--batch",          type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--rnn_hidden",     type=int,   default=64)
    parser.add_argument("--hidden_dim",     type=int,   default=64)
    parser.add_argument("--n_layers",       type=int,   default=4)
    parser.add_argument("--step_emb_dim",   type=int,   default=64)
    parser.add_argument("--diff_steps",     type=int,   default=100)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--patience",       type=int,   default=15)
    args = parser.parse_args()
    main(args)
