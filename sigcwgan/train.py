"""
train.py – SigCWGAN 学習・生成 CLI

Algorithm 1 (Ni et al. 2023):
  Step 1 (one-off): 実データから S_past→S_future の線形回帰 L̂ を fit
  Step 2 (training): Generator を L̂ベースの損失で最適化
    loss = ||L̂(S_past) - E_{fake}[S_future_fake]||_2

Usage
-----
python sigcwgan/train.py train --csv output.csv --epochs 200
python sigcwgan/train.py generate --csv output.csv --n_paths 20 --business_days 252
"""

import argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sigcwgan.model    import ARFNNGenerator
from sigcwgan.dataset  import build_dataset, ReturnStats
from sigcwgan.signature import prepare_and_sign, sig_dim


# ─────────────────────────────────────────────────────────────
# Step 1: Linear Regression Estimator  L̂(S_past) → E[S_future]
# ─────────────────────────────────────────────────────────────

def fit_linear_regression(sig_past: torch.Tensor,
                          sig_future: torch.Tensor,
                          ridge: float = 1e-4) -> nn.Linear:
    """
    Fit L̂: R^{sig_past_dim} → R^{sig_future_dim} using ridge regression.
    L̂(x) = x @ W + b

    sig_past   : (N, sig_past_dim)
    sig_future : (N, sig_future_dim)
    ridge      : L2 regularisation coefficient

    Returns: nn.Linear (no grad) with fitted weights
    """
    Sp = sig_past.float()
    Sf = sig_future.float()
    N, d_in  = Sp.shape
    _, d_out = Sf.shape

    # Add bias column
    ones = torch.ones(N, 1, device=Sp.device)
    Sp_b = torch.cat([Sp, ones], dim=1)  # (N, d_in+1)

    # Ridge: (Sp_b^T Sp_b + ridge * I)^{-1} Sp_b^T Sf
    A = Sp_b.T @ Sp_b + ridge * torch.eye(d_in + 1, device=Sp.device)
    B = Sp_b.T @ Sf
    try:
        W_b = torch.linalg.solve(A, B)          # (d_in+1, d_out)
    except Exception:
        W_b = torch.linalg.lstsq(Sp_b, Sf).solution

    W = W_b[:d_in, :].T    # (d_out, d_in)
    b = W_b[d_in, :]                  # (d_out,)

    lin = nn.Linear(d_in, d_out, bias=True)
    with torch.no_grad():
        lin.weight.copy_(W)
        lin.bias.copy_(b)
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


# ─────────────────────────────────────────────────────────────
# SigCWGAN loss
# ─────────────────────────────────────────────────────────────

def sigcwgan_loss(L_hat: nn.Linear,
                  sig_past: torch.Tensor,
                  gen_futures: torch.Tensor,
                  degree_future: int,
                  n_mc: int) -> torch.Tensor:
    """
    loss = ||L̂(S_past) - E[S_future_fake]||_2

    sig_past    : (B, sig_past_dim)
    gen_futures : (B * n_mc, q_bar, 2)  generated future paths
    """
    B = sig_past.shape[0]

    # Target: L̂(S_past)
    target = L_hat(sig_past)   # (B, sig_future_dim)

    # Estimate E[S_future_fake] via Monte-Carlo mean
    sig_fake = prepare_and_sign(gen_futures, degree_future)      # (B*n_mc, sig_future_dim)
    sig_fake = sig_fake.reshape(B, n_mc, -1).mean(dim=1)         # (B, sig_future_dim)

    return (target - sig_fake).norm(dim=1).mean()


# ─────────────────────────────────────────────────────────────
# train
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds, val_ds, stats, abs_vals, dates = build_dataset(
        args.csv,
        p_bar=args.p_bar, q_bar=args.q_bar,
        stride=args.stride, val_ratio=args.val_ratio,
        degree_past=args.deg_past, degree_future=args.deg_future,
    )
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")

    # ── Step 1: Fit L̂ from all training data ──
    print("Fitting L̂ (linear regression) …")
    all_sp  = train_ds.sig_past.to(device)
    all_sf  = train_ds.sig_future.to(device)
    L_hat   = fit_linear_regression(all_sp, all_sf, ridge=args.ridge).to(device)
    print(f"  L̂: {all_sp.shape[1]} → {all_sf.shape[1]}")

    # ── Step 2: Train generator ──
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    G = ARFNNGenerator(
        data_dim=2, p_bar=args.p_bar, noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"Generator params: {n_params:,}")

    opt = torch.optim.Adam(G.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        G.train()
        train_losses = []
        for x_past, x_future, sp, sf in train_loader:
            x_past = x_past.to(device)
            sp     = sp.to(device)
            B      = x_past.shape[0]

            # Generate n_mc fake futures per sample
            x_past_rep = x_past.repeat_interleave(args.n_mc, dim=0)  # (B*n_mc, p_bar, 2)
            fake_future = G.generate_sequence(x_past_rep, args.q_bar) # (B*n_mc, q_bar, 2)

            loss = sigcwgan_loss(L_hat, sp, fake_future, args.deg_future, args.n_mc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        G.eval()
        val_losses = []
        with torch.no_grad():
            for x_past, x_future, sp, sf in val_loader:
                x_past = x_past.to(device)
                sp     = sp.to(device)
                B      = x_past.shape[0]
                x_past_rep  = x_past.repeat_interleave(args.n_mc, dim=0)
                fake_future = G.generate_sequence(x_past_rep, args.q_bar)
                val_losses.append(
                    sigcwgan_loss(L_hat, sp, fake_future, args.deg_future, args.n_mc).item()
                )

        tr_loss = np.mean(train_losses)
        va_loss = np.mean(val_losses)
        print(f"epoch {epoch:4d}/{args.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}", end="")

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({
                "G_state":     G.state_dict(),
                "L_hat_state": L_hat.state_dict(),
                "stats_mean":  stats.mean,
                "stats_std":   stats.std,
                "hyperparams": {
                    "p_bar": args.p_bar, "q_bar": args.q_bar,
                    "noise_dim": args.noise_dim,
                    "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
                    "deg_past": args.deg_past, "deg_future": args.deg_future,
                },
            }, args.ckpt)
            print("  → saved")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break


# ─────────────────────────────────────────────────────────────
# generate
# ─────────────────────────────────────────────────────────────

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp   = ckpt["hyperparams"]

    G = ARFNNGenerator(
        data_dim=2, p_bar=hp["p_bar"], noise_dim=hp["noise_dim"],
        hidden_dim=hp["hidden_dim"], n_layers=hp["n_layers"],
    ).to(device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    stats = ReturnStats.__new__(ReturnStats)
    stats.mean = ckpt["stats_mean"]
    stats.std  = ckpt["stats_std"]

    df_real   = pd.read_csv(args.csv, parse_dates=["Date"])
    last_row  = df_real.iloc[-1]
    last_sp   = float(last_row["sp500_abs"])
    last_dgs  = float(last_row["DGS10_abs"])
    last_date = df_real["Date"].iloc[-1]

    p_bar = hp["p_bar"]
    q_bar = hp["q_bar"]
    bd    = args.business_days

    # 最後の p_bar ステップを初期コンテキストとして使用
    recent_returns = df_real[["sp500", "DGS10"]].values[-p_bar:].astype(np.float32)
    ctx_norm = torch.tensor(stats.normalize(recent_returns), dtype=torch.float32, device=device)
    ctx_norm = ctx_norm.unsqueeze(0)  # (1, p_bar, 2)

    n_chunks = math.ceil(bd / q_bar)
    rows = []

    for path_id in range(args.n_paths):
        sp_cur  = last_sp
        dgs_cur = last_dgs
        sp_all, dgs_all = [], []
        buf = ctx_norm.clone()  # running context buffer

        for _ in range(n_chunks):
            with torch.no_grad():
                fake = G.generate_sequence(buf, q_bar)  # (1, q_bar, 2)
            ret_norm = fake[0].cpu().numpy()             # (q_bar, 2)
            ret = ret_norm * stats.std + stats.mean
            sp_all.append(ret[:, 0])
            dgs_all.append(ret[:, 1])
            # Update context buffer (values already in normalized space)
            new_buf = np.vstack([buf[0, q_bar:].cpu().numpy(), ret_norm])
            buf = torch.tensor(new_buf, dtype=torch.float32, device=device).unsqueeze(0)

        sp_returns  = np.concatenate(sp_all)[:bd]
        dgs_returns = np.concatenate(dgs_all)[:bd]
        biz_dates   = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=bd)

        for d, r_sp, r_dgs in zip(biz_dates, sp_returns, dgs_returns):
            sp_cur  = sp_cur  * (1.0 + r_sp)
            dgs_cur = dgs_cur + r_dgs
            rows.append({"path_id": path_id, "Date": d.strftime("%Y-%m-%d"),
                         "sp500_abs": sp_cur, "DGS10_abs": dgs_cur,
                         "sp500": r_sp, "DGS10": r_dgs})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved → {args.out}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SigCWGAN")
    sub    = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("train")
    p.add_argument("--csv",          default="output.csv")
    p.add_argument("--p_bar",        type=int,   default=20,    help="過去窓長")
    p.add_argument("--q_bar",        type=int,   default=5,     help="未来窓長")
    p.add_argument("--stride",       type=int,   default=1)
    p.add_argument("--val_ratio",    type=float, default=0.1)
    p.add_argument("--deg_past",     type=int,   default=3,     help="過去 signature 次数")
    p.add_argument("--deg_future",   type=int,   default=3,     help="未来 signature 次数")
    p.add_argument("--ridge",        type=float, default=1e-4,  help="線形回帰の ridge 係数")
    p.add_argument("--noise_dim",    type=int,   default=5)
    p.add_argument("--hidden_dim",   type=int,   default=64)
    p.add_argument("--n_layers",     type=int,   default=3)
    p.add_argument("--n_mc",         type=int,   default=10,    help="Monte-Carlo サンプル数")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--patience",     type=int,   default=15,    help="early stopping patience")
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--ckpt",         default="sigcwgan/ckpt_best.pt")

    p = sub.add_parser("generate")
    p.add_argument("--csv",           default="output.csv")
    p.add_argument("--ckpt",          default="sigcwgan/ckpt_best.pt")
    p.add_argument("--n_paths",       type=int, default=20)
    p.add_argument("--business_days", type=int, default=252)
    p.add_argument("--out",           default="sigcwgan/generated_paths.csv")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "generate":
        generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
