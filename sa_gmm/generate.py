"""generate.py – sa_gmm パス生成"""

import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sa_gmm.dataset import load_returns, ReturnStats
from sa_gmm.model   import GMMSignTimeGrad


def load_model(device):
    ckpt = torch.load("sa_gmm/ckpt.pt", map_location=device, weights_only=False)
    m = GMMSignTimeGrad(
        data_dim=2, rnn_hidden=ckpt["rnn_hidden"],
        sign_window=ckpt.get("sign_window", 30),
        hidden_dim=ckpt["hidden_dim"], n_layers=ckpt["n_layers"],
        step_emb_dim=ckpt["step_emb_dim"], diff_steps=ckpt["diff_steps"],
        dropout=0.0,
        sigma_n=np.array(ckpt["sigma_n"], dtype=np.float32),
        sigma_r=np.array(ckpt["sigma_r"], dtype=np.float32),
        p_base=ckpt["p_base"], tau=ckpt["tau"],
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    r_stats      = ReturnStats.__new__(ReturnStats)
    r_stats.mean = np.array(ckpt["r_stats_mean"], dtype=np.float32)
    r_stats.std  = np.array(ckpt["r_stats_std"],  dtype=np.float32)
    return m, r_stats, ckpt["context_length"]


def resolve_start(df, start_date, ctx_len):
    if start_date is None:
        start_idx = len(df) - 1
    else:
        mask = df["Date"] >= pd.Timestamp(start_date)
        if not mask.any():
            raise ValueError(f"start_date={start_date} がデータ範囲外")
        start_idx = df.index.get_loc(df[mask].index[0])
    start_row = df.iloc[start_idx]
    ctx_end   = start_idx
    ctx_begin = max(0, ctx_end - ctx_len)
    pad_len   = ctx_len - (ctx_end - ctx_begin)
    print(f"  開始日: {start_row['Date'].date()}  "
          f"SP500={start_row['sp500_abs']:.2f}  DGS10={start_row['DGS10_abs']:.4f}")
    return start_idx, start_row, ctx_begin, ctx_end, pad_len


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, r_stats, ctx_len = load_model(device)

    r_raw, _, _ = load_returns(args.csv)
    df = pd.read_csv(args.csv, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    start_idx, start_row, ctx_begin, ctx_end, pad_len = \
        resolve_start(df, args.start_date, ctx_len)

    r_norm_all = r_stats.normalize(r_raw)
    ctx_r_norm = r_norm_all[ctx_begin:ctx_end]

    if pad_len > 0:
        ctx_r_norm = np.vstack([np.zeros((pad_len, 2), np.float32), ctx_r_norm])
        print(f"  コンテキスト先頭 {pad_len} 日分を 0 パディング")

    B = args.n_paths
    ctx_r_t = (torch.tensor(ctx_r_norm, dtype=torch.float32)
               .unsqueeze(0).expand(B, -1, -1).contiguous().to(device))

    horizon = args.business_days
    print(f"  Generating {B} paths × {horizon} days ...")
    with torch.no_grad():
        gen_norm = model.generate(ctx_r_t, horizon)

    gen = r_stats.denormalize(gen_norm.cpu().numpy())

    init_sp  = float(start_row["sp500_abs"])
    init_dgs = float(start_row["DGS10_abs"])
    bdays    = pd.bdate_range(start=start_row["Date"] + pd.Timedelta(days=1), periods=horizon)

    rows = []
    for pid in tqdm(range(B), desc="building paths", unit="path"):
        sp_cur, dg_cur = init_sp, init_dgs
        for t in range(horizon):
            r_sp = float(gen[pid, t, 0]); r_dg = float(gen[pid, t, 1])
            sp_cur = sp_cur * (1.0 + r_sp); dg_cur = dg_cur + r_dg
            rows.append({"path_id": pid, "Date": bdays[t].strftime("%Y-%m-%d"),
                         "sp500_abs": sp_cur, "DGS10_abs": dg_cur,
                         "sp500": r_sp, "DGS10": r_dg})

    out = "sa_gmm/generated_paths.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           default="output.csv")
    parser.add_argument("--n_paths",       type=int,  default=20)
    parser.add_argument("--business_days", type=int,  default=252)
    parser.add_argument("--start_date",    default=None)
    args = parser.parse_args()
    main(args)
