"""
generate.py – sign_ablation 共通パス生成

生成時は全モードでランダム Ber(1/2) sign window を使う。
（oracle モードも生成時はランダム — 学習時のみ真の sign を使用）

使い方:
  # デフォルト（最新日から生成）
  python sign_ablation/generate.py --mode cond_random

  # 20年前の値を初期値にして5040日（20年分）生成
  python sign_ablation/generate.py --mode cond_random --business_days 5040 --start_date 2006-05-14

  # 60年前（データ先頭）の値を初期値にして生成
  python sign_ablation/generate.py --mode cond_random --business_days 5040 --start_date 1966-01-03
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sign_ablation.dataset import load_returns, ReturnStats, _compute_sign_windows
from sign_ablation.model   import SignAblationTimeGrad

MODES = ["cond_random", "lstm_random", "cond_oracle", "lstm_oracle"]


def load_model(mode, device):
    ckpt_path = os.path.join("sign_ablation", mode, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = SignAblationTimeGrad(
        sign_mode=ckpt["sign_mode"], data_dim=2,
        sign_window=ckpt["sign_window"],
        rnn_hidden=ckpt["rnn_hidden"], hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"], step_emb_dim=ckpt["step_emb_dim"],
        diff_steps=ckpt["diff_steps"], dropout=0.0,
    ).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    r_stats = ReturnStats.__new__(ReturnStats)
    r_stats.mean = np.array(ckpt["r_stats_mean"], dtype=np.float32)
    r_stats.std  = np.array(ckpt["r_stats_std"],  dtype=np.float32)
    return m, r_stats, ckpt["context_length"], ckpt["sign_window"]


def resolve_start(df, start_date, ctx_len):
    """
    start_date に対応する行インデックスと初期価格を返す。

    Returns:
        start_idx : df における開始行インデックス（その日の終値が初期値）
        start_row : 開始行の Series
        ctx_slice : [start_idx - ctx_len, start_idx) のスライス範囲
                    データ先頭より前は 0 パディングで補完
    """
    if start_date is None:
        # デフォルト: データ末尾
        start_idx = len(df) - 1
    else:
        # start_date 以降で最初に存在する行
        mask = df["Date"] >= pd.Timestamp(start_date)
        if not mask.any():
            raise ValueError(f"start_date={start_date} がデータ範囲外です "
                             f"（最終日: {df['Date'].iloc[-1].date()}）")
        start_idx = df[mask].index[0]
        # iloc に変換
        start_idx = df.index.get_loc(start_idx)

    start_row = df.iloc[start_idx]
    print(f"  開始日: {start_row['Date'].date()}  "
          f"SP500={start_row['sp500_abs']:.2f}  DGS10={start_row['DGS10_abs']:.4f}")

    ctx_end   = start_idx          # この行は含まない（初期値として使うため）
    ctx_begin = max(0, ctx_end - ctx_len)
    pad_len   = ctx_len - (ctx_end - ctx_begin)   # 先頭不足分

    return start_idx, start_row, ctx_begin, ctx_end, pad_len


def generate_one(mode, args, device):
    model, r_stats, ctx_len, sign_window = load_model(mode, device)

    r_raw, abs_vals, dates = load_returns(args.csv)
    df = pd.read_csv(args.csv, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    start_idx, start_row, ctx_begin, ctx_end, pad_len = \
        resolve_start(df, args.start_date, ctx_len)

    init_sp  = float(start_row["sp500_abs"])
    init_dgs = float(start_row["DGS10_abs"])

    # コンテキスト用リターン（足りない先頭は 0 パディング）
    ctx_r_raw  = r_raw[ctx_begin:ctx_end]              # (ctx_end - ctx_begin, 2)
    ctx_sw_raw = _compute_sign_windows(r_raw, sign_window)[ctx_begin:ctx_end]

    if pad_len > 0:
        ctx_r_raw  = np.concatenate([np.zeros((pad_len, 2), dtype=np.float32), ctx_r_raw], axis=0)
        ctx_sw_raw = np.concatenate([np.zeros((pad_len, ctx_sw_raw.shape[1]), dtype=np.float32),
                                     ctx_sw_raw], axis=0)
        print(f"  コンテキスト先頭 {pad_len} 日分を 0 パディング")

    r_norm    = r_stats.normalize(ctx_r_raw)
    sign_wins = ctx_sw_raw

    ctx_r  = (torch.tensor(r_norm,    dtype=torch.float32)
              .unsqueeze(0).expand(args.n_paths, -1, -1).contiguous().to(device))
    ctx_sw = (torch.tensor(sign_wins, dtype=torch.float32)
              .unsqueeze(0).expand(args.n_paths, -1, -1).contiguous().to(device))

    horizon = args.business_days
    with torch.no_grad():
        gen_norm = model.generate(ctx_r, ctx_sw, horizon)   # (N, T, 2)

    gen = r_stats.denormalize(gen_norm.cpu().numpy())

    # 生成パスの日付（開始日の翌営業日から）
    bdays = pd.bdate_range(
        start=start_row["Date"] + pd.Timedelta(days=1), periods=horizon
    )
    rows = []
    for pid in tqdm(range(args.n_paths), desc="building paths", unit="path"):
        sp_cur, dg_cur = init_sp, init_dgs
        for t in range(horizon):
            r_sp = float(gen[pid, t, 0])
            r_dg = float(gen[pid, t, 1])
            sp_cur = sp_cur * (1.0 + r_sp)
            dg_cur = dg_cur + r_dg
            rows.append({"path_id": pid, "Date": bdays[t].strftime("%Y-%m-%d"),
                         "sp500_abs": sp_cur, "DGS10_abs": dg_cur,
                         "sp500": r_sp, "DGS10": r_dg})

    out_path = os.path.join("sign_ablation", mode, "generated_paths.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  [{mode}] saved → {out_path}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modes  = [args.mode] if args.mode else MODES
    for mode in tqdm(modes, desc="modes", unit="mode"):
        ckpt = os.path.join("sign_ablation", mode, "ckpt.pt")
        if not os.path.exists(ckpt):
            print(f"  [{mode}] checkpoint not found, skipping")
            continue
        print(f"Generating [{mode}] {args.n_paths} paths × {args.business_days} days ...")
        generate_one(mode, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",          default=None, choices=MODES + [None])
    parser.add_argument("--csv",           default="output.csv")
    parser.add_argument("--n_paths",       type=int,    default=20)
    parser.add_argument("--business_days", type=int,    default=252)
    parser.add_argument("--start_date",    default=None,
                        help="生成開始日（YYYY-MM-DD）。省略時はデータ末尾。"
                             "指定した日の価格を初期値とし、直前 ctx_len 日を"
                             "LSTMコンテキストとして使用する。")
    args = parser.parse_args()
    main(args)
