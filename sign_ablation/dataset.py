"""
dataset.py – sign_ablation 共通データセット

各サンプル: (ctx_r_norm, tgt_r_norm, ctx_sw, tgt_sw)

ctx_r_norm : (T_ctx, 2)        正規化済み生リターン
tgt_r_norm : (T_pred, 2)       正規化済み未来リターン
ctx_sw     : (T_ctx, win*2)    Oracle sign window（直近 win 日の sign）
tgt_sw     : (T_pred, win*2)   Oracle sign window

random variants では ctx_sw/tgt_sw を学習中にランダム置換して使う。
oracle variants では ctx_sw/tgt_sw をそのまま使う。
生成時は全 variant でランダム Ber(1/2) sign window を使う。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


def load_returns(csv_path: str):
    df   = pd.read_csv(csv_path)
    r    = df[["sp500", "DGS10"]].values.astype(np.float32)
    abs_vals = df[["sp500_abs", "DGS10_abs"]].values.astype(np.float32)
    dates    = df["Date"].values
    return r, abs_vals, dates


class ReturnStats:
    def __init__(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0) + 1e-8

    def normalize(self, x):   return (x - self.mean) / self.std
    def denormalize(self, x): return x * self.std + self.mean


def _compute_sign_windows(r: np.ndarray, win: int = 30) -> np.ndarray:
    """
    r: (T, 2) リターン列
    Returns: (T, win*2) 各ステップ t における直近 win 日の sign を flatten

    sign_window[t] = sign(r[t-win:t]) (t < win の場合は 0 パディング)
    時刻 t で利用可能な情報のみを使う（look-ahead なし）
    """
    T, D = r.shape
    signs = np.sign(r).astype(np.float32)
    signs[signs == 0] = 1.0

    # ゼロパディングして sliding window
    padded = np.concatenate([np.zeros((win, D), dtype=np.float32), signs], axis=0)
    sw_sp = sliding_window_view(padded[:, 0], win)   # (T, win)
    sw_dg = sliding_window_view(padded[:, 1], win)   # (T, win)
    return np.concatenate([sw_sp, sw_dg], axis=1)    # (T, win*2)


class SignAblationDataset(Dataset):
    def __init__(self, r_norm, sign_windows, starts, ctx_len, pred_len):
        self.samples = []
        for s in starts:
            e = s + ctx_len
            self.samples.append((
                torch.tensor(r_norm      [s:e],        dtype=torch.float32),
                torch.tensor(r_norm      [e:e+pred_len], dtype=torch.float32),
                torch.tensor(sign_windows[s:e],        dtype=torch.float32),
                torch.tensor(sign_windows[e:e+pred_len], dtype=torch.float32),
            ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def build_dataset(csv_path: str, context_length: int = 252,
                  pred_length: int = 21, sign_window: int = 30,
                  stride: int = 1, val_ratio: float = 0.1):
    """
    Returns: train_ds, val_ds, r_stats, abs_vals, dates
    """
    r, abs_vals, dates = load_returns(csv_path)
    total = context_length + pred_length
    split = int(len(r) * (1 - val_ratio))

    r_stats    = ReturnStats(r[:split])
    r_norm     = r_stats.normalize(r)
    sign_wins  = _compute_sign_windows(r, sign_window)   # (T, win*2)

    t_st = list(range(0, split - total + 1, stride))
    v_st = list(range(split, len(r) - total + 1, stride))

    train_ds = SignAblationDataset(r_norm, sign_wins, t_st, context_length, pred_length)
    val_ds   = SignAblationDataset(r_norm, sign_wins, v_st, context_length, pred_length)
    return train_ds, val_ds, r_stats, abs_vals, dates
