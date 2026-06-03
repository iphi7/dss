"""
dataset.py – SigCWGAN データ準備

スライディングウィンドウで (X_past, X_future) ペアを構築。
signature の事前計算も行う。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sigcwgan.signature import prepare_and_sign, sig_dim


class ReturnStats:
    def __init__(self, returns: np.ndarray):
        self.mean = returns.mean(axis=0).astype(np.float32)
        self.std  = returns.std(axis=0).astype(np.float32) + 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class SigCWGANDataset(Dataset):
    """
    各サンプル: (x_past_norm, x_future_norm, sig_past, sig_future)

    sig_past   : (sig_past_dim,)  past path の truncated signature
    sig_future : (sig_future_dim,) future path の truncated signature
    """

    def __init__(self, past_seqs: np.ndarray, future_seqs: np.ndarray,
                 degree_past: int = 3, degree_future: int = 3,
                 batch_size: int = 512):
        N = len(past_seqs)
        self.past   = torch.tensor(past_seqs,   dtype=torch.float32)   # (N, p_bar, 2)
        self.future = torch.tensor(future_seqs, dtype=torch.float32)   # (N, q_bar, 2)

        # バッチ単位で signature を事前計算
        print(f"  Computing signatures for {N} windows …", end="", flush=True)
        sig_p_list, sig_f_list = [], []
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            sp   = prepare_and_sign(self.past[start:end],   degree_past)
            sf   = prepare_and_sign(self.future[start:end], degree_future)
            sig_p_list.append(sp)
            sig_f_list.append(sf)
        self.sig_past   = torch.cat(sig_p_list, dim=0)    # (N, sig_past_dim)
        self.sig_future = torch.cat(sig_f_list, dim=0)    # (N, sig_future_dim)
        print(f" done. sig_past={self.sig_past.shape[1]}, sig_future={self.sig_future.shape[1]}")

    def __len__(self) -> int:
        return len(self.past)

    def __getitem__(self, idx: int):
        return (self.past[idx], self.future[idx],
                self.sig_past[idx], self.sig_future[idx])


def build_dataset(
    csv_path: str,
    p_bar: int = 20,     # past window length
    q_bar: int = 5,      # future window length
    stride: int = 1,
    val_ratio: float = 0.1,
    degree_past: int = 3,
    degree_future: int = 3,
):
    """
    Returns
    -------
    train_ds, val_ds, stats, abs_vals, dates
    """
    df       = pd.read_csv(csv_path)
    returns  = df[["sp500", "DGS10"]].values.astype(np.float32)
    abs_vals = df[["sp500_abs", "DGS10_abs"]].values.astype(np.float32)
    dates    = df["Date"].values

    N     = len(returns)
    window = p_bar + q_bar
    split = int(N * (1 - val_ratio))

    train_starts = list(range(0, split - window + 1, stride))
    val_starts   = list(range(split, N - window + 1, stride))

    stats = ReturnStats(returns[:split])

    def make_seqs(starts):
        past_seqs   = np.stack([stats.normalize(returns[s:s+p_bar])       for s in starts])
        future_seqs = np.stack([stats.normalize(returns[s+p_bar:s+window]) for s in starts])
        return past_seqs, future_seqs

    print("[train]")
    t_past, t_future = make_seqs(train_starts)
    train_ds = SigCWGANDataset(t_past, t_future, degree_past, degree_future)

    if val_starts:
        print("[val]")
        v_past, v_future = make_seqs(val_starts)
        val_ds = SigCWGANDataset(v_past, v_future, degree_past, degree_future)
    else:
        val_ds = train_ds

    return train_ds, val_ds, stats, abs_vals, dates
