"""
dataset.py – データ読み込み・正規化・ウィンドウ化 (TimeGrad-SF 用)

base timegrad と同じ構造。SF alignment loss は学習ループ内で
normalized returns に直接計算するため、特徴量の事前計算は不要。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_returns(csv_path: str):
    df = pd.read_csv(csv_path)
    returns  = df[["sp500", "DGS10"]].values.astype(np.float32)
    abs_vals = df[["sp500_abs", "DGS10_abs"]].values.astype(np.float32)
    dates    = df["Date"].values
    return returns, abs_vals, dates


class ReturnStats:
    def __init__(self, returns: np.ndarray):
        self.mean = returns.mean(axis=0)
        self.std  = returns.std(axis=0) + 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class TimeGradSFDataset(Dataset):
    """
    Each sample: (context, targets) pair.
      context : (context_length, 2)  normalized returns
      targets : (pred_length,   2)   normalized returns
    """

    def __init__(self, returns_raw: np.ndarray, starts: list,
                 stats: ReturnStats, context_length: int = 252,
                 pred_length: int = 21):
        norm  = stats.normalize(returns_raw)
        total = context_length + pred_length

        self.start_positions: list[int] = list(starts)
        self.samples = []
        for s in starts:
            ctx = norm[s : s + context_length]
            tgt = norm[s + context_length : s + total]
            self.samples.append((
                torch.tensor(ctx, dtype=torch.float32),
                torch.tensor(tgt, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def build_dataset(csv_path: str, context_length: int = 252,
                  pred_length: int = 21, stride: int = 1,
                  val_ratio: float = 0.1,
                  val_method: str = "chronological",
                  seed: int = 42):
    """
    val_method:
      "chronological"   : 末尾 val_ratio 分を val に使う（デフォルト）
      "random_disjoint" : ターゲット期間が重複しない窓をランダムに val に選ぶ

    Returns
    -------
    train_ds, val_ds, stats, abs_vals, dates, val_target_ranges
    """
    returns_raw, abs_vals, dates = load_returns(csv_path)
    total = context_length + pred_length

    if val_method == "chronological":
        split = int(len(returns_raw) * (1 - val_ratio))
        train_starts = list(range(0, split - total + 1, stride))
        val_starts   = list(range(split, len(returns_raw) - total + 1, stride))
        stats = ReturnStats(returns_raw[:split])

    elif val_method == "random_disjoint":
        all_starts = list(range(0, len(returns_raw) - total + 1, stride))
        candidates = [s for s in all_starts if s % pred_length == 0]
        rng      = np.random.default_rng(seed)
        order    = rng.permutation(len(candidates)).tolist()
        n_val    = max(1, int(len(candidates) * val_ratio))
        val_set  = set(candidates[j] for j in order[:n_val])
        val_starts   = sorted(val_set)
        train_starts = [s for s in all_starts if s not in val_set]
        stats = ReturnStats(returns_raw)

    else:
        raise ValueError(f"Unknown val_method: {val_method!r}")

    train_ds = TimeGradSFDataset(returns_raw, train_starts, stats,
                                 context_length, pred_length)
    val_ds   = TimeGradSFDataset(returns_raw, val_starts,   stats,
                                 context_length, pred_length)

    val_target_ranges = [
        (s + context_length, s + context_length + pred_length)
        for s in val_starts
    ]

    return train_ds, val_ds, stats, abs_vals, dates, val_target_ranges
