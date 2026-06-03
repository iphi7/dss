"""
dataset.py – データ読み込み・正規化・ウィンドウ化 (TimeGrad 用)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


def load_returns(csv_path: str):
    df = pd.read_csv(csv_path)
    returns = df[["sp500", "DGS10"]].values.astype(np.float32)
    abs_vals = df[["sp500_abs", "DGS10_abs"]].values.astype(np.float32)
    dates = df["Date"].values
    return returns, abs_vals, dates


class ReturnStats:
    def __init__(self, returns: np.ndarray):
        self.mean = returns.mean(axis=0)
        self.std = returns.std(axis=0) + 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class TimeGradDataset(Dataset):
    """
    Each sample: (context, targets) pair.
      context : (context_length, 2)  normalized returns fed to the RNN
      targets : (pred_length,   2)  normalized returns to be generated
    """

    def __init__(self, returns: np.ndarray, stats: ReturnStats,
                 context_length: int = 252, pred_length: int = 21,
                 stride: int = 1):
        norm = stats.normalize(returns)
        total = context_length + pred_length
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.start_positions: list[int] = []
        for start in range(0, len(norm) - total + 1, stride):
            ctx = norm[start:start + context_length]
            tgt = norm[start + context_length:start + total]
            self.samples.append((
                torch.tensor(ctx, dtype=torch.float32),
                torch.tensor(tgt, dtype=torch.float32),
            ))
            self.start_positions.append(start)

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
      "random_disjoint" : 全体からターゲット期間が重複しない窓をランダムに val に選ぶ
    """
    returns, abs_vals, dates = load_returns(csv_path)

    if val_method == "chronological":
        split = int(len(returns) * (1 - val_ratio))
        train_ret = returns[:split]
        val_ret   = returns[split:]
        stats     = ReturnStats(train_ret)

        train_ds = TimeGradDataset(train_ret, stats, context_length, pred_length, stride)
        val_ds   = TimeGradDataset(val_ret,   stats, context_length, pred_length, stride)

        # val の start_positions は val_ret 先頭からのオフセットなので full data に戻す
        val_target_ranges = [
            (split + s + context_length,
             split + s + context_length + pred_length)
            for s in val_ds.start_positions
        ]

    elif val_method == "random_disjoint":
        stats  = ReturnStats(returns)
        all_ds = TimeGradDataset(returns, stats, context_length, pred_length, stride)

        # ターゲット期間が互いに重複しない候補：
        # target_start = s + context_length が pred_length の倍数になる窓を選ぶ
        candidates = [
            i for i, s in enumerate(all_ds.start_positions)
            if s % pred_length == 0
        ]

        rng   = np.random.default_rng(seed)
        order = rng.permutation(len(candidates)).tolist()
        n_val = max(1, int(len(candidates) * val_ratio))
        val_idx   = [candidates[j] for j in order[:n_val]]
        val_set   = set(val_idx)
        train_idx = [i for i in range(len(all_ds)) if i not in val_set]

        val_target_ranges = [
            (all_ds.start_positions[i] + context_length,
             all_ds.start_positions[i] + context_length + pred_length)
            for i in val_idx
        ]

        train_ds = Subset(all_ds, train_idx)
        val_ds   = Subset(all_ds, val_idx)

    else:
        raise ValueError(f"Unknown val_method: {val_method!r}")

    return train_ds, val_ds, stats, abs_vals, dates, val_target_ranges
