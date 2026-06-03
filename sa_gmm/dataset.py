"""
dataset.py – sa_gmm データセット

各サンプル: (ctx_r_norm, tgt_r_norm, ctx_is_rare, tgt_is_rare, ctx_sw, tgt_sw)

is_rare[t,d] = |r[t,d]| > 3 * σ[t,d]
σ[t,d]       = sqrt(var(r[t-k:t,d]))  スライディングウィンドウ k=30（O(1)更新）
sign_window  = 直近 30 日の実際の符号（cond_random と異なり真の符号）
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

SLIDE_K = 30
TAU     = 3.0


def load_returns(csv_path: str):
    df = pd.read_csv(csv_path)
    r        = df[["sp500", "DGS10"]].values.astype(np.float32)
    abs_vals = df[["sp500_abs", "DGS10_abs"]].values.astype(np.float32)
    dates    = df["Date"].values
    return r, abs_vals, dates


class ReturnStats:
    def __init__(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0) + 1e-8

    def normalize(self, x):   return (x - self.mean) / self.std
    def denormalize(self, x): return x * self.std + self.mean


def calc_moving_kth_average_and_variance(x_array, k):
    """
    O(T) 合計でスライディングウィンドウ（幅 k）の平均・分散を計算（O(1)/step）。
    返り値: (mean_list, var_list)  各長さ T-k+1
      mean_list[i] = mean(x[i:i+k])
      var_list[i]  = var(x[i:i+k])
    """
    m_init = np.mean(x_array[:k])
    mean_list = [float(m_init)]
    v_init = np.var(x_array[:k])
    var_list  = [float(v_init)]
    for i in range(len(x_array) - k):
        new_mean = mean_list[-1] + (-x_array[i] + x_array[i + k]) / k
        new_var  = (var_list[-1] + mean_list[-1]**2
                    + (x_array[i + k]**2 - x_array[i]**2) / k
                    - new_mean**2)
        mean_list.append(float(new_mean))
        var_list.append(max(float(new_var), 0.0))
    return mean_list, var_list


def compute_sliding_sigma(r_norm: np.ndarray, k: int = SLIDE_K) -> np.ndarray:
    """
    σ[t,d] = sqrt(var(r[t-k:t, d]))   t < k のとき var(r[0:k]) で初期化。
    shape: (T, D)
    """
    T, D = r_norm.shape
    sigma = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        _, var_list = calc_moving_kth_average_and_variance(r_norm[:, d].tolist(), k)
        var_arr = np.maximum(var_list, 0.0, dtype=np.float64)
        # var_arr[i] = var(r[i:i+k])  → σ_t uses var_arr[max(t-k, 0)]
        indices = np.clip(np.arange(T) - k, 0, len(var_arr) - 1)
        sigma[:, d] = np.sqrt(np.maximum(var_arr[indices], 1e-8)).astype(np.float32)
    return sigma


def compute_is_rare(r_norm: np.ndarray, sigma: np.ndarray,
                    tau: float = TAU) -> np.ndarray:
    """is_rare[t,d] = (|r[t,d]| > tau * sigma[t,d]).float()"""
    return (np.abs(r_norm) > tau * sigma).astype(np.float32)


def compute_sign_windows(r_norm: np.ndarray, win: int = SLIDE_K) -> np.ndarray:
    """
    sign_window[t] = sign(r[t-win:t])  (t < win → 0 パディング)
    0 は +1 として扱う。shape: (T, win*D)
    """
    T, D = r_norm.shape
    signs  = np.where(r_norm >= 0, 1.0, -1.0).astype(np.float32)
    padded = np.concatenate([np.zeros((win, D), dtype=np.float32), signs], axis=0)
    parts  = [sliding_window_view(padded[:, d], win) for d in range(D)]
    return np.concatenate(parts, axis=1)   # (T, win*D)


def compute_gmm_params(r_norm: np.ndarray, is_rare: np.ndarray,
                       n_years: int = 5, bdays_per_year: int = 252):
    """
    最初の n_years 年のデータから GMM パラメータを推定（以降は固定）。
    Returns: p_base (float), sigma_n (D,), sigma_r (D,)
    """
    n  = min(n_years * bdays_per_year, len(r_norm))
    r3 = r_norm[:n]
    m3 = is_rare[:n]                          # (n, D)

    D      = r_norm.shape[1]
    p_base = float(m3.mean())
    sigma_n = np.ones(D,           dtype=np.float32)
    sigma_r = np.full(D, 4.5,     dtype=np.float32)

    for d in range(D):
        normal_mask = m3[:, d] == 0
        rare_mask   = m3[:, d] == 1
        if normal_mask.sum() > 1:
            sigma_n[d] = float(np.std(r3[normal_mask, d]))
        if rare_mask.sum() > 1:
            sigma_r[d] = float(np.std(r3[rare_mask, d]))

    return p_base, sigma_n, sigma_r


class GMMDataset(Dataset):
    def __init__(self, r_norm, starts, ctx_len, pred_len):
        self.samples = []
        for s in starts:
            e = s + ctx_len
            self.samples.append((
                torch.tensor(r_norm[s:e],          dtype=torch.float32),
                torch.tensor(r_norm[e:e+pred_len], dtype=torch.float32),
            ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def build_dataset(csv_path: str, context_length: int = 252,
                  pred_length: int = 21, stride: int = 1,
                  val_ratio: float = 0.1, slide_k: int = SLIDE_K):
    r, abs_vals, dates = load_returns(csv_path)
    total = context_length + pred_length
    split = int(len(r) * (1 - val_ratio))

    r_stats   = ReturnStats(r[:split])
    r_norm    = r_stats.normalize(r)
    sigma     = compute_sliding_sigma(r_norm, slide_k)
    is_rare   = compute_is_rare(r_norm, sigma)
    sign_wins = compute_sign_windows(r_norm, slide_k)
    p_base, sigma_n, sigma_r = compute_gmm_params(r_norm, is_rare)

    print(f"  GMM params (first 3yr): p_base={p_base:.4f}  "
          f"σ_n={sigma_n.round(4)}  σ_r={sigma_r.round(4)}")

    t_st = list(range(0, split - total + 1, stride))
    v_st = list(range(split, len(r) - total + 1, stride))

    train_ds = GMMDataset(r_norm, t_st, context_length, pred_length)
    val_ds   = GMMDataset(r_norm, v_st, context_length, pred_length)
    return (train_ds, val_ds, r_stats, abs_vals, dates,
            p_base, sigma_n, sigma_r, sigma)
