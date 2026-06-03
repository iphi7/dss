"""
signature.py – バッチ対応 truncated signature 計算

Path signature の定義 (Ni et al. 2023, arXiv:2006.05421):
  d 次元経路 X = (X_0, ..., X_T) に対する level-k 係数:
  S^{i_1,...,i_k} = Σ_{1 ≤ t_1 < ... < t_k ≤ T} ΔX^{i_1}_{t_1} · ... · ΔX^{i_k}_{t_k}

  逐次更新公式 (Chen's identity):
  S^{i_1,...,i_k}_{0:t} = S^{i_1,...,i_k}_{0:t-1} + ΔX^{i_k}_t · S^{i_1,...,i_{k-1}}_{0:t-1}

Path 前処理:
  1. time_augment   : 時刻 t を先頭次元に付加 → d+1 次元
  2. cumsum_embed   : returns を累積和に変換 (連続パスへの埋め込み)
  3. lead_lag       : lead-lag 変換 → 2d 次元 (選択的)
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Path 前処理
# ─────────────────────────────────────────────────────────────

def time_augment(path: torch.Tensor) -> torch.Tensor:
    """
    path : (B, T, d)  — d 次元離散時系列（位置ベクトル）
    Returns (B, T, d+1): 先頭に均一時刻 [0/(T-1), ..., 1] を付加
    """
    B, T, d = path.shape
    t_axis = torch.linspace(0, 1, T, device=path.device, dtype=path.dtype)
    t_axis = t_axis.unsqueeze(0).unsqueeze(-1).expand(B, T, 1)  # (B, T, 1)
    return torch.cat([t_axis, path], dim=-1)  # (B, T, d+1)


def cumsum_embed(returns: torch.Tensor) -> torch.Tensor:
    """
    returns : (B, T, d)  日次リターン/変化 (Δx_t)
    Returns  : (B, T+1, d)  累積和パス（x_0=0 から始まる）
    """
    B, T, d = returns.shape
    zeros = torch.zeros(B, 1, d, device=returns.device, dtype=returns.dtype)
    return torch.cat([zeros, torch.cumsum(returns, dim=1)], dim=1)  # (B, T+1, d)


def lead_lag_transform(path: torch.Tensor) -> torch.Tensor:
    """
    path : (B, T, d)
    Lead-lag transform: (B, 2T, 2d)
    時刻 2k   → (x_k, x_k)    lead = lag
    時刻 2k+1 → (x_{k+1}, x_k) lead one step ahead
    """
    B, T, d = path.shape
    lead = path.repeat_interleave(2, dim=1)[:, 1:2*T-1, :]  # shift by 1
    lag  = path.repeat_interleave(2, dim=1)[:, :2*T-2, :]
    # Simpler: interleave the path with itself shifted
    # LL_t = (x_{ceil(t/2)}, x_{floor(t/2)})
    x_even = path.repeat_interleave(2, dim=1)[:, :2*T, :]
    x_odd  = torch.cat([path, path[:, -1:, :]], dim=1).repeat_interleave(2, dim=1)[:, :2*T, :]
    ll = torch.cat([x_even, x_odd], dim=-1)  # (B, 2T, 2d)
    return ll


# ─────────────────────────────────────────────────────────────
# Signature 計算
# ─────────────────────────────────────────────────────────────

def compute_signature(path: torch.Tensor, degree: int) -> torch.Tensor:
    """
    truncated signature of degree M.

    path   : (B, T, d)  ← 位置ベクトルの系列（既に cumsum 済みの連続パス）
    degree : int        ← truncation 次数 M

    Returns : (B, sig_dim)  where sig_dim = sum_{k=1}^{M} d^k
              level-0 の定数 1 は除外する（フィーチャーとして不要）

    Chen's identity による O(T·d^M) 逐次更新:
      S_k ∈ R^{d^k} 形状のテンソルを持ち、各ステップで
        S_k[i_1,...,i_k] += delta[i_k] * S_{k-1}[i_1,...,i_{k-1}]
      すなわち新しい S_k = S_{k-1} ⊗_outer delta に累積加算
    """
    B, T, d = path.shape
    device  = path.device
    dtype   = path.dtype

    # ステップ増分
    deltas = path[:, 1:, :] - path[:, :-1, :]  # (B, T-1, d)
    n_steps = T - 1

    # 各 level の累積テンソル
    # S[k] : (B, d, d, ..., d)  k 個の d 次元インデックス
    S = [None] + [
        torch.zeros([B] + [d]*k, device=device, dtype=dtype)
        for k in range(1, degree+1)
    ]  # S[k] for k=1..M

    for t in range(n_steps):
        dt = deltas[:, t, :]  # (B, d)
        # 高次から更新（低次 S を使うため逆順）
        for k in range(degree, 0, -1):
            if k == 1:
                S[1] = S[1] + dt                       # (B, d)
            else:
                # S[k] += S[k-1] ⊗ dt  (outer product along last dim)
                # S[k-1]: (B, d, ..., d) (k-1 axes)
                # dt:     (B, d)
                # result: (B, d, ..., d) (k axes)
                new_term = torch.einsum(
                    'b...,bi->b...i',
                    S[k-1],
                    dt
                )
                S[k] = S[k] + new_term

    # 全 level を連結して 1D ベクトルに flatten
    parts = []
    for k in range(1, degree+1):
        parts.append(S[k].reshape(B, -1))   # (B, d^k)
    return torch.cat(parts, dim=-1)         # (B, sig_dim)


def sig_dim(d: int, degree: int) -> int:
    """シグネチャの次元数（level 1..M の合計）"""
    return sum(d**k for k in range(1, degree+1))


# ─────────────────────────────────────────────────────────────
# 前処理 + 計算のパイプライン
# ─────────────────────────────────────────────────────────────

def prepare_and_sign(returns: torch.Tensor, degree: int,
                     use_time_aug: bool = True) -> torch.Tensor:
    """
    returns : (B, T, d)  正規化済み日次リターン
    degree  : int

    前処理: cumsum → [time_augment →] signature 計算

    Returns: (B, sig_dim)
    """
    path = cumsum_embed(returns)          # (B, T+1, d)
    if use_time_aug:
        path = time_augment(path)         # (B, T+1, d+1)
    return compute_signature(path, degree)


# ─────────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, d = 4, 21, 2
    degree = 3
    x = torch.randn(B, T, d)
    sig = prepare_and_sign(x, degree)
    expected_dim = sig_dim(d + 1, degree)  # +1 for time augmentation
    print(f"sig shape: {sig.shape}  expected: (B={B}, sig_dim={expected_dim})")
    assert sig.shape == (B, expected_dim), f"Shape mismatch: {sig.shape}"
    print("signature.py OK")
