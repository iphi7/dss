"""
model.py – GMMSignTimeGrad (sign-only conditioning)

アーキテクチャ:
  Main LSTM:  r_t (2d, signed) → h_t (64d)
  GMM ε_t:   (1-p)·TN_center + p·TN_tail  ← arch_gm の ε_t
  EpsilonNet: cond = [h_t(64) || sign(ε_t)(2) || sign_window_rand(60)] = 126d
              target = |r_tgt|（符号なし大きさ）
  生成:       r_t = sign(ε_t) × |DDPM output|

学習時の sign(ε_t): oracle = sign(r_tgt)（DDPM が符号と大きさの関係を学習）
生成時の sign(ε_t): GMM からのサンプルの符号（ランダム）
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm as scipy_truncnorm


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half-1, 1))
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.linear1   = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)
        self.linear2   = nn.Linear(hidden_dim, hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)

    def forward(self, x, cond):
        h = F.silu(self.linear1(x))
        scale, shift = self.cond_proj(F.silu(cond)).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = F.silu(self.linear2(h))
        return self.norm(x + h)


class EpsilonNet(nn.Module):
    def __init__(self, data_dim, hidden_dim, cond_dim, n_layers, step_emb_dim, dropout):
        super().__init__()
        self.step_emb   = SinusoidalEmbedding(step_emb_dim)
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            ResidualBlock(hidden_dim, step_emb_dim + cond_dim) for _ in range(n_layers)
        ])
        self.dropout     = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t, cond):
        c = torch.cat([self.step_emb(t), cond], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = self.dropout(block(h, c))
        return self.output_proj(h)


class DDPMScheduler(nn.Module):
    def __init__(self, n_steps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.n_steps = n_steps
        betas     = torch.linspace(beta_start, beta_end, n_steps)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        a = self.alpha_bar[t].unsqueeze(-1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    def p_sample_step(self, x, eps_pred, step):
        a  = self.alphas[step]
        ab = self.alpha_bar[step]
        x  = (1 / a.sqrt()) * (x - (1 - a) / (1 - ab).sqrt() * eps_pred)
        if step > 0:
            x = x + self.betas[step].sqrt() * torch.randn_like(x)
        return x


def _sample_eps_gen(N, D, p_base, sigma_n, sigma_r, tau, device):
    """生成時: GMM からランダムサンプリング（arch_gm の ε_t と同じ）。"""
    comp = (np.random.rand(N) < p_base).astype(np.float32)
    eps  = np.empty((N, D), dtype=np.float32)
    for d in range(D):
        sn, sr   = float(sigma_n[d]), float(sigma_r[d])
        a_n, b_n = -tau / sn, tau / sn
        z_n      = scipy_truncnorm.rvs(a_n, b_n, scale=sn, size=N).astype(np.float32)
        a_r      = tau / sr
        z_abs    = scipy_truncnorm.rvs(a_r, np.inf, scale=sr, size=N).astype(np.float32)
        sgn      = np.where(np.random.rand(N) < 0.5, 1., -1.).astype(np.float32)
        z_r      = z_abs * sgn
        eps[:, d] = comp * z_r + (1 - comp) * z_n
    return torch.from_numpy(eps).to(device)


class GMMSignTimeGrad(nn.Module):
    """
    cond = [h_t(64) || sign(ε_t)(2) || sign_window_rand(60)] = 126d
    target = |r_tgt|,  生成 = sign(ε_t) × |DDPM output|
    """

    def __init__(self, data_dim=2, rnn_hidden=64, sign_window=30,
                 hidden_dim=64, n_layers=4, step_emb_dim=64,
                 diff_steps=100, dropout=0.1,
                 sigma_n=None, sigma_r=None, p_base=0.013, tau=3.0):
        super().__init__()
        self.data_dim   = data_dim
        self.diff_steps = diff_steps
        self.sign_dim   = sign_window * data_dim   # 60
        self.tau        = tau
        self.p_base     = float(p_base)
        self.sigma_n    = sigma_n if sigma_n is not None else np.ones(data_dim, np.float32)
        self.sigma_r    = sigma_r if sigma_r is not None else np.full(data_dim, 4.5, np.float32)

        self.main_lstm = nn.LSTM(data_dim, rnn_hidden, batch_first=True)
        self.rnn_drop  = nn.Dropout(dropout)

        # cond = h_t(rnn_hidden) + sign(ε_t)(data_dim) + sign_window(sign_dim) = 126
        eps_cond = rnn_hidden + data_dim + self.sign_dim
        self.eps_net   = EpsilonNet(data_dim, hidden_dim, eps_cond,
                                    n_layers, step_emb_dim, dropout)
        self.scheduler = DDPMScheduler(diff_steps)

    def _rand_sw(self, shape, device):
        return torch.randint(0, 2, shape, device=device).float() * 2 - 1

    def compute_loss(self, ctx_r, tgt_r) -> torch.Tensor:
        """
        ctx_r / tgt_r: (B, T, 2) 正規化済みリターン（signed）
        oracle sign = sign(tgt_r)  →  DDPM が符号と大きさの関係を学習
        target       = |tgt_r|
        """
        B, T_pred, D = tgt_r.shape
        dev = ctx_r.device

        _, (h, c) = self.main_lstm(ctx_r)
        h_drop    = self.rnn_drop(h)

        if T_pred > 1:
            out, _ = self.main_lstm(tgt_r[:, :-1, :], (h, c))
            out    = self.rnn_drop(out)
            h_seq  = torch.cat([h_drop.squeeze(0).unsqueeze(1), out], dim=1)
        else:
            h_seq  = h_drop.squeeze(0).unsqueeze(1)

        BT = B * T_pred
        h_flat = h_seq.reshape(BT, -1)

        # oracle sign: sign(r_tgt)  ∈ {-1, +1}
        sign_oracle = tgt_r.reshape(BT, D).sign()
        sign_oracle = torch.where(sign_oracle == 0,
                                  torch.ones_like(sign_oracle), sign_oracle)

        sw   = self._rand_sw((BT, self.sign_dim), dev)
        cond = torch.cat([h_flat, sign_oracle, sw], dim=-1)   # (BT, 126)

        x0_f      = tgt_r.reshape(BT, D).abs()                # TARGET = |r_tgt|
        t_diff    = torch.randint(0, self.diff_steps, (BT,), device=dev)
        xt, noise = self.scheduler.q_sample(x0_f, t_diff)
        eps_pred  = self.eps_net(xt, t_diff, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def generate(self, ctx_r, n_steps):
        B   = ctx_r.shape[0]
        dev = next(self.parameters()).device

        _, (h, c) = self.main_lstm(ctx_r)

        generated = []
        try:
            from tqdm import tqdm
            steps_iter = tqdm(range(n_steps), desc="generating", unit="day", leave=False)
        except ImportError:
            steps_iter = range(n_steps)

        for _ in steps_iter:
            h_sq = self.rnn_drop(h.squeeze(0))

            # GMM ε_t → 符号のみ条件入力、ε_t 全体は r_t の符号決定に使用
            eps_t    = _sample_eps_gen(B, self.data_dim, self.p_base,
                                       self.sigma_n, self.sigma_r, self.tau, dev)
            sign_eps = eps_t.sign()   # EpsilonNet には符号のみ渡す
            sw       = self._rand_sw((B, self.sign_dim), dev)

            cond = torch.cat([h_sq, sign_eps, sw], dim=-1)      # (B, 126)

            x = torch.randn(B, self.data_dim, device=dev)
            for step in reversed(range(self.diff_steps)):
                t_vec = torch.full((B,), step, device=dev, dtype=torch.long)
                eps   = self.eps_net(x, t_vec, cond)
                x     = self.scheduler.p_sample_step(x, eps, step)

            r_t = eps_t * x.abs()      # ε_t × |DDPM output|（スケールも含む）

            generated.append(r_t)
            _, (h, c) = self.main_lstm(r_t.unsqueeze(1), (h, c))

        return torch.stack(generated, dim=1)
