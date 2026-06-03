"""
model.py – TimeGrad with Multi-Scale LSTM (Clockwork RNN style)

Two LSTM modules operating at different time scales:
  - fast_rnn : updates every day (step-level patterns, short-term trends)
  - slow_rnn : updates every slow_period days (monthly patterns, regime changes)

EpsilonNet is conditioned on [h_fast_t, h_slow_t] — the concatenation of
both hidden states — capturing micro and macro structure simultaneously.

How slow_rnn works (causal, no lookahead):
  - The full sequence is divided into non-overlapping windows of slow_period days.
  - Each window is summarised as the mean return over that window.
  - slow_rnn processes these summaries sequentially.
  - For steps in window k, the slow conditioning is the slow_rnn *output from
    window k-1* (the previous completed window), so no future data leaks in.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear1  = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)
        self.linear2  = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.linear1(x))
        scale, shift = self.cond_proj(F.silu(cond)).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = F.silu(self.linear2(h))
        return self.norm(x + h)


class EpsilonNet(nn.Module):
    """ε_θ(x^n, n, h_fast, h_slow)  →  ε̂ ∈ R^2"""

    def __init__(self, data_dim: int = 2, hidden_dim: int = 64,
                 rnn_dim: int = 128, n_layers: int = 4,
                 step_emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.step_emb  = SinusoidalEmbedding(step_emb_dim)
        cond_dim       = step_emb_dim + rnn_dim          # sinusoidal + [h_fast, h_slow]
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks    = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim) for _ in range(n_layers)
        ])
        self.dropout   = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                rnn_h: torch.Tensor) -> torch.Tensor:
        # rnn_h : (B, fast_hidden + slow_hidden)
        cond = torch.cat([self.step_emb(t), rnn_h], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = self.dropout(block(h, cond))
        return self.output_proj(h)


class DDPMScheduler(nn.Module):
    def __init__(self, n_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.n_steps = n_steps
        betas     = torch.linspace(beta_start, beta_end, n_steps)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        a = self.alpha_bar[t].unsqueeze(-1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    def p_sample_step(self, x: torch.Tensor, eps_pred: torch.Tensor,
                      step: int) -> torch.Tensor:
        a  = self.alphas[step]
        ab = self.alpha_bar[step]
        coeff = (1 - a) / (1 - ab).sqrt()
        x = (1 / a.sqrt()) * (x - coeff * eps_pred)
        if step > 0:
            x = x + self.betas[step].sqrt() * torch.randn_like(x)
        return x


class MultiScaleTimeGradModel(nn.Module):
    """
    TimeGrad with two LSTM modules at different time scales.

    Training (teacher forcing):
      1. ctx + targets → slow_encode → per-step slow context (causal shift)
      2. ctx → fast_rnn → h_fast_0 ; teacher-force targets → h_fast_1..T
      3. DDPM loss conditioned on [h_fast_t, h_slow_t] at each target step

    Generation (autoregressive):
      1. ctx → initialize both RNNs
      2. Each step: DDPM reverse ← [h_fast_t, h_slow_t]
      3. fast_rnn updates every step; slow_rnn updates every slow_period steps
         using the MEAN of the last slow_period generated values
    """

    def __init__(self, data_dim: int = 2,
                 fast_hidden: int = 64, slow_hidden: int = 64,
                 fast_ctx_len: int = 63, slow_period: int = 21,
                 hidden_dim: int = 64, n_layers: int = 4,
                 step_emb_dim: int = 64, diff_steps: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        self.data_dim    = data_dim
        self.diff_steps  = diff_steps
        self.slow_period = slow_period
        self.fast_ctx_len = fast_ctx_len   # fast LSTM uses only the last fast_ctx_len days

        self.fast_rnn    = nn.LSTM(data_dim, fast_hidden, batch_first=True)
        self.slow_rnn    = nn.LSTM(data_dim, slow_hidden, batch_first=True)
        self.fast_drop   = nn.Dropout(dropout)
        self.slow_drop   = nn.Dropout(dropout)

        rnn_dim = fast_hidden + slow_hidden
        self.eps_net   = EpsilonNet(data_dim, hidden_dim, rnn_dim,
                                     n_layers, step_emb_dim, dropout)
        self.scheduler = DDPMScheduler(diff_steps)

    # ------------------------------------------------------------------
    # Slow encoding (causal)
    # ------------------------------------------------------------------
    def _slow_encode(self, x: torch.Tensor):
        """
        Compute a per-step slow context with a causal one-window shift.

        For step t in window k  →  slow context = slow_rnn output from window k-1.
        Window 0 uses a zero vector (no prior slow output).

        x       : (B, T, D)
        returns : slow_ctx (B, T, slow_hidden),
                  h_slow (1, B, slow_hidden),  c_slow (1, B, slow_hidden)
        """
        B, T, D = x.shape
        P = self.slow_period
        n_win = math.ceil(T / P)
        pad   = n_win * P - T

        x_pad    = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        win_mean = x_pad.reshape(B, n_win, P, D).mean(dim=2)   # (B, n_win, D)

        slow_out, (h_slow, c_slow) = self.slow_rnn(win_mean)   # (B, n_win, slow_h)

        # Shift right: prepend zeros, drop last
        zeros   = slow_out.new_zeros(B, 1, slow_out.shape[-1])
        shifted = torch.cat([zeros, slow_out[:, :-1, :]], dim=1)  # (B, n_win, slow_h)

        # Repeat each window P times, trim to T
        slow_ctx = shifted.repeat_interleave(P, dim=1)[:, :T, :]   # (B, T, slow_h)
        return slow_ctx, h_slow, c_slow

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def compute_loss(self, slow_ctx: torch.Tensor, targets: torch.Tensor,
                     return_x0: bool = False):
        """
        slow_ctx : (B, T_slow, 2)  normalized context for slow LSTM (long window)
        targets  : (B, T_pred, 2)  normalized targets

        fast LSTM uses only the last fast_ctx_len days of slow_ctx.
        Slow context is computed over the full slow_ctx+targets sequence with a
        one-window causal shift (no lookahead).
        """
        B, T_slow, D = slow_ctx.shape
        _, T_pred, _ = targets.shape

        # ── fast LSTM uses only the most recent fast_ctx_len days ──────
        fast_ctx = slow_ctx[:, -self.fast_ctx_len:, :]       # (B, T_fast, D)

        # ── Slow context for all target steps ──────────────────────────
        full_seq = torch.cat([slow_ctx, targets], dim=1)     # (B, T_slow+T_pred, D)
        slow_full, _, _ = self._slow_encode(full_seq)
        slow_cond = self.slow_drop(slow_full[:, T_slow:, :]) # (B, T_pred, slow_h)

        # ── Fast context (teacher forcing) ─────────────────────────────
        _, (h_fast, c_fast) = self.fast_rnn(fast_ctx)
        h_fast_d = self.fast_drop(h_fast)

        if T_pred > 1:
            fast_out, _ = self.fast_rnn(targets[:, :-1, :], (h_fast, c_fast))
            fast_cond   = torch.cat(
                [h_fast_d.squeeze(0).unsqueeze(1),
                 self.fast_drop(fast_out)], dim=1)            # (B, T_pred, fast_h)
        else:
            fast_cond = h_fast_d.squeeze(0).unsqueeze(1)     # (B, 1, fast_h)

        # ── Combined conditioning ──────────────────────────────────────
        cond_states = torch.cat([fast_cond, slow_cond], dim=-1)  # (B, T_pred, rnn_dim)

        x0_flat   = targets.reshape(B * T_pred, D)
        cond_flat = cond_states.reshape(B * T_pred, -1)

        t_diff = torch.randint(0, self.diff_steps, (B * T_pred,),
                               device=targets.device)
        xt, noise = self.scheduler.q_sample(x0_flat, t_diff)
        eps_pred  = self.eps_net(xt, t_diff, cond_flat)
        ddpm_loss = F.mse_loss(eps_pred, noise)

        if return_x0:
            ab    = self.scheduler.alpha_bar[t_diff].unsqueeze(-1)
            x0_hat = (xt - (1 - ab).sqrt() * eps_pred) / (ab.sqrt() + 1e-8)
            return ddpm_loss, x0_hat.reshape(B, T_pred, D)

        return ddpm_loss

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, slow_ctx: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        slow_ctx : (B, T_slow, 2)  normalized context (long window for slow LSTM)
        Returns  : (B, n_steps, 2) generated normalized returns
        """
        B, T_slow, D = slow_ctx.shape
        P      = self.slow_period
        device = next(self.parameters()).device

        # ── Initialize fast RNN with recent fast_ctx_len days ─────────
        fast_ctx = slow_ctx[:, -self.fast_ctx_len:, :]
        _, (h_fast, c_fast) = self.fast_rnn(fast_ctx)

        # ── Initialize slow RNN with full slow_ctx windows ────────────
        n_win_ctx = math.ceil(T_slow / P)
        pad = n_win_ctx * P - T_slow
        ctx_pad  = F.pad(slow_ctx, (0, 0, 0, pad)) if pad > 0 else slow_ctx
        win_mean = ctx_pad.reshape(B, n_win_ctx, P, D).mean(dim=2)
        _, (h_slow, c_slow) = self.slow_rnn(win_mean)

        generated  = []
        slow_buf   = []    # accumulate generated values for the next slow update

        for step in range(n_steps):
            cond = torch.cat([h_fast.squeeze(0), h_slow.squeeze(0)], dim=-1)  # (B, rnn_dim)

            # DDPM reverse: xᴺ → x⁰
            x = torch.randn(B, D, device=device)
            for diff_step in reversed(range(self.diff_steps)):
                t_vec = torch.full((B,), diff_step, device=device, dtype=torch.long)
                eps   = self.eps_net(x, t_vec, cond)
                x     = self.scheduler.p_sample_step(x, eps, diff_step)

            generated.append(x)
            slow_buf.append(x)

            # Update fast RNN every step
            _, (h_fast, c_fast) = self.fast_rnn(x.unsqueeze(1), (h_fast, c_fast))

            # Update slow RNN every P steps using mean of the generated window
            if len(slow_buf) == P:
                win = torch.stack(slow_buf, dim=1).mean(dim=1, keepdim=True)  # (B, 1, D)
                _, (h_slow, c_slow) = self.slow_rnn(win, (h_slow, c_slow))
                slow_buf = []

        return torch.stack(generated, dim=1)    # (B, n_steps, D)
