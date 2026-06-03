"""
model.py – TimeGrad + SFAG-style stylized fact alignment loss

x_t^0 : 2D return vector [sp500_ret, dgs10_change] at time step t

Architecture is identical to base TimeGrad (LSTM + DDPM) but uses a
smaller EpsilonNet (4 layers, hidden=64) appropriate for the 2D output
space.  Stylized fact alignment is enforced through the training loss
(sf_loss.py), not via conditioning inputs.

Reference for SFAG loss:
  Zhang et al. "Beyond Visual Realism: Toward Reliable Financial
  Time Series Generation" (arXiv:2601.12990, 2026)
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
    """FiLM-conditioned residual block."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.linear1(x))
        scale, shift = self.cond_proj(F.silu(cond)).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = F.silu(self.linear2(h))
        return self.norm(x + h)


class EpsilonNet(nn.Module):
    """
    Noise predictor ε_θ(x^n, n, h_t).
    Input : x^n ∈ R^2 (noisy return), n (diffusion step), h_t (RNN hidden)
    Output: ε̂ ∈ R^2

    Smaller than base timegrad (4 layers, hidden=64) — sufficient for a
    2D→2D mapping; the original 8-layer/128-hidden version was over-parameterised.
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 64,
                 rnn_hidden: int = 64, n_layers: int = 4,
                 step_emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.step_emb = SinusoidalEmbedding(step_emb_dim)
        cond_dim = step_emb_dim + rnn_hidden
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                rnn_h: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.step_emb(t), rnn_h], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = self.dropout(block(h, cond))
        return self.output_proj(h)


class DDPMScheduler(nn.Module):
    """DDPM noise schedule stored as buffers."""

    def __init__(self, n_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        a = self.alpha_bar[t].unsqueeze(-1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    def p_sample_step(self, x: torch.Tensor, eps_pred: torch.Tensor,
                      step: int) -> torch.Tensor:
        a = self.alphas[step]
        ab = self.alpha_bar[step]
        coeff = (1 - a) / (1 - ab).sqrt()
        x = (1 / a.sqrt()) * (x - coeff * eps_pred)
        if step > 0:
            x = x + self.betas[step].sqrt() * torch.randn_like(x)
        return x


class TimeGradSFModel(nn.Module):
    """
    TimeGrad with SFAG-style stylized fact alignment.

    Architecture is the same as base TimeGrad:
      1. RNN encodes history x_{1:t-1} → h_t
      2. DDPM generates x_t^0 conditioned on h_t
      3. x_t^0 fed back to RNN for t+1

    The stylized fact alignment is in the *loss*, not in the conditioning.
    See sf_loss.py and the training loop in train.py.
    """

    def __init__(self, data_dim: int = 2, rnn_hidden: int = 64,
                 hidden_dim: int = 64, n_layers: int = 4,
                 step_emb_dim: int = 64, diff_steps: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        self.data_dim = data_dim
        self.diff_steps = diff_steps
        self.rnn = nn.LSTM(data_dim, rnn_hidden, batch_first=True)
        self.rnn_dropout = nn.Dropout(dropout)
        self.eps_net = EpsilonNet(data_dim, hidden_dim, rnn_hidden,
                                  n_layers, step_emb_dim, dropout)
        self.scheduler = DDPMScheduler(diff_steps)

    def compute_loss(self, ctx: torch.Tensor, targets: torch.Tensor,
                     return_x0: bool = False):
        """
        ctx       : (B, T_ctx, 2)   normalized historical returns
        targets   : (B, T_pred, 2)  normalized future returns
        return_x0 : if True, also return Tweedie x̂0 = (xⁿ − √(1−ᾱₙ)·ε̂) / √ᾱₙ
                    shaped (B, T_pred, 2), for use in the SF alignment loss
        """
        B, T_pred, D = targets.shape

        _, (h, c) = self.rnn(ctx)
        h_drop = self.rnn_dropout(h)

        if T_pred > 1:
            tgt_out, _ = self.rnn(targets[:, :-1, :], (h, c))
            tgt_out = self.rnn_dropout(tgt_out)
            cond_states = torch.cat(
                [h_drop.squeeze(0).unsqueeze(1), tgt_out], dim=1)
        else:
            cond_states = h_drop.squeeze(0).unsqueeze(1)

        x0_flat   = targets.reshape(B * T_pred, D)
        cond_flat = cond_states.reshape(B * T_pred, -1)

        t_diff = torch.randint(0, self.diff_steps, (B * T_pred,),
                               device=targets.device)
        xt, noise = self.scheduler.q_sample(x0_flat, t_diff)
        eps_pred = self.eps_net(xt, t_diff, cond_flat)
        ddpm_loss = F.mse_loss(eps_pred, noise)

        if return_x0:
            ab = self.scheduler.alpha_bar[t_diff].unsqueeze(-1)
            x0_hat = (xt - (1 - ab).sqrt() * eps_pred) / (ab.sqrt() + 1e-8)
            x0_hat = x0_hat.reshape(B, T_pred, D)
            return ddpm_loss, x0_hat

        return ddpm_loss

    @torch.no_grad()
    def generate(self, ctx: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        ctx    : (B, T_ctx, 2)  normalized context
        Returns: (B, n_steps, 2) generated normalized returns
        """
        B, D = ctx.shape[0], self.data_dim
        device = next(self.parameters()).device

        _, (h, c) = self.rnn(ctx)

        generated = []
        for _ in range(n_steps):
            cond = h.squeeze(0)

            x = torch.randn(B, D, device=device)
            for step in reversed(range(self.diff_steps)):
                t_vec = torch.full((B,), step, device=device, dtype=torch.long)
                eps = self.eps_net(x, t_vec, cond)
                x = self.scheduler.p_sample_step(x, eps, step)

            generated.append(x)
            _, (h, c) = self.rnn(x.unsqueeze(1), (h, c))

        return torch.stack(generated, dim=1)
