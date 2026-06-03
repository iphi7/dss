"""
model.py – Sign Ablation TimeGrad

4つのモード:
  cond_random : ランダム sign window → EpsilonNet 条件
  lstm_random : ランダム sign window → LSTM 入力
  cond_oracle : 真の sign window    → EpsilonNet 条件  (学習時のみ)
  lstm_oracle : 真の sign window    → LSTM 入力       (学習時のみ)

生成時は全モードでランダム Ber(1/2) sign window を使う。
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
    def __init__(self, data_dim: int, hidden_dim: int, cond_dim: int,
                 n_layers: int, step_emb_dim: int, dropout: float):
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


class SignAblationTimeGrad(nn.Module):
    """
    sign_mode:
      "cond_*" → sign_window を EpsilonNet の条件に追加
      "lstm_*" → sign_window を LSTM の入力に追加
      "*_random" → 学習時もランダム sign window
      "*_oracle" → 学習時は真の sign window、生成時はランダム

    sign_window: 直近何日分の sign を使うか（デフォルト 30）
    sign_dim = sign_window * data_dim = 30 * 2 = 60
    """

    def __init__(self, sign_mode: str, data_dim: int = 2,
                 sign_window: int = 30, rnn_hidden: int = 64,
                 hidden_dim: int = 64, n_layers: int = 4,
                 step_emb_dim: int = 64, diff_steps: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        self.sign_mode   = sign_mode
        self.data_dim    = data_dim
        self.diff_steps  = diff_steps
        self.sign_dim    = sign_window * data_dim   # 60

        use_lstm_sign = "lstm" in sign_mode
        use_cond_sign = "cond" in sign_mode

        rnn_in   = data_dim + (self.sign_dim if use_lstm_sign else 0)
        eps_cond = rnn_hidden + (self.sign_dim if use_cond_sign else 0)

        self.rnn      = nn.LSTM(rnn_in, rnn_hidden, batch_first=True)
        self.rnn_drop = nn.Dropout(dropout)
        self.eps_net  = EpsilonNet(data_dim, hidden_dim, eps_cond,
                                    n_layers, step_emb_dim, dropout)
        self.scheduler = DDPMScheduler(diff_steps)

    def _rand_sw(self, shape, device) -> torch.Tensor:
        """Ber(1/2) ランダム sign window ∈ {-1, +1}"""
        return torch.randint(0, 2, shape, device=device).float() * 2 - 1

    def _get_sw(self, sw_oracle: torch.Tensor, force_random: bool = False) -> torch.Tensor:
        """oracle or random sign window を返す"""
        if force_random or "random" in self.sign_mode:
            return self._rand_sw(sw_oracle.shape, sw_oracle.device)
        return sw_oracle  # oracle

    def _lstm_input(self, r_seq: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
        """lstm モード: [r, sw] 結合。cond モード: r のみ"""
        if "lstm" in self.sign_mode:
            return torch.cat([r_seq, sw], dim=-1)
        return r_seq

    def _eps_cond(self, h: torch.Tensor, sw_flat: torch.Tensor) -> torch.Tensor:
        """cond モード: [h, sw] 結合。lstm モード: h のみ"""
        if "cond" in self.sign_mode:
            return torch.cat([h, sw_flat], dim=-1)
        return h

    def compute_loss(self, ctx: torch.Tensor, tgt: torch.Tensor,
                     ctx_sw: torch.Tensor, tgt_sw: torch.Tensor) -> torch.Tensor:
        """
        ctx    : (B, T_ctx, 2)       正規化済み生リターン
        tgt    : (B, T_pred, 2)      正規化済み未来リターン
        ctx_sw : (B, T_ctx, 60)      Oracle sign window（random variant では内部でランダム置換）
        tgt_sw : (B, T_pred, 60)     同上
        """
        B, T_pred, D = tgt.shape

        ctx_sw = self._get_sw(ctx_sw)
        tgt_sw = self._get_sw(tgt_sw)

        # コンテキスト → LSTM
        ctx_in = self._lstm_input(ctx, ctx_sw)
        _, (h, c) = self.rnn(ctx_in)
        h_drop = self.rnn_drop(h)

        # 教師強制
        if T_pred > 1:
            tgt_in  = self._lstm_input(tgt[:, :-1, :], tgt_sw[:, :-1, :])
            out, _  = self.rnn(tgt_in, (h, c))
            out     = self.rnn_drop(out)
            h0      = h_drop.squeeze(0).unsqueeze(1)
            h_seq   = torch.cat([h0, out], dim=1)    # (B, T_pred, H)
        else:
            h_seq   = h_drop.squeeze(0).unsqueeze(1)

        # EpsilonNet 条件
        cond_seq = self._eps_cond(h_seq, tgt_sw)     # (B, T_pred, H[+60])

        BT    = B * T_pred
        x0_f  = tgt.reshape(BT, D)
        cond_f = cond_seq.reshape(BT, -1)

        t_diff = torch.randint(0, self.diff_steps, (BT,), device=tgt.device)
        xt, noise = self.scheduler.q_sample(x0_f, t_diff)
        eps_pred  = self.eps_net(xt, t_diff, cond_f)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def generate(self, ctx: torch.Tensor, ctx_sw: torch.Tensor,
                 n_steps: int) -> torch.Tensor:
        """
        生成時は常にランダム sign window を使う（全 variant 共通）。

        ctx    : (B, T_ctx, 2)
        ctx_sw : (B, T_ctx, 60)  ※生成時は無視してランダム置換される
        """
        B   = ctx.shape[0]
        dev = next(self.parameters()).device

        ctx_sw_rand = self._rand_sw(ctx_sw.shape, dev)
        ctx_in = self._lstm_input(ctx, ctx_sw_rand)
        _, (h, c) = self.rnn(ctx_in)

        try:
            from tqdm import tqdm
            steps_iter = tqdm(range(n_steps), desc="generating", unit="day", leave=False)
        except ImportError:
            steps_iter = range(n_steps)

        generated = []
        for _ in steps_iter:
            sw_t = self._rand_sw((B, self.sign_dim), dev)    # 毎ステップ新しいランダム sign
            cond = self._eps_cond(self.rnn_drop(h.squeeze(0)), sw_t)  # (B, H[+60])

            x = torch.randn(B, self.data_dim, device=dev)
            for step in reversed(range(self.diff_steps)):
                t_vec = torch.full((B,), step, device=dev, dtype=torch.long)
                eps   = self.eps_net(x, t_vec, cond)
                x     = self.scheduler.p_sample_step(x, eps, step)

            generated.append(x)
            lstm_in = self._lstm_input(x.unsqueeze(1),
                                        sw_t.unsqueeze(1))   # (B, 1, 2[+60])
            _, (h, c) = self.rnn(lstm_in, (h, c))

        return torch.stack(generated, dim=1)   # (B, n_steps, 2)
