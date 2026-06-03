"""
model.py – AR-FNN Generator (Conditional AR-FNN)

論文 (Ni et al. 2023) Section 5.3 の "Conditional AR-FNN" を実装。
X_{t+1} = G(X_{t-p̄+1:t}, Z_t)  where Z_t ~ N(0, I)

Structure (Appendix B.2):
  input  : [X_{t-p̄+1:t}.flatten(), Z_t]  ← past p̄ steps + noise
  hidden : 3-layer FNN with residual connections and LeakyReLU
  output : X_{t+1} ∈ R^d

Generation:
  - 最初の p̄ ステップは X_past そのものを使用
  - t > p̄ では直前の p̄ ステップの生成値を入力として再帰的に生成
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ARFNNGenerator(nn.Module):
    """
    G(X_past_window, Z) → X_{next}

    X_past_window : (..., p_bar * data_dim)  flattened past window
    Z             : (..., noise_dim)

    Returns       : (..., data_dim)
    """

    def __init__(self, data_dim: int = 2, p_bar: int = 5,
                 noise_dim: int = 5, hidden_dim: int = 64,
                 n_layers: int = 3):
        super().__init__()
        self.data_dim  = data_dim
        self.p_bar     = p_bar
        self.noise_dim = noise_dim

        in_dim = p_bar * data_dim + noise_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x_window: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
        x_window : (B, p_bar, data_dim) or (B, p_bar * data_dim)
        z        : (B, noise_dim)
        Returns  : (B, data_dim)
        """
        if x_window.dim() == 3:
            x_window = x_window.reshape(x_window.shape[0], -1)
        h = torch.cat([x_window, z], dim=-1)
        h = self.input_proj(h)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)

    def generate_sequence(self, x_past: torch.Tensor,
                          q_bar: int) -> torch.Tensor:
        """
        Autoregressively generate q_bar steps given x_past.

        x_past   : (B, p_bar, data_dim)  context window (already normalised)
        q_bar    : number of steps to generate
        Returns  : (B, q_bar, data_dim)  generated sequence
        """
        B = x_past.shape[0]
        device = x_past.device

        # Running buffer: starts with the last p_bar steps of x_past
        buf = x_past.clone()          # (B, p_bar, data_dim)
        generated = []

        for _ in range(q_bar):
            z = torch.randn(B, self.noise_dim, device=device)
            x_next = self(buf, z)     # (B, data_dim)
            generated.append(x_next.unsqueeze(1))  # (B, 1, data_dim)
            # Slide window
            buf = torch.cat([buf[:, 1:, :], x_next.unsqueeze(1)], dim=1)

        return torch.cat(generated, dim=1)  # (B, q_bar, data_dim)
