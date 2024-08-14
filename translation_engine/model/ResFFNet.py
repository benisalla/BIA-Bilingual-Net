import torch
from torch import nn

class ResFFNet(nn.Module):
    """
    Residual Feedforward Network: This module acts as a reasoning block, employing a residual connection strategy to stabilize and improve learning.
    """
    def __init__(self, n_emb: int, exp_fac: int = 4, d_rate: float = 0.0) -> None:
        super(ResFFNet, self).__init__()
        self.n_emb = n_emb
        self.exp_fac = exp_fac
        self.d_rate = d_rate

        self.ln = nn.LayerNorm(self.n_emb)
        self.fc1 = nn.Linear(self.n_emb, exp_fac * self.n_emb)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(exp_fac * self.n_emb, self.n_emb)
        self.drop = nn.Dropout(d_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o_x = x.clone() # fork
        x = self.ln(x)  # norm

        # feedforward
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)

        # dropout and res conn
        x = self.drop(x) + o_x
        return x