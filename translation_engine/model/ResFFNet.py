import torch
from torch import nn


class ResFFNet(nn.Module):
    """
    Residual Feedforward Network (ResFFNet): This module serves as a reasoning block, using a residual connection strategy
    to stabilize and enhance learning, particularly in deep neural networks.

    Attributes:
        n_emb (int): The embedding dimension of the input features.
        exp_fac (int): The expansion factor for the hidden layer, typically a multiple of the embedding dimension.
        d_rate (float): The dropout rate used to prevent overfitting.
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
        """
        Forward pass of the ResFFNet module.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, n_emb), where B is the batch size, T is the sequence length,
                            and n_emb is the embedding dimension.

        Returns:
            torch.Tensor: The output tensor after applying the feedforward network and residual connection,
                            maintaining the input shape (B, T, n_emb).
        """

        o_x = x.clone()  # fork
        x = self.ln(x)  # norm

        # feedforward
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)

        # dropout and res conn
        x = self.drop(x) + o_x
        return x
