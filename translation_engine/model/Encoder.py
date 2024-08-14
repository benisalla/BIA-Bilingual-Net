import math
import torch
from torch import nn
from translation_engine.model.ResFFNet import ResFFNet
from translation_engine.model.ResMHAtten import ResMHAtten


class Encoder(nn.Module):
    """
    Encoder module for a transformer model, combining token embeddings, positional embeddings, and a series of
    multi-head attention and feedforward blocks with residual connections.

    This encoder is suitable for various sequence modeling tasks, providing a flexible architecture that can be
    customized with different numbers of layers, attention heads, and feedforward expansion factors.

    Attributes:
        ev_size (int): The size of the vocabulary or embedding vector.
        n_emb (int): The dimension of the embedding vector.
        n_head (int): The number of attention heads in the multi-head attention mechanism.
        h_size (int): The hidden size of each attention head.
        n_block (int): The number of blocks, each containing a multi-head attention and feedforward sub-layer.
        exp_fac (int): The expansion factor for the feedforward network.
        d_rate (float): The dropout rate used for regularization.
        device (str): The device on which the computations will be performed ('cpu' or 'cuda').
        max_seq_len (int): The maximum sequence length that the encoder can process.
    """

    def __init__(
        self,
        ev_size: int,
        n_emb: int,
        n_head: int,
        h_size: int,
        max_seq_len: int = 10000,
        n_block: int = 1,
        exp_fac: int = 4,
        d_rate: float = 0.0,
        device: str = "cpu",
    ):
        super(Encoder, self).__init__()

        self.ev_size = ev_size
        self.n_emb = n_emb
        self.n_head = n_head
        self.h_size = h_size
        self.n_block = n_block
        self.d_rate = d_rate
        self.exp_fac = exp_fac
        self.device = device
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(ev_size, n_emb)
        self.pos_emb = nn.Embedding(max_seq_len, n_emb)

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResMHAtten(
                            n_emb=n_emb,
                            n_head=n_head,
                            h_size=h_size,
                            d_rate=d_rate,
                            is_decoder=False,
                            device=device,
                        ),
                        ResFFNet(n_emb=n_emb, exp_fac=exp_fac, d_rate=d_rate),
                    ]
                )
                for _ in range(n_block)
            ]
        )

        self.drop = nn.Dropout(d_rate)
        self.ln = nn.LayerNorm(n_emb)

    def forward(self, idx: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder module.

        Args:
            idx (torch.Tensor): Input tensor of token indices, shape (B, T), where B is the batch size and T is the sequence length.
            seq_len (torch.Tensor): Tensor containing the lengths of each sequence in the batch, used for masking.

        Returns:
            torch.Tensor: The encoded output tensor of shape (B, T, n_emb).

        Raises:
            ValueError: If the input sequence length T exceeds the maximum allowed sequence length `max_seq_len`.
        """

        B, T = idx.size()

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T = {T} exceeds maximum allowed seq_len = {self.max_seq_len}"
            )

        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        tok_emb = self.tok_emb(idx) * math.sqrt(self.n_emb)
        pos_emb = self.pos_emb(pos)

        x = self.drop(tok_emb + pos_emb)

        for attn, ffnet in self.blocks:
            x = attn(q=x, kv=x, seq_len=seq_len)
            x = ffnet(x)

        x = self.ln(x)
        return x
