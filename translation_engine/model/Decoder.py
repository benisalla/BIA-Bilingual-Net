import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from translation_engine.model.ResFFNet import ResFFNet
from translation_engine.model.ResMHAtten import ResMHAtten


class Decoder(nn.Module):
    """
    Decoder module for a transformer model, which combines token embeddings, positional embeddings, and a series of
    self-attention, cross-attention, and feedforward blocks with residual connections.

    This decoder is suitable for various sequence generation tasks, providing a flexible architecture that can be
    customized with different numbers of layers, attention heads, and feedforward expansion factors.

    Attributes:
        dv_size (int): The size of the target vocabulary.
        n_emb (int): The dimension of the embedding vector.
        n_head (int): The number of attention heads in the multi-head attention mechanism.
        h_size (int): The hidden size of each attention head.
        n_block (int): The number of blocks, each containing self-attention, cross-attention, and feedforward sub-layers.
        exp_fac (int): The expansion factor for the feedforward network.
        d_rate (float): The dropout rate used for regularization.
        device (str): The device on which the computations will be performed ('cpu' or 'cuda').
        max_seq_len (int): The maximum sequence length that the decoder can process.
    """

    def __init__(
        self,
        dv_size: int,
        n_emb: int,
        n_head: int,
        h_size: int,
        max_seq_len: int = 10000,
        n_block: int = 1,
        exp_fac: int = 4,
        d_rate: float = 0.0,
        device: str = "cpu",
    ):
        super(Decoder, self).__init__()

        self.dv_size = dv_size
        self.n_emb = n_emb
        self.n_head = n_head
        self.h_size = h_size
        self.n_block = n_block
        self.d_rate = d_rate
        self.exp_fac = exp_fac
        self.device = device
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(dv_size, n_emb)
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
                            is_decoder=True,
                            device=device,
                        ),
                        ResMHAtten(
                            n_emb=n_emb,
                            n_head=n_head,
                            h_size=h_size,
                            d_rate=d_rate,
                            is_decoder=True,
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
        self.fc = nn.Linear(n_emb, dv_size)

    def forward(
        self,
        d_idx: torch.Tensor,
        d_seq_len: torch.Tensor,
        e_x: torch.Tensor,
        e_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            d_idx (torch.Tensor): Input tensor of target token indices, shape (B, T), where B is the batch size and T is the sequence length.
            d_seq_len (torch.Tensor): Tensor containing the lengths of each sequence in the batch for the decoder, used for masking.
            e_x (torch.Tensor): Encoded input tensor from the encoder, shape (B, T_enc, n_emb).
            e_seq_len (torch.Tensor): Tensor containing the lengths of each sequence in the batch for the encoder, used for masking.

        Returns:
            torch.Tensor: The output tensor of shape (B, T, dv_size) representing the logits over the vocabulary for each time step.

        Raises:
            ValueError: If the input sequence length T exceeds the maximum allowed sequence length `max_seq_len`.
        """

        _, dT = d_idx.shape

        if dT > self.max_seq_len:
            raise ValueError(
                f"Sequence length dT = {dT} exceeds maximum allowed seq_len = {self.max_seq_len}"
            )

        pos = torch.arange(dT, dtype=torch.long, device=self.device)
        tok_emb = self.tok_emb(d_idx) * math.sqrt(self.n_emb)
        pos_emb = self.pos_emb(pos)

        d_idx = self.drop(tok_emb + pos_emb)

        for self_attn, cross_attn, ffnet in self.blocks:
            d_idx = self_attn(q=d_idx, kv=d_idx, seq_len=d_seq_len)
            d_idx = cross_attn(q=d_idx, kv=e_x, seq_len=e_seq_len)
            d_idx = ffnet(d_idx)

        d_idx = self.ln(d_idx)
        d_idx = self.fc(d_idx)
        return d_idx
