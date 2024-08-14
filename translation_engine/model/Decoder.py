import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from translation_engine.model.ResFFNet import ResFFNet
from translation_engine.model.ResMHAtten import ResMHAtten

class Decoder(nn.Module):
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
        device: str = 'cpu'):
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

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ResMHAtten(n_emb=n_emb, n_head=n_head, h_size=h_size, d_rate=d_rate, is_decoder=True, device=device),
                ResMHAtten(n_emb=n_emb, n_head=n_head, h_size=h_size, d_rate=d_rate, is_decoder=True, device=device),
                ResFFNet(n_emb=n_emb, exp_fac=exp_fac, d_rate=d_rate)
            ]) for _ in range(n_block)
        ])

        self.drop = nn.Dropout(d_rate)
        self.ln = nn.LayerNorm(n_emb)
        self.fc = nn.Linear(n_emb, dv_size)

    def forward(self, d_idx: torch.Tensor, d_seq_len: torch.Tensor, e_x: torch.Tensor, e_seq_len: torch.Tensor) -> torch.Tensor:
        _, dT = d_idx.shape

        if dT > self.max_seq_len:
            raise ValueError(f"Sequence length dT = {dT} exceeds maximum allowed seq_len = {self.max_seq_len}")

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
