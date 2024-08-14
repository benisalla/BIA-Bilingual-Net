import math
import torch
from torch import nn
from translation_engine.model.ResFFNet import ResFFNet
from translation_engine.model.ResMHAtten import ResMHAtten

class Encoder(nn.Module):
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
        device: str = 'cpu'):
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

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ResMHAtten(n_emb=n_emb, n_head=n_head, h_size=h_size, d_rate=d_rate, is_decoder=False, device=device),
                ResFFNet(n_emb=n_emb, exp_fac=exp_fac, d_rate=d_rate)
            ]) for _ in range(n_block)
        ])

        self.drop = nn.Dropout(d_rate)
        self.ln = nn.LayerNorm(n_emb)

    def forward(self, idx: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length T = {T} exceeds maximum allowed seq_len = {self.max_seq_len}")

        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        tok_emb = self.tok_emb(idx) * math.sqrt(self.n_emb)
        pos_emb = self.pos_emb(pos)

        x = self.drop(tok_emb + pos_emb)

        for attn, ffnet in self.blocks:
            x = attn(q=x, kv=x, seq_len=seq_len)
            x = ffnet(x)

        x = self.ln(x)
        return x