import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResMHAtten(nn.Module):
    """
    Multi-Head Attention mechanism with a residual connection, similar to how CNNs are used in vision.
    This layer can be used in the encoder or decoder parts of a transformer model, supporting both
    self-attention and cross-attention functionalities.
    """
    def __init__(
        self,
        n_emb: int,
        n_head: int,
        h_size: int,
        d_rate: float,
        is_decoder: bool = False,
        device: str = 'cpu') -> None:
        super(ResMHAtten, self).__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.h_size = h_size
        self.is_decoder = is_decoder
        self.device = device

        self.q_proj = nn.Linear(n_emb, n_head * h_size)
        self.kv_proj = nn.Linear(n_emb, n_head * (h_size + h_size))
        self.o_proj = nn.Linear(n_head * h_size, n_emb)

        self.smax = nn.Softmax(dim=-1)
        self.ln = nn.LayerNorm(n_emb)
        self.drop = nn.Dropout(d_rate)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        B, qT, _ = q.shape
        _, kvT, _ = kv.shape

        # is self atten or cross atten
        is_self = torch.equal(kv, q)

        # fork for res con
        oq = q.clone()

        # pre-norm (diff than original tranfsormer)
        q = self.ln(q)
        kv = self.ln(kv) if is_self else kv

        # q, k, v --> qp, kp, vp    [B, xT, C]
        qp = self.q_proj(q)
        kp, vp = self.kv_proj(kv).split(split_size=self.n_head * self.h_size, dim=-1)

        # qp, kp, vp ==> (B, xT, nh, h)
        qp = qp.contiguous().view(B, qT, self.n_head, self.h_size)
        kp = kp.contiguous().view(B, kvT, self.n_head, self.h_size)
        vp = vp.contiguous().view(B, kvT, self.n_head, self.h_size)

        # [B, xT, h, nh] ==> [B * nh, xT, h]
        qp = qp.permute(0, 2, 1, 3).contiguous().view(-1, qT, self.h_size)
        kp = kp.permute(0, 2, 1, 3).contiguous().view(-1, kvT, self.h_size)
        vp = vp.permute(0, 2, 1, 3).contiguous().view(-1, kvT, self.h_size)

        # [B * nh, qT, h]   x   [B * nh, h, kvT]   ==>   [B * nh, qT, kvT]
        attn = torch.bmm(qp, kp.permute(0, 2, 1))
        attn = (1. / math.sqrt(self.h_size)) * attn # /sqrt(h)

        # pad mask
        valid_pos = torch.LongTensor(range(kvT)).unsqueeze(0).unsqueeze(0).expand_as(attn).to(self.device)
        mask = valid_pos < seq_len.repeat_interleave(self.n_head).unsqueeze(1).unsqueeze(2).expand_as(attn)
        attn = attn.masked_fill(~mask, -float('inf'))

        # self atten mask
        if self.is_decoder and is_self:
            hide_future_toks = torch.ones_like(attn).tril().bool().to(self.device)
            attn = attn.masked_fill(~hide_future_toks, -float('inf'))

        # softmax + dropout
        attn = self.drop(self.smax(attn))

        # [B * nh, qT, kvT]   x   [B * nh, kvT, h]    ==>    [B * nh, qT, h]
        attn = torch.bmm(attn, vp)                                                          # [B * nh, qT, h]
        attn = attn.contiguous().view(B, self.n_head, qT, self.h_size).permute(0, 2, 1, 3)  # [B, qT, nh, h]
        attn = attn.contiguous().view(B, qT, -1)                                            # [B, qT, nh * h]
        attn = self.o_proj(attn)                                                            # [B, qT, e]

        # dropout + res-conn
        out = self.drop(attn) + oq
        return out