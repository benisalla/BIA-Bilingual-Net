import torch
from torch import nn
import torch.nn.functional as F
import math


class BIALinguaNet(nn.Module):
    def __init__(
        self,
        ev_size: int = 10000,
        dv_size: int = 10000,
        n_emb: int = 512,
        n_head: int = 8,
        h_size: int = 64,
        n_block: int = 6,
        exp_fac: int = 4,
        max_seq_len: int = 10000,
        d_rate: float = 0.0,
        device: str = 'cpu'
    ):
        super(BIALinguaNet, self).__init__()

        self.ev_size = ev_size
        self.dv_size = dv_size
        self.n_emb = n_emb
        self.n_head = n_head
        self.h_size = h_size
        self.n_block = n_block
        self.exp_fac = exp_fac
        self.max_seq_len = max_seq_len
        self.d_rate = d_rate
        self.device = device

        self.encoder = Encoder(
            ev_size=ev_size,
            n_emb=n_emb,
            n_head=n_head,
            h_size=h_size,
            max_seq_len=max_seq_len,
            n_block=n_block,
            exp_fac=exp_fac,
            d_rate=d_rate,
            device=device
        )

        self.decoder = Decoder(
            dv_size=dv_size,
            n_emb=n_emb,
            n_head=n_head,
            h_size=h_size,
            max_seq_len=max_seq_len,
            n_block=n_block,
            exp_fac=exp_fac,
            d_rate=d_rate,
            device=device
        )

        self.init_weights()

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.)

        nn.init.normal_(self.encoder.tok_emb.weight, mean=0., std=math.pow(self.n_emb, -0.5))
        self.decoder.tok_emb.weight = self.encoder.tok_emb.weight
        self.decoder.fc.weight = self.decoder.tok_emb.weight

        print("Our BIALinguaNet Model initialized ...")

    def forward(self, e_idx: torch.Tensor, d_idx: torch.Tensor, e_seq_len: torch.Tensor, d_seq_len: torch.Tensor) -> torch.Tensor:
        e_idx = self.encoder(e_idx, e_seq_len)
        d_idx = self.decoder(d_idx, d_seq_len, e_idx, e_seq_len)
        return d_idx
    
    def get_init_args(self):
        return {
            'ev_size': self.ev_size,
            'dv_size': self.dv_size,
            'n_emb': self.n_emb,
            'n_head': self.n_head,
            'h_size': self.h_size,
            'n_block': self.n_block,
            'exp_fac': self.exp_fac,
            'max_seq_len': self.max_seq_len,
            'd_rate': self.d_rate,
            'device': str(self.device)
        }
    
    @torch.no_grad()
    def translate(self, sx, tokenizer, temperature=1.0, beam_size=4, len_norm_coeff=0.6, is_ar=False, max_beam_fork=128):
        device = self.device
        self = self.to(device)
        self.eval()

        with torch.no_grad():
            k = beam_size          # Beam size
            n_hypo = min(k, 10)    # n of hypotheses
            vs = self.dv_size      # Vocab size

            if isinstance(sx, str):
                ex = tokenizer.encode(sx)
                ex = torch.LongTensor(ex).unsqueeze(0)
            else:
                ex = sx
            ex = ex.to(device)
            e_seq_len = torch.LongTensor([ex.size(1)]).to(device)

            # Encoder Forward
            ex = self.encoder(idx=ex, seq_len=e_seq_len)

            hypo = torch.LongTensor([[tokenizer.bos_token_id]]).to(device) # d_idx: <SOS>
            hypo_len = torch.LongTensor([hypo.size(1)]).to(device)         # d_seq_len: 1
            hypo_scores = torch.zeros(1).to(device)                        # 1 score

            com_hypo = list()
            com_hypo_scores = list()

            step = 1
            while True:
                s = hypo.size(0) # s
                logits = self.decoder(d_idx=hypo, d_seq_len=hypo_len, e_x=ex.repeat(s, 1, 1), e_seq_len=e_seq_len.repeat(s)) # [s, step, vs]
                scores = logits[:, -1, :] / temperature  # [s, vs]
                scores = F.log_softmax(scores, dim=-1)  # [s, vs]
                scores = hypo_scores.unsqueeze(1) + scores          # prev scores + curr scores
                
                top_k_hypo_scores, fttn_idx = scores.view(-1).topk(k, 0, True, True)  # top(vs) = k

                prev_tok_idx = fttn_idx // vs  # prev [k]
                next_tok_idx = fttn_idx  % vs  # next [k]

                top_k_hypo = torch.cat([hypo[prev_tok_idx], next_tok_idx.unsqueeze(1)], dim=1)  # [k, step + 1]

                complete = next_tok_idx == tokenizer.eos_token_id  # <EOS>? : [k], bool

                com_hypo.extend(top_k_hypo[complete].tolist())
                norm = math.pow(((5 + step) / (5 + 1)), len_norm_coeff) # chance(long sentence ~ short sentence)
                com_hypo_scores.extend((top_k_hypo_scores[complete] / norm).tolist())

                if len(com_hypo) >= n_hypo:
                    break # enough hypos

                hypo = top_k_hypo[~complete]                                           # [s, step + 1] incomplete hypos
                hypo_scores = top_k_hypo_scores[~complete]                             # d_idx(s): [s]
                hypo_len = torch.LongTensor(hypo.size(0) * [hypo.size(1)]).to(device)  # d_seq_len: [s]

                if step > max_beam_fork:
                    break # stop if no EOS is found
                step += 1

            if len(com_hypo) == 0: # in case no EOS is found
                com_hypo = hypo.tolist()
                com_hypo_scores = hypo_scores.tolist()

            all_hypos = list() # idx ==> string
            spec_toks = [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
            if is_ar:
                com_hypo = [tokenizer.decode([t for t in s if t not in spec_toks])[::-1] for s in com_hypo]
            else:
                com_hypo = [tokenizer.decode([t for t in s if t not in spec_toks]) for s in com_hypo]
                
            for i, h in enumerate(com_hypo):
                all_hypos.append({"hypothesis": h, "score": com_hypo_scores[i]})

            max_idx = com_hypo_scores.index(max(com_hypo_scores))
            best_hypo = all_hypos[max_idx]["hypothesis"]

            return best_hypo, all_hypos