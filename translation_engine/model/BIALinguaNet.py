import torch
from torch import nn
import torch.nn.functional as F
import math
from translation_engine.model.Decoder import Decoder
from translation_engine.model.Encoder import Encoder


class BIALinguaNet(nn.Module):
    """
    BIALinguaNet: A transformer-based architecture for bilingual language translation.

    This model consists of an encoder-decoder architecture, where the encoder processes the source language input,
    and the decoder generates the target language output. It supports various configurations for embedding size,
    number of attention heads, number of transformer blocks, and more.

    Attributes:
        ev_size (int): The size of the source language vocabulary.
        dv_size (int): The size of the target language vocabulary.
        n_emb (int): The dimension of the embedding vector.
        n_head (int): The number of attention heads in the multi-head attention mechanism.
        h_size (int): The hidden size of each attention head.
        n_block (int): The number of transformer blocks in both encoder and decoder.
        exp_fac (int): The expansion factor for the feedforward network.
        max_seq_len (int): The maximum sequence length that the model can process.
        d_rate (float): The dropout rate used for regularization.
        device (str): The device on which the computations will be performed ('cpu' or 'cuda').
    """

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
        device: str = "cpu",
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
        self.device = device if torch.cuda.is_available() else "cpu"

        self.encoder = Encoder(
            ev_size=ev_size,
            n_emb=n_emb,
            n_head=n_head,
            h_size=h_size,
            max_seq_len=max_seq_len,
            n_block=n_block,
            exp_fac=exp_fac,
            d_rate=d_rate,
            device=device,
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
            device=device,
        )

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize the weights of the model using Xavier uniform initialization for linear layers and
        normal initialization for embeddings. The decoder's embedding and output layers are tied to
        share weights with the encoder's embedding layer.
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        nn.init.normal_(
            self.encoder.tok_emb.weight, mean=0.0, std=math.pow(self.n_emb, -0.5)
        )
        self.decoder.tok_emb.weight = self.encoder.tok_emb.weight
        self.decoder.fc.weight = self.decoder.tok_emb.weight

    def forward(
        self,
        e_idx: torch.Tensor,
        d_idx: torch.Tensor,
        e_seq_len: torch.Tensor,
        d_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the BIALinguaNet model.

        Args:
            e_idx (torch.Tensor): Input tensor of source language token indices, shape (B, T), where B is the batch size and T is the sequence length.
            d_idx (torch.Tensor): Input tensor of target language token indices for teacher forcing, shape (B, T).
            e_seq_len (torch.Tensor): Tensor containing the lengths of each sequence in the source batch.
            d_seq_len (torch.Tensor): Tensor containing the lengths of each sequence in the target batch.

        Returns:
            torch.Tensor: The output tensor of shape (B, T, dv_size) representing the logits over the target vocabulary.
        """

        e_idx = self.encoder(e_idx, e_seq_len)
        d_idx = self.decoder(d_idx, d_seq_len, e_idx, e_seq_len)
        return d_idx

    def get_init_args(self):
        """
        Get the initial arguments used to configure the model.

        Returns:
            dict: A dictionary containing the model's initialization parameters.
        """

        return {
            "ev_size": self.ev_size,
            "dv_size": self.dv_size,
            "n_emb": self.n_emb,
            "n_head": self.n_head,
            "h_size": self.h_size,
            "n_block": self.n_block,
            "exp_fac": self.exp_fac,
            "max_seq_len": self.max_seq_len,
            "d_rate": self.d_rate,
            "device": str(self.device),
        }

    @torch.no_grad()
    def translate(
        self,
        sx,
        tokenizer,
        temperature: float = 0.0,
        beam_size: int = 4,
        len_norm_coeff: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        is_ltr: bool = False,
        max_beam_fork: int = 128,
    ):
        """
        Perform translation using beam search decoding.

        Args:
            sx (str or torch.Tensor): Source sentence or tensor to translate.
            tokenizer: Tokenizer object for encoding/decoding sentences.
            temperature (float): Temperature parameter for controlling randomness in predictions.
            beam_size (int): Number of beams to use in beam search.
            len_norm_coeff (float): Length normalization coefficient to balance sentence length during decoding.
            is_ltr (bool): Whether to reverse the output sequence (useful for autoregressive decoding).
            max_beam_fork (int): Maximum number of steps to consider in beam search.

        Returns:
            tuple: A tuple containing the best hypothesis (translated sentence) and all hypotheses with their scores.
        """

        device = self.device
        self = self.to(device)
        self.eval()

        with torch.no_grad():
            k = beam_size  # Beam size
            n_hypo = min(k, 10)  # n of hypotheses
            vs = self.dv_size  # Vocab size

            if isinstance(sx, str):
                ex = tokenizer.encode(sx)
                ex = torch.LongTensor(ex).unsqueeze(0)
            else:
                ex = sx
            ex = ex.to(device)
            e_seq_len = torch.LongTensor([ex.size(1)]).to(device)

            # Encoder Forward
            ex = self.encoder(idx=ex, seq_len=e_seq_len)

            hypo = torch.LongTensor([[tokenizer.bos_token_id]]).to(
                device
            )  # d_idx: <SOS>
            hypo_len = torch.LongTensor([hypo.size(1)]).to(device)  # d_seq_len: 1
            hypo_scores = torch.zeros(1).to(device)  # 1 score

            com_hypo = list()
            com_hypo_scores = list()

            step = 1
            while True:
                s = hypo.size(0)  # s
                logits = self.decoder(
                    d_idx=hypo,
                    d_seq_len=hypo_len,
                    e_x=ex.repeat(s, 1, 1),
                    e_seq_len=e_seq_len.repeat(s),
                )  # [s, step, vs]
                flogits = self.top_k_top_p_filtering(
                    logits=logits[:, -1, :], top_k=top_k, top_p=top_p
                )
                scores = flogits / max(temperature + 1.0, 1e-8)  # [s, vs]
                scores = F.log_softmax(scores, dim=-1)      # [s, vs]
                scores = hypo_scores.unsqueeze(1) + scores  # prev scores + curr scores

                top_k_hypo_scores, fttn_idx = scores.view(-1).topk(
                    k, 0, True, True
                )  # top(vs) = k

                prev_tok_idx = fttn_idx // vs  # prev [k]
                next_tok_idx = fttn_idx % vs   # next [k]

                top_k_hypo = torch.cat(
                    [hypo[prev_tok_idx], next_tok_idx.unsqueeze(1)], dim=1
                )  # [k, step + 1]

                complete = next_tok_idx == tokenizer.eos_token_id  # <EOS>? : [k], bool

                com_hypo.extend(top_k_hypo[complete].tolist())
                norm = math.pow(
                    ((5 + step) / (5 + 1)), len_norm_coeff
                )  # chance(long sentence ~ short sentence)
                com_hypo_scores.extend((top_k_hypo_scores[complete] / norm).tolist())

                if len(com_hypo) >= n_hypo:
                    break  # enough hypos

                hypo = top_k_hypo[~complete]  # [s, step + 1] incomplete hypos
                hypo_scores = top_k_hypo_scores[~complete]  # d_idx(s): [s]
                hypo_len = torch.LongTensor(hypo.size(0) * [hypo.size(1)]).to(
                    device
                )  # d_seq_len: [s]

                if step > max_beam_fork:
                    break  # stop if no EOS is found
                step += 1

            if len(com_hypo) == 0:  # in case no EOS is found
                com_hypo = hypo.tolist()
                com_hypo_scores = hypo_scores.tolist()

            all_hypos = list()  # idx ==> string
            spec_toks = [
                tokenizer.pad_token_id,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
            ]
            if is_ltr:
                com_hypo = [
                    tokenizer.decode([t for t in s if t not in spec_toks])[::-1]
                    for s in com_hypo
                ]
            else:
                com_hypo = [
                    tokenizer.decode([t for t in s if t not in spec_toks])
                    for s in com_hypo
                ]

            for i, h in enumerate(com_hypo):
                all_hypos.append({"hypothesis": h, "score": com_hypo_scores[i]})

            max_idx = com_hypo_scores.index(max(com_hypo_scores))
            best_hypo = all_hypos[max_idx]["hypothesis"]

            return best_hypo, all_hypos

    def top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        return logits
