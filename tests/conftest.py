"""Shared fixtures. Everything runs on CPU with tiny models and a fake tokenizer (no network)."""

import json

import pytest
import torch

from translation_engine.model.BIALinguaNet import BIALinguaNet

# the models here are tiny: thread start-up costs more than the maths (up to 10x slower)
torch.set_num_threads(1)

PAD, BOS, EOS = 0, 1, 2
WORDS = "salam labas ana bghit nmchi daba fin nta hello fine i want to go now where you are".split()


class FakeTokenizer:
    """Word-level tokenizer with the same interface the project uses from the GPT-2 tokenizer."""

    pad_token_id, bos_token_id, eos_token_id = PAD, BOS, EOS

    def __init__(self, words=WORDS):
        self.vocab = {w: i + 3 for i, w in enumerate(words)}
        self.inv = {i: w for w, i in self.vocab.items()}

    def __len__(self):
        return len(self.vocab) + 3

    def encode(self, text):
        return [self.vocab[w] for w in text.split()]

    def decode(self, ids):
        return " ".join(self.inv.get(int(i), "?") for i in ids)


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def make_model(vocab=24, n_emb=32, n_head=4, n_block=2, max_seq_len=16, d_rate=0.0, seed=0):
    torch.manual_seed(seed)
    return BIALinguaNet(
        ev_size=vocab,
        dv_size=vocab,
        n_emb=n_emb,
        n_head=n_head,
        h_size=n_emb // n_head,
        n_block=n_block,
        exp_fac=2,
        max_seq_len=max_seq_len,
        d_rate=d_rate,
        device="cpu",
    )


@pytest.fixture
def model():
    return make_model()


def random_batch(B=3, sT=6, tT=7, vocab=24, seed=0):
    """Random source/target ids with different lengths; padding (0) after each length."""
    g = torch.Generator().manual_seed(seed)
    s_len = torch.tensor([sT - i for i in range(B)])
    t_len = torch.tensor([tT - i for i in range(B)])
    sx = torch.randint(3, vocab, (B, sT), generator=g)
    tx = torch.randint(3, vocab, (B, tT), generator=g)
    tx[:, 0] = BOS
    for b in range(B):
        sx[b, s_len[b]:] = PAD
        tx[b, t_len[b] - 1] = EOS
        tx[b, t_len[b]:] = PAD
    return sx, tx, s_len, t_len


PAIRS = [
    {"dr": "salam", "en": "hello"},
    {"dr": "labas", "en": "fine"},
    {"dr": "ana bghit nmchi", "en": "i want to go"},
    {"dr": "fin nta", "en": "where are you"},
    {"dr": "bghit nmchi daba", "en": "i want to go now"},
    {"dr": "salam labas", "en": "hello fine"},
]


@pytest.fixture
def data_dir(tmp_path):
    """A folder with train/val/test json files in the project's format."""
    for split in ("train", "val", "test"):
        with open(tmp_path / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(PAIRS, f)
    return tmp_path
