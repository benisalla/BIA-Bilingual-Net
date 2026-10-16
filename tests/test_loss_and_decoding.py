"""Label-smoothing loss, top-k / top-p filtering and beam-search translation."""

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import BOS, EOS, PAD, PAIRS, FakeTokenizer, make_model
from translation_engine.model.LSRCrossEntropy import LSRCrossEntropy


# --- loss -------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.3])
def test_lsr_matches_pytorch_label_smoothing(eps):
    torch.manual_seed(0)
    x = torch.randn(3, 5, 11)
    y = torch.randint(0, 11, (3, 5))
    lens = torch.tensor([5, 3, 1])
    ours = LSRCrossEntropy(eps=eps)(x, y, lens)
    mask = torch.arange(5)[None] < lens[:, None]
    ref = F.cross_entropy(x[mask], y[mask], label_smoothing=eps)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_lsr_ignores_padding_positions():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 7)
    y = torch.randint(0, 7, (2, 4))
    lens = torch.tensor([4, 2])
    x2, y2 = x.clone(), y.clone()
    x2[1, 2:] = 100.0  # whatever the model says on padding...
    y2[1, 2:] = 3
    criterion = LSRCrossEntropy()
    assert torch.allclose(criterion(x, y, lens), criterion(x2, y2, lens))  # ...is not counted


# --- filtering --------------------------------------------------------------------


def test_top_k_keeps_k_highest(model):
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = model.top_k_top_p_filtering(logits, top_k=2)
    assert torch.isfinite(out).tolist() == [[False, True, False, True, False]]
    assert logits.tolist() == [[1.0, 5.0, 3.0, 4.0, 2.0]], "input must not be modified"


def test_top_p_keeps_smallest_nucleus(model):
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = model.top_k_top_p_filtering(probs.log(), top_p=0.7)
    assert torch.isfinite(out).tolist() == [[True, True, False, False]]  # 0.5 + 0.3 >= 0.7
    out = model.top_k_top_p_filtering(probs.log(), top_p=0.1)
    assert torch.isfinite(out).tolist() == [[True, False, False, False]]  # always keep the best


# --- translation --------------------------------------------------------------------


def train_copy_model(tokenizer, pairs, steps=300):
    """A tiny model trained to translate a few fixed sentences (dr -> en)."""
    model = make_model(vocab=len(tokenizer), n_emb=32, n_head=4, n_block=2, max_seq_len=16)
    src = [tokenizer.encode(p["dr"]) for p in pairs]
    tgt = [[BOS] + tokenizer.encode(p["en"]) + [EOS] for p in pairs]
    pad = torch.nn.utils.rnn.pad_sequence
    sx = pad([torch.tensor(s) for s in src], batch_first=True, padding_value=PAD)
    tx = pad([torch.tensor(t) for t in tgt], batch_first=True, padding_value=PAD)
    s_len = torch.tensor([len(s) for s in src])
    t_len = torch.tensor([len(t) for t in tgt])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = LSRCrossEntropy(eps=0.0)
    model.train()
    for _ in range(steps):
        loss = criterion(model(sx, tx, s_len, t_len), tx[:, 1:], t_len - 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


@pytest.fixture(scope="module")
def trained():
    tokenizer = FakeTokenizer()
    return train_copy_model(tokenizer, PAIRS), tokenizer, PAIRS


@pytest.mark.parametrize("beam_size", [1, 4])
def test_translate_recovers_memorised_sentences(trained, beam_size):
    model, tokenizer, pairs = trained
    for pair in pairs:
        best, hypos = model.translate(pair["dr"], tokenizer, beam_size=beam_size, top_k=0, top_p=1.0)
        assert best == pair["en"]
        assert all(math.isfinite(h["score"]) for h in hypos)


def test_translate_with_tight_top_p_has_no_garbage_beams(trained):
    """With top_p small only ~1 token per beam survives: the beam must not be filled with -inf hypotheses."""
    model, tokenizer, pairs = trained
    best, hypos = model.translate(pairs[2]["dr"], tokenizer, beam_size=4, top_k=50, top_p=0.05)
    assert best == pairs[2]["en"]
    assert hypos and all(math.isfinite(h["score"]) for h in hypos)


def test_translate_never_exceeds_max_seq_len(tokenizer):
    model = make_model(vocab=len(tokenizer), max_seq_len=8)  # untrained: may never emit EOS
    long_input = " ".join(["salam"] * 30)  # longer than max_seq_len
    best, hypos = model.translate(long_input, tokenizer, beam_size=3, max_beam_fork=500, top_k=0, top_p=1.0)
    assert isinstance(best, str) and hypos


def test_translate_empty_input(model, tokenizer):
    assert model.translate("", tokenizer) == ("", [])


@pytest.mark.parametrize("temperature", [0.5, 1.0, 3.0])
def test_temperature_keeps_the_best_translation_on_a_confident_model(trained, temperature):
    model, tokenizer, pairs = trained
    best, _ = model.translate(pairs[3]["dr"], tokenizer, temperature=temperature, top_k=0, top_p=1.0)
    assert best == pairs[3]["en"]
