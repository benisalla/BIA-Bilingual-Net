"""Neural-network sanity checks for BIALinguaNet (CPU, tiny model)."""

import math

import pytest
import torch

from conftest import PAD, make_model, random_batch
from translation_engine.model.LSRCrossEntropy import LSRCrossEntropy
from translation_engine.model.ResMHAtten import ResMHAtten


def test_output_shape(model):
    sx, tx, s_len, t_len = random_batch()
    logits = model(sx, tx, s_len, t_len)
    assert logits.shape == (3, 7, 24)
    assert torch.isfinite(logits).all()


def test_initial_loss_is_close_to_uniform_guess():
    """At init the model should know nothing: loss ~ ln(vocab), not much higher."""
    vocab = 500
    model = make_model(vocab=vocab, n_emb=64)
    sx, tx, s_len, t_len = random_batch(B=8, sT=10, tT=12, vocab=vocab)
    loss = LSRCrossEntropy(eps=0.0)(model(sx, tx, s_len, t_len), tx[:, 1:], t_len - 1)
    assert abs(loss.item() - math.log(vocab)) < 1.0


def test_every_parameter_gets_a_gradient(model):
    sx, tx, s_len, t_len = random_batch()
    loss = LSRCrossEntropy()(model(sx, tx, s_len, t_len), tx[:, 1:], t_len - 1)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name


def test_decoder_is_causal(model):
    """Changing target token t must not change the predictions made before t."""
    model.eval()
    sx, tx, s_len, t_len = random_batch(B=1, tT=7)
    base = model(sx, tx, s_len, t_len)
    for t in range(1, 7):
        changed = tx.clone()
        changed[0, t] = (changed[0, t] + 1) % 24
        out = model(sx, changed, s_len, t_len)
        assert torch.allclose(out[:, :t], base[:, :t], atol=1e-5), f"position < {t} saw token {t}"
        assert not torch.allclose(out[:, t:], base[:, t:])


def test_source_padding_is_ignored(model):
    model.eval()
    sx, tx, s_len, t_len = random_batch(B=1)
    padded = torch.cat([sx, torch.full((1, 5), PAD)], dim=1)
    padded[0, s_len[0]:] = 7  # garbage after the real length must not matter either
    assert torch.allclose(model(sx, tx, s_len, t_len), model(padded, tx, s_len, t_len), atol=1e-5)


def test_target_padding_is_ignored(model):
    model.eval()
    sx, tx, s_len, t_len = random_batch(B=1)
    padded = torch.cat([tx, torch.full((1, 4), 9)], dim=1)
    out = model(sx, padded, s_len, t_len)
    assert torch.allclose(model(sx, tx, s_len, t_len), out[:, : tx.size(1)], atol=1e-5)


def test_samples_in_a_batch_do_not_interact(model):
    model.eval()
    sx, tx, s_len, t_len = random_batch(B=3)
    batched = model(sx, tx, s_len, t_len)
    for b in range(3):
        alone = model(
            sx[b : b + 1, : s_len[b]], tx[b : b + 1, : t_len[b]], s_len[b : b + 1], t_len[b : b + 1]
        )
        assert torch.allclose(batched[b : b + 1, : t_len[b]], alone, atol=1e-5)


def test_dropout_only_in_train_mode():
    model = make_model(d_rate=0.3)
    sx, tx, s_len, t_len = random_batch()
    model.eval()
    assert torch.equal(model(sx, tx, s_len, t_len), model(sx, tx, s_len, t_len))
    model.train()
    assert not torch.equal(model(sx, tx, s_len, t_len), model(sx, tx, s_len, t_len))


def test_embeddings_and_output_layer_are_tied(model):
    assert model.encoder.tok_emb.weight is model.decoder.tok_emb.weight
    assert model.decoder.fc.weight is model.decoder.tok_emb.weight


def test_explicit_is_self_matches_implicit():
    torch.manual_seed(0)
    attn = ResMHAtten(n_emb=16, n_head=2, h_size=8, d_rate=0.0, is_decoder=True)
    x = torch.randn(2, 5, 16)
    seq_len = torch.tensor([5, 3])
    assert torch.equal(attn(q=x, kv=x, seq_len=seq_len), attn(q=x, kv=x, seq_len=seq_len, is_self=True))
    # cross-attention (different tensor) is not causal: the last query sees every key
    kv = torch.randn(2, 5, 16)
    out = attn(q=x, kv=kv, seq_len=seq_len)
    kv2 = kv.clone()
    kv2[0, 4] += 1.0
    assert not torch.allclose(attn(q=x, kv=kv2, seq_len=seq_len)[0, 0], out[0, 0])


def test_model_built_for_cuda_runs_on_cpu():
    """Checkpoints saved on a GPU store device='cuda'; they must still work on a CPU machine."""
    model = make_model()
    model_cuda_args = type(model)(**{**model.get_init_args(), "device": "cuda"})
    model_cuda_args.load_state_dict(model.state_dict())
    sx, tx, s_len, t_len = random_batch()
    model.eval(), model_cuda_args.eval()
    assert torch.allclose(model(sx, tx, s_len, t_len), model_cuda_args(sx, tx, s_len, t_len))


def test_too_long_input_raises(model):
    sx = torch.randint(3, 24, (1, 17))
    with pytest.raises(ValueError):
        model.encoder(sx, torch.tensor([17]))


def test_overfits_one_batch():
    """The classic sanity check: a small model must be able to memorise a single batch."""
    model = make_model(d_rate=0.0)
    sx, tx, s_len, t_len = random_batch(B=4, sT=6, tT=8)
    criterion = LSRCrossEntropy(eps=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        loss = criterion(model(sx, tx, s_len, t_len), tx[:, 1:], t_len - 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.02
    model.eval()
    preds = model(sx, tx, s_len, t_len).argmax(-1)
    for b in range(4):
        n = t_len[b] - 1
        assert torch.equal(preds[b, :n], tx[b, 1 : n + 1])  # ends by predicting EOS
