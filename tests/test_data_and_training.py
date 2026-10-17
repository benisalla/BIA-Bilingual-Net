"""Data loader, config, learning-rate schedule, training loop, checkpoints and scripts."""

import copy
import json

import pytest
import torch

from conftest import BOS, EOS, PAD, PAIRS, FakeTokenizer, make_model, random_batch
from translation_engine.core.config import Config
from translation_engine.data.dataloader import DataLoader
from translation_engine.model.LSRCrossEntropy import LSRCrossEntropy
from translation_engine.model.utils import (
    fit,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    trainer,
    validator,
)


def make_loader(data_dir, split="train", toks_in_batch=12, **kwargs):
    return DataLoader(
        data_dir=str(data_dir),
        s_suffix="dr",
        t_suffix="en",
        split=split,
        toks_in_batch=toks_in_batch,
        tokenizer=FakeTokenizer(),
        **kwargs,
    )


# --- data -------------------------------------------------------------------------


def test_loader_yields_every_pair_once_per_epoch(data_dir):
    loader = make_loader(data_dir)
    for _ in range(2):  # iterating twice must restart from the first batch
        seen = []
        for sx, tx, s_len, t_len in loader:
            for b in range(sx.size(0)):
                seen.append((tuple(sx[b, : s_len[b]].tolist()), tuple(tx[b, : t_len[b]].tolist())))
        assert len(seen) == len(PAIRS)
        tok = FakeTokenizer()
        expected = {(tuple(tok.encode(p["dr"])), tuple([BOS] + tok.encode(p["en"]) + [EOS])) for p in PAIRS}
        assert set(seen) == expected


def test_batches_are_padded_and_respect_the_token_budget(data_dir):
    loader = make_loader(data_dir, toks_in_batch=12)
    for sx, tx, s_len, t_len in loader:
        assert tx[:, 0].eq(BOS).all()
        assert tx.size(1) == t_len.max() and sx.size(1) == s_len.max()
        assert tx.size(0) * t_len.max() <= 12 or tx.size(0) == 1
        for b in range(tx.size(0)):
            assert tx[b, t_len[b] - 1] == EOS
            assert tx[b, t_len[b]:].eq(PAD).all() and sx[b, s_len[b]:].eq(PAD).all()


def test_loader_drops_too_long_and_empty_pairs(tmp_path):
    pairs = PAIRS + [{"dr": "", "en": "hello"}, {"dr": "salam " * 20, "en": "hello"}]
    with open(tmp_path / "train.json", "w") as f:
        json.dump(pairs, f)
    loader = make_loader(tmp_path, max_seq_len=8)
    assert len(loader.data) == len(PAIRS) and loader.n_dropped == 2


def test_loader_limit_and_huge_sentences(data_dir):
    loader = make_loader(data_dir, toks_in_batch=1, limit=3)  # budget smaller than any sentence
    assert len(loader.data) == 3
    assert sum(sx.size(0) for sx, *_ in loader) == 3


def test_validation_split_is_not_shuffled(data_dir):
    loader = make_loader(data_dir, split="val")
    first = [b[0].tolist() for b in loader]
    loader.create_batches()
    assert [b[0].tolist() for b in loader] == first


# --- config & schedule ------------------------------------------------------------


def test_config_overrides_and_derived_values():
    config = Config(s_suffix="en", t_suffix="dr", n_emb=128, n_head=8, toks_in_batch=500)
    assert config.h_size == 16
    assert config.batches_per_step == 50
    assert config.checkpoints_path.endswith("en_dr_chpts.pth.tar")
    with pytest.raises(AttributeError):
        Config(n_embd=3)


def test_lr_schedule_warms_up_then_decays():
    lrs = [get_lr(step, n_emb=64, warmup_steps=100) for step in range(0, 1000)]
    peak = max(range(1000), key=lambda i: lrs[i])
    assert peak == 100
    assert all(a <= b for a, b in zip(lrs[1:100], lrs[2:101]))
    assert all(a >= b for a, b in zip(lrs[100:-1], lrs[101:]))


# --- training loop ----------------------------------------------------------------


def test_gradient_accumulation_equals_one_big_batch():
    """batches_per_step=2 must use both batches (the old loop zeroed the first one)."""
    model = make_model()
    reference = copy.deepcopy(model)
    batches = [random_batch(seed=1), random_batch(seed=2)]
    criterion = LSRCrossEntropy(eps=0.1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    _, step = trainer(
        batches, model, criterion, optimizer, epoch=0, step=0, device="cpu",
        batches_per_step=2, epochs=1, warmup_steps=10, n_emb=32, grad_clip=None,
    )
    assert step == 1

    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    reference.train()
    loss = sum(criterion(reference(sx, tx, sl, tl), tx[:, 1:], tl - 1) for sx, tx, sl, tl in batches) / 2
    loss.backward()
    ref_optimizer.step()

    for p, q in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_validator_is_token_weighted():
    model = make_model()
    criterion = LSRCrossEntropy(eps=0.0)
    batches = [random_batch(B=1, seed=3), random_batch(B=3, seed=4)]
    meter = validator(batches, model, criterion, "cpu")
    model.eval()
    with torch.no_grad():
        total = sum(
            criterion(model(sx, tx, sl, tl), tx[:, 1:], tl - 1) * (tl - 1).sum() for sx, tx, sl, tl in batches
        )
        n = sum((tl - 1).sum() for *_, tl in batches)
    assert meter.avg == pytest.approx((total / n).item(), rel=1e-5)


def test_checkpoint_round_trip(tmp_path):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "sub" / "ckpt.pth.tar"
    save_checkpoint(4, model, optimizer, file_path=str(path))
    loaded, loaded_opt, start_epoch = load_checkpoint(str(path), device="cpu")
    assert start_epoch == 5
    assert loaded.decoder.fc.weight is loaded.encoder.tok_emb.weight  # still tied
    sx, tx, sl, tl = random_batch()
    model.eval(), loaded.eval()
    assert torch.allclose(model(sx, tx, sl, tl), loaded(sx, tx, sl, tl))


def tiny_config(tmp_path, **kwargs):
    config = Config(
        data_dir=str(tmp_path), n_emb=32, n_head=4, n_block=1, max_seq_len=16, d_rate=0.0,
        label_smoothing=0.0, toks_in_batch=12, batches_per_step=1, warmup_steps=20,
        device=torch.device("cpu"), **kwargs,
    )
    config.set_vocab_size(FakeTokenizer())
    return config


def test_fit_learns_and_saves_the_best_model(data_dir, tmp_path):
    from translation_engine.model.utils import build_model

    torch.manual_seed(0)
    config = tiny_config(data_dir)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    save_path = tmp_path / "out" / "best.pth.tar"
    stats = fit(
        config, model, optimizer, LSRCrossEntropy(eps=0.0),
        make_loader(data_dir), make_loader(data_dir, split="val"),
        start_epoch=0, epochs=30, save_path=str(save_path), plot_path=str(tmp_path / "loss.png"),
        verbose=False,
    )
    assert stats["val_loss_avg"][-1] < 0.5 * stats["val_loss_avg"][0]
    assert save_path.exists() and (tmp_path / "loss.png").exists()


# --- scripts and app (end to end, fake tokenizer) ----------------------------------


@pytest.fixture
def small_scripts(monkeypatch):
    """Run train.py / finetune.py with the fake tokenizer and a tiny model."""
    import translation_engine.finetune as finetune_module
    import translation_engine.train as train_module

    for module in (train_module, finetune_module):
        monkeypatch.setattr(module, "build_tokenizer", lambda name="gpt2": FakeTokenizer())
    original = Config.__init__

    def small_init(self, **overrides):
        overrides.setdefault("n_emb", 32)
        overrides.setdefault("n_head", 4)
        overrides.setdefault("toks_in_batch", 12)
        overrides.setdefault("batches_per_step", 1)
        original(self, **overrides)

    monkeypatch.setattr(Config, "__init__", small_init)
    return train_module, finetune_module


def test_train_overfit_then_finetune_then_app(data_dir, tmp_path, small_scripts, monkeypatch):
    train_module, finetune_module = small_scripts
    ckpt = tmp_path / "ckpt" / "dr_en_chpts.pth.tar"
    stats = train_module.main(
        ["--overfit", "6", "--epochs", "60", "--warmup-steps", "50",
         "--data-dir", str(data_dir), "--checkpoint", str(ckpt)]
    )
    assert stats["val_loss_avg"][-1] < 0.1, stats["val_loss_avg"][-5:]
    assert ckpt.exists()

    # resume: a second run continues after the saved (best) epoch, not from 0
    saved_epoch = torch.load(ckpt, map_location="cpu")["epoch"]
    stats = train_module.main(["--epochs", "62", "--data-dir", str(data_dir), "--checkpoint", str(ckpt)])
    assert len(stats["val_loss_avg"]) == 62 - (saved_epoch + 1)

    out = tmp_path / "ckpt" / "ft.pth.tar"
    finetune_module.main(
        ["--data-dir", str(data_dir), "--pretrained", str(ckpt), "--output", str(out), "--epochs", "2"]
    )
    assert out.exists()

    from translation_engine.app import utils as app_utils

    model = app_utils.load_model(str(ckpt))
    best, _ = model.translate("ana bghit nmchi", FakeTokenizer(), top_k=0, top_p=1.0)
    assert best == "i want to go"


def test_streamlit_app(tmp_path, monkeypatch):
    """Drive the real Streamlit script: missing model -> clear error; with models -> translation shown."""
    from streamlit.testing.v1 import AppTest

    from translation_engine.app import utils as app_utils

    monkeypatch.setattr(app_utils, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(app_utils, "load_tokenizer", FakeTokenizer)

    at = AppTest.from_file("../translation_engine/app/main.py", default_timeout=60).run()
    assert not at.exception
    assert "No trained model" in at.error[0].value

    model = make_model(vocab=len(FakeTokenizer()))
    optimizer = torch.optim.Adam(model.parameters())
    for name in app_utils.MODEL_FILES.values():
        save_checkpoint(0, model, optimizer, file_path=str(tmp_path / name))
    at = AppTest.from_file("../translation_engine/app/main.py", default_timeout=60).run()
    at.text_area[0].input("salam labas")
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("Score" in m.value for m in at.markdown)
