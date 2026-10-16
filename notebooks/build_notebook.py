"""Build notebooks/train_colab_kaggle.ipynb: a self-contained training notebook.

Every module of translation_engine (model, data, config) is embedded verbatim as a
%%writefile cell, so the notebook runs the exact repo code and needs no GitHub access.
Regenerate after changing the package:

    python notebooks/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "train_colab_kaggle.ipynb"
MODULES = [
    "translation_engine/__init__.py",
    "translation_engine/core/__init__.py",
    "translation_engine/core/config.py",
    "translation_engine/data/__init__.py",
    "translation_engine/data/dataloader.py",
    "translation_engine/data/prepare_data.py",
    "translation_engine/model/__init__.py",
    "translation_engine/model/ResFFNet.py",
    "translation_engine/model/ResMHAtten.py",
    "translation_engine/model/Encoder.py",
    "translation_engine/model/Decoder.py",
    "translation_engine/model/BIALinguaNet.py",
    "translation_engine/model/LSRCrossEntropy.py",
    "translation_engine/model/utils.py",
]

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# BIA-Bilingual-Net: training notebook (Kaggle & Colab)

Train the **Darija ⇄ English** BIALinguaNet translator on a **free Kaggle or Colab GPU**.
The notebook is **self-contained**: section 2 writes the `translation_engine` package
(identical to the repo), and training goes through the same `fit()` loop as
`python -m translation_engine.train`, with the **same default configuration**.
Just upload it and press **Run all**.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Internet | *Settings → Internet → On* (to download data) | on by default |
| Session limit | 12 h per session, 30 h GPU / week | ~12 h, can disconnect when idle |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

The first run downloads the DODa corpus (~3 MB, ~45k sentence pairs) and the GPT-2 tokenizer.
One run trains **one direction** (`DIRECTION`); run it twice to get both models the app uses.

**Sections:** 1. Install · 2. The package · 3. Settings · 4. Data · 5. Model & sanity checks ·
6. Overfit a small chunk · 7. Training · 8. Results · 9. Download
""")

md("""
## 1. Install

Kaggle and Colab already have PyTorch, transformers and matplotlib; nothing to install.
""")
code("""
import importlib.util
import sys

missing = [m for m in ("torch", "transformers", "matplotlib") if importlib.util.find_spec(m) is None]
if missing:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import torch
import transformers

print("torch", torch.__version__, "| transformers", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

# ---------------------------------------------------------------------------
md("""
## 2. The package

The cells below write `translation_engine/*.py`, copied from the repo when this notebook was
built (`python notebooks/build_notebook.py`). Nothing to change here.
""")
code("""
import os

for folder in ("translation_engine/core", "translation_engine/data", "translation_engine/model"):
    os.makedirs(folder, exist_ok=True)
""")
for module in MODULES:
    source = (ROOT / module).read_text()
    code(f"%%writefile {module}\n{source}")

# ---------------------------------------------------------------------------
md("""
## 3. Settings

- `DIRECTION`: `"dr-en"` (Darija → English) or `"en-dr"` (English → Darija).
- `SMOKE_TEST = True` runs a tiny version (256 pairs, a few epochs) to check the whole
  notebook in minutes, even on a CPU. Keep it `False` for real training.
- The model and optimisation settings are the repo's `Config` defaults; change them with
  `MODEL_OVERRIDES` (e.g. `dict(n_emb=256, n_head=8, n_block=4)` for a bigger model).
""")
code("""
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.getcwd())
from translation_engine.core.config import Config
from translation_engine.data import prepare_data
from translation_engine.data.dataloader import DataLoader, build_tokenizer
from translation_engine.model.LSRCrossEntropy import LSRCrossEntropy
from translation_engine.model.utils import build_model, fit, load_checkpoint

DIRECTION = "dr-en"         # "dr-en" or "en-dr"
SMOKE_TEST = False          # True = tiny quick check of the whole notebook
USE_GOOGLE_DRIVE = False    # Colab only: keep checkpoints on Drive (survive disconnects)
RESUME = True               # continue from the checkpoint in OUTPUT_DIR if it exists
TIME_BUDGET_HOURS = 11.0    # stop cleanly before the 12 h session limit
EPOCHS = None               # None = derived from Config.n_steps (100k optimizer steps)
OVERFIT_PAIRS = 32          # section 6: pairs the model must memorise
MODEL_OVERRIDES = dict()    # e.g. dict(n_emb=256, n_head=8, n_block=4)
SEED = 0

# --- where checkpoints go ------------------------------------------------------
ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path("/content/drive/MyDrive/bia-bilingual-net")
elif ON_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working/checkpoints")
else:
    OUTPUT_DIR = Path("checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SRC, TGT = DIRECTION.split("-")
DATA_DIR = "translation_engine/src/dataset"
config = Config(
    s_suffix=SRC, t_suffix=TGT, data_dir=DATA_DIR, checkpoints_dir=str(OUTPUT_DIR),
    epochs=EPOCHS, **MODEL_OVERRIDES,
)
SPLITS = ("tiny_train", "tiny_val", "tiny_test") if SMOKE_TEST else ("train", "val", "test")
if SMOKE_TEST:
    config.epochs = 3
    OVERFIT_PAIRS, TIME_BUDGET_HOURS = 8, 0.5
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = config.cudnn_benchmark
print(config)
""")

# ---------------------------------------------------------------------------
md("""
## 4. Data

[DODa](https://github.com/darija-open-dataset/dataset) sentence pairs (Darija written in Latin
script, e.g. *"bghit nmchi l dar"*), from
[atlasia/darija_english](https://huggingface.co/datasets/atlasia/darija_english).
`prepare_data` removes empty rows and duplicates, shuffles with a fixed seed and writes
`train / val / test` json files (94% / 3% / 3%).

The tokenizer is GPT-2's BPE plus `<PAD>`, `<SOS>`, `<EOS>`, shared by both languages (that is
what lets the model tie its input and output embeddings).
""")
code("""
if not Path(DATA_DIR, "train.json").exists():
    prepare_data.main(["--out-dir", DATA_DIR])

tokenizer = build_tokenizer(config.tokenizer_name)
config.set_vocab_size(tokenizer)
print("vocab size:", config.ev_size, "| pad / sos / eos ids:",
      tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)

loader_args = dict(data_dir=DATA_DIR, s_suffix=SRC, t_suffix=TGT, toks_in_batch=config.toks_in_batch,
                   tokenizer=tokenizer, max_seq_len=config.max_seq_len)
train_loader = DataLoader(split=SPLITS[0], **loader_args)
val_loader = DataLoader(split=SPLITS[1], **loader_args)
print(f"train: {len(train_loader.data):,} pairs, {train_loader.n_batches:,} batches "
      f"({train_loader.n_dropped} dropped) | val: {len(val_loader.data):,} pairs")
""")
md("### 4.1 Look at the data")
code("""
with open(Path(DATA_DIR, f"{SPLITS[0]}.json"), encoding="utf-8") as f:
    raw_pairs = json.load(f)
for pair in raw_pairs[:8]:
    print(f"{SRC}: {pair[SRC]:45}  {TGT}: {pair[TGT]}")

src_lens = [item[2] for item in train_loader.data]
tgt_lens = [item[3] for item in train_loader.data]
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist([src_lens, tgt_lens], bins=40, label=[f"source ({SRC})", f"target ({TGT}, with SOS/EOS)"])
ax.set_xlabel("tokens per sentence")
ax.legend()
plt.show()

sx, tx, s_len, t_len = next(iter(train_loader))
print("one batch:", tuple(sx.shape), tuple(tx.shape), "| target tokens:", int(t_len.sum()))
print("decoded target:", tokenizer.decode(tx[0, : t_len[0]].tolist()))
""")

# ---------------------------------------------------------------------------
md("""
## 5. Model & neural-network sanity checks

Encoder-decoder Transformer with pre-norm residual blocks, learned positions and tied
embeddings. Before spending GPU hours we check, on one real batch, that:

1. the **initial loss ≈ ln(vocab)** (the untrained model guesses uniformly, it is not confidently wrong),
2. **every parameter gets a gradient**,
3. the decoder is **causal** (changing a future target token does not change earlier predictions),
4. **padding is ignored** (adding padding to the source does not change the output).
""")
code("""
device = config.device
model = build_model(config).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = LSRCrossEntropy(eps=0.0)
sx, tx, s_len, t_len = (t.to(device) for t in next(iter(val_loader)))

# 1. initial loss
model.eval()
with torch.no_grad():
    loss0 = criterion(model(sx, tx, s_len, t_len), tx[:, 1:], t_len - 1).item()
print(f"1. initial loss {loss0:.3f} vs ln(vocab) = {math.log(config.dv_size):.3f}")
assert abs(loss0 - math.log(config.dv_size)) < 1.5

# 2. gradients
model.train()
criterion(model(sx, tx, s_len, t_len), tx[:, 1:], t_len - 1).backward()
no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
print("2. parameters without gradient:", no_grad or "none")
assert not no_grad
model.zero_grad()

# 3. causality and 4. padding, on one sentence
model.eval()
with torch.no_grad():
    s1, t1 = sx[:1, : s_len[0]], tx[:1, : t_len[0]]
    sl1, tl1 = s_len[:1], t_len[:1]
    base = model(s1, t1, sl1, tl1)
    changed = t1.clone()
    changed[0, -1] = (changed[0, -1] + 1) % config.dv_size
    assert torch.allclose(model(s1, changed, sl1, tl1)[:, :-1], base[:, :-1], atol=1e-4)
    print("3. causal decoder: ok")
    padded = torch.cat([s1, torch.full((1, 5), tokenizer.pad_token_id, device=device)], dim=1)
    assert torch.allclose(model(padded, t1, sl1, tl1), base, atol=1e-4)
    print("4. source padding ignored: ok")
""")

# ---------------------------------------------------------------------------
md("""
## 6. Overfit a small chunk of data

The classic test before a long run: a fresh model, without dropout and label smoothing, must
**memorise** a few real pairs. The validation set here *is* those pairs, so the loss should drop
close to 0 and the beam search should give back the exact training translations. If it cannot
do this, something is broken and full training would be wasted time.
""")
code("""
overfit_config = Config(
    s_suffix=SRC, t_suffix=TGT, data_dir=DATA_DIR, d_rate=0.0, label_smoothing=0.0,
    batches_per_step=1, warmup_steps=400, **MODEL_OVERRIDES,
)
overfit_config.set_vocab_size(tokenizer)
overfit_loader = DataLoader(split=SPLITS[0], limit=OVERFIT_PAIRS, **loader_args)
overfit_model = build_model(overfit_config).to(device)
overfit_optimizer = torch.optim.Adam(overfit_model.parameters(), lr=overfit_config.lr,
                                     betas=overfit_config.betas, eps=overfit_config.eps)
overfit_stats = fit(
    overfit_config, overfit_model, overfit_optimizer, LSRCrossEntropy(eps=0.0),
    overfit_loader, overfit_loader, start_epoch=0, epochs=20 if SMOKE_TEST else 60, verbose=False,
)
print(f"loss: {overfit_stats['val_loss_avg'][0]:.3f} -> {min(overfit_stats['val_loss_avg']):.4f}")

exact = 0
for pair in raw_pairs[:OVERFIT_PAIRS]:
    best, _ = overfit_model.translate(pair[SRC], tokenizer, beam_size=4)
    exact += best.strip() == pair[TGT].strip()
print(f"exact translations of the memorised pairs: {exact}/{OVERFIT_PAIRS}")
del overfit_model, overfit_optimizer
""")

# ---------------------------------------------------------------------------
md("""
## 7. Training

Same loop as `python -m translation_engine.train`: label-smoothed cross-entropy, Adam with the
warmup + 1/√step ("Noam") learning-rate schedule, gradients accumulated over
`batches_per_step` batches (~25k target tokens per optimizer step), gradient clipping.
After each epoch it validates and saves the checkpoint when the validation loss improves.
The plot is redrawn after every epoch.

**Interrupted, or hit the session limit?** Run the notebook again with `RESUME = True`: it
continues from the saved checkpoint. On Kaggle, to continue in a *new* session, add the
previous version's output as input data and copy the checkpoint back to `/kaggle/working/checkpoints`.
""")
code("""
model = build_model(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas,
                             eps=config.eps, weight_decay=config.weight_decay)
start_epoch = 0
if RESUME and Path(config.checkpoints_path).exists():
    model, optimizer, start_epoch = load_checkpoint(config.checkpoints_path, device=device)
    print(f"resumed from {config.checkpoints_path} at epoch {start_epoch}")
model = model.to(device)

steps_per_epoch = max(1, train_loader.n_batches // config.batches_per_step)
epochs = config.epochs or config.n_steps // steps_per_epoch + 1
print(f"{steps_per_epoch} optimizer steps per epoch, training until epoch {epochs}")

plot_path = OUTPUT_DIR / f"{SRC}_{TGT}_loss.png"
stats = fit(
    config, model, optimizer, LSRCrossEntropy(eps=config.label_smoothing, device=device),
    train_loader, val_loader, start_epoch=start_epoch, epochs=epochs,
    save_path=config.checkpoints_path, plot_path=str(plot_path),
    time_budget=TIME_BUDGET_HOURS * 3600,
)
""")

# ---------------------------------------------------------------------------
md("""
## 8. Results

Loads the best checkpoint (exactly as the app does) and translates sentences it never trained
on: some from the test split, and a few everyday phrases.
""")
code("""
best_model, _, best_epoch = load_checkpoint(config.checkpoints_path, device=device)
print(f"best checkpoint: epoch {best_epoch}")


def translate(sentence, **kwargs):
    options = dict(beam_size=4, len_norm_coeff=0.6, top_k=50, top_p=0.95, temperature=1.0)
    options.update(kwargs)
    return best_model.translate(sentence, tokenizer, **options)


with open(Path(DATA_DIR, f"{SPLITS[2]}.json"), encoding="utf-8") as f:
    test_pairs = json.load(f)

print("--- test split (never seen in training) ---")
for pair in test_pairs[:15]:
    best, _ = translate(pair[SRC])
    print(f"{SRC}: {pair[SRC]}\\n   model: {best}\\n   human: {pair[TGT]}\\n")

EXAMPLES = {
    "dr": ["fin awa ghadi?", "Mashi mochkil", "Chno katdir daba?", "bghit nmchi ldari",
           "smah lia, ma fhemtkch", "bghit nakol shi haja", "lah y3tik lkhir"],
    "en": ["where are you going?", "it's not a problem", "what are you doing now?",
           "I want to go home", "sorry, I don't understand", "I want to eat something", "thank you"],
}
print("--- everyday phrases ---")
for sentence in EXAMPLES[SRC]:
    best, hypos = translate(sentence)
    others = " | ".join(h["hypothesis"] for h in sorted(hypos, key=lambda h: -h["score"])[1:3])
    print(f"{sentence:30} -> {best:35} (other hypotheses: {others})")
""")

# ---------------------------------------------------------------------------
md("""
## 9. Download the trained model

Zips the checkpoint and the loss curve. To use it with the repo's Streamlit app, put the
checkpoint in `translation_engine/src/checkpoints/` (the app expects `dr_en_chpts.pth.tar`
and `en_dr_chpts.pth.tar`), then run `streamlit run translation_engine/app/main.py`.
""")
code("""
with tempfile.TemporaryDirectory() as tmp:
    for path in (Path(config.checkpoints_path), plot_path):
        if path.exists():
            shutil.copy(path, Path(tmp) / path.name)
    archive = shutil.make_archive(str(OUTPUT_DIR / f"bia-bilingual-net-{SRC}-{TGT}"), "zip", tmp)
print("created", archive)

if ON_COLAB:
    from google.colab import files

    files.download(archive)
elif ON_KAGGLE:
    print("Kaggle: click 'Save Version', then download it from the notebook's Output tab.")
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"gpuType": "T4", "provenance": []},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
