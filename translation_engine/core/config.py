import os
import torch
from translation_engine.model.utils import get_lr


class Config:
    """
    All hyper-parameters in one place. Any attribute can be overridden with a keyword,
    e.g. Config(s_suffix="en", t_suffix="dr", n_steps=1000). Values that depend on
    others (h_size, batches_per_step, checkpoints_path, lr) are derived after the overrides.
    """

    def __init__(self, **overrides):
        # Data parameters
        self.data_dir = './translation_engine/src/dataset'
        self.s_suffix = "dr"  # source language key in the json files
        self.t_suffix = "en"  # target language key in the json files
        self.tokenizer_name = "gpt2"

        # Model parameters
        self.ev_size = 1000  # replaced by the tokenizer size (set_vocab_size)
        self.dv_size = 1000
        self.n_emb = 64
        self.n_head = 16
        self.h_size = None  # None = n_emb // n_head
        self.n_block = 1
        self.exp_fac = 3
        self.max_seq_len = 128
        self.d_rate = 0.2

        # Learning parameters
        self.checkpoints_dir = './translation_engine/src/checkpoints'
        self.checkpoints_path = None  # None = <checkpoints_dir>/<s>_<t>_chpts.pth.tar
        self.toks_in_batch = 1000  # target tokens per batch
        self.batches_per_step = None  # None = 25000 // toks_in_batch (~25k tokens per optimizer step)
        self.n_steps = 100000  # total optimizer steps (sets the number of epochs)
        self.epochs = None  # None = derived from n_steps
        self.warmup_steps = 8000
        self.step = 1
        self.lr = None  # initial lr; None = get_lr(step)
        self.start_epoch = 0
        self.betas = (0.9, 0.98)
        self.eps = 1e-9
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        self.label_smoothing = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cudnn_benchmark = False

        # Fine-tuning parameters
        self.finetune_lr_scale = 0.1  # fine-tune with 10x smaller learning rates
        self.finetune_epochs = 10

        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config option: {key}")
            setattr(self, key, value)

        if self.h_size is None:
            self.h_size = self.n_emb // self.n_head
        if self.batches_per_step is None:
            self.batches_per_step = max(1, 25000 // self.toks_in_batch)
        if self.checkpoints_path is None:
            self.checkpoints_path = os.path.join(
                self.checkpoints_dir, f"{self.s_suffix}_{self.t_suffix}_chpts.pth.tar"
            )
        if self.lr is None:
            self.lr = get_lr(step=self.step, n_emb=self.n_emb, warmup_steps=self.warmup_steps)

    def set_vocab_size(self, tokenizer):
        # len(tokenizer) counts the added special tokens (<PAD>, <SOS>, <EOS>)
        self.ev_size = len(tokenizer)
        self.dv_size = self.ev_size

    def __repr__(self):
        return (f"<Config data_dir={self.data_dir}, direction={self.s_suffix}->{self.t_suffix}, "
                f"checkpoint_path={self.checkpoints_path}, "
                f"ev_size={self.ev_size}, dv_size={self.dv_size}, n_emb={self.n_emb}, n_head={self.n_head}, "
                f"h_size={self.h_size}, n_block={self.n_block}, exp_fac={self.exp_fac}, "
                f"max_seq_len={self.max_seq_len}, d_rate={self.d_rate}, toks_in_batch={self.toks_in_batch}, "
                f"batches_per_step={self.batches_per_step}, n_steps={self.n_steps}, "
                f"warmup_steps={self.warmup_steps}, lr={self.lr:.2e}, start_epoch={self.start_epoch}, "
                f"betas={self.betas}, eps={self.eps}, weight_decay={self.weight_decay}, "
                f"grad_clip={self.grad_clip}, label_smoothing={self.label_smoothing}, device={self.device}, "
                f"cudnn_benchmark={self.cudnn_benchmark}>")
