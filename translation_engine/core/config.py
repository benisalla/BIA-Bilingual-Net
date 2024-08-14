import torch
from translation_engine.model.utils import get_lr

class Config:
    def __init__(self):
        # Data parameters
        self.data_dir = './translation_engine/src/dataset'
        self.s_suffix = "dr"
        self.t_suffix = "en"

        # Model parameters
        self.ev_size = 10000
        self.dv_size = 10000
        self.n_emb = 512
        self.n_head = 16
        self.h_size = self.n_emb // self.n_head
        self.n_block = 6
        self.exp_fac = 5
        self.max_seq_len = 1024
        self.d_rate = 0.2  

        # Learning parameters
        self.checkpoints_path = f"./translation_engine/src/{self.s_suffix}_{self.t_suffix}_chpts.pth.tar"
        self.toks_in_batch = 1000
        self.batches_per_step = 25000 // self.toks_in_batch
        self.n_steps = 100000
        self.warmup_steps = 8000
        self.step = 1
        self.lr = get_lr(step=self.step, n_emb=self.n_emb, warmup_steps=self.warmup_steps)
        self.start_epoch = 0
        self.betas = (0.9, 0.98)
        self.eps = 1e-9
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        self.label_smoothing = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cudnn_benchmark = False

    def set_vocab_size(self, tokenizer):
        self.ev_size = len(tokenizer.get_vocab())
        self.dv_size = self.ev_size

    def __repr__(self):
        return (f"<Config data_dir={self.data_dir}, checkpoint_path={self.checkpoint_path}, "
                f"ev_size={self.ev_size}, dv_size={self.dv_size}, n_emb={self.n_emb}, n_head={self.n_head}, "
                f"h_size={self.h_size}, n_block={self.n_block}, exp_fac={self.exp_fac}, "
                f"max_seq_len={self.max_seq_len}, d_rate={self.d_rate}, toks_in_batch={self.toks_in_batch}, "
                f"batches_per_step={self.batches_per_step}, n_steps={self.n_steps}, "
                f"warmup_steps={self.warmup_steps}, lr={self.lr:.2e}, start_epoch={self.start_epoch}, "
                f"betas={self.betas}, eps={self.eps}, weight_decay={self.weight_decay}, "
                f"grad_clip={self.grad_clip}, label_smoothing={self.label_smoothing}, device={self.device}, "
                f"cudnn_benchmark={self.cudnn_benchmark}>")
