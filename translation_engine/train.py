import torch
import torch.nn as nn
import torch.optim as optim
from translation_engine.core.config import Config
from translation_engine.data.dataloader import DataLoader
from translation_engine.model import LSRCrossEntropy
from translation_engine.model.BIALinguaNet import BIALinguaNet
from translation_engine.model.utils import live_plot_dual, load_checkpoint, save_checkpoint, trainer, validator


# Build configs
config = Config()


# Initialize data-loaders
train_loader = DataLoader(
    data_dir=config.data_dir,
    s_suffix=config.s_suffix,
    t_suffix=config.t_suffix,
    split="train",
    toks_in_batch=config.toks_in_batch)

val_loader = DataLoader(
    data_dir=config.data_dir,
    s_suffix="dr",
    t_suffix="en",
    split="val",
    toks_in_batch=config.toks_in_batch)

# Initialize model or load checkpoint
model = BIALinguaNet(
    ev_size=config.ev_size,
    dv_size=config.dv_size,
    n_emb=config.n_emb,
    n_head=config.n_head,
    h_size=config.h_size,
    n_block=config.n_block,
    exp_fac=config.exp_fac,
    max_seq_len=config.max_seq_len,
    d_rate=config.d_rate,
    device=config.device)

optimizer = torch.optim.Adam(
    params=[p for p in model.parameters() if p.requires_grad],
    lr=config.lr,
    betas=config.betas,
    eps=config.eps,
    weight_decay=config.weight_decay)


if os.path.exists(config.checkpoints_path):
    model, optimizer, start_epoch = load_checkpoint(
        file_path=config.checkpoints_path,
        model=model,
        optimizer=optimizer,
        device=config.device)
    print("Model loaded from checkpoint ...")
else:
    print("Model loaded from scratch ...")

# Loss function
criterion = LSRCrossEntropy(eps=config.label_smoothing)

# Move to default device
model = model.to(config.device)
criterion = criterion.to(config.device)

# Find total epochs to train
epochs = (config.n_steps // (train_loader.n_batches // config.batches_per_step)) + 1


# Start training
stats = {
    "train_loss_value": [],
    "train_loss_avg": [],
    "val_loss_value": [],
    "val_loss_avg": [],
}

print("traing started ...")
for epoch in range(start_epoch, epochs):
    step = epoch * train_loader.n_batches // config.batches_per_step

    train_loader.create_batches()
    val_loader.create_batches()

    train_losses = trainer(train_loader=train_loader,
                         model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         epoch=epoch,
                         step=step,
                         device=config.device,
                         batches_per_step=config.batches_per_step,
                         epochs=epochs,
                         warmup_steps=config.warmup_steps,
                         n_emb=config.n_emb,
                         grad_clip=config.grad_clip)

    val_losses = validator(
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        device=config.device)

    stats["train_loss_value"].append(train_losses.value)
    stats["train_loss_avg"].append(train_losses.avg)
    stats["val_loss_value"].append(val_losses.value)
    stats["val_loss_avg"].append(val_losses.avg)

    live_plot_dual(stats, figsize=(12, 5), title=f"Epoch {epoch + 1}/{epochs}")
    save_checkpoint(epoch, model, optimizer, file_path=config.checkpoints_path)