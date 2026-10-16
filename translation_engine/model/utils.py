import os
import time
import torch
from IPython import get_ipython
from IPython.display import clear_output
import matplotlib.pyplot as plt
import math

from translation_engine.model.BIALinguaNet import BIALinguaNet


def get_lr(step: int, n_emb: int, warmup_steps: int) -> float:
    """
    "Noam" schedule from "Attention Is All You Need" (x2): linear warmup for
    `warmup_steps` optimizer steps, then decay proportional to 1/sqrt(step).
    """

    step = max(step, 1)  # step 0 would divide by zero
    sf = 2.0 * math.pow(n_emb, -0.5)  # Scale based on n_emb
    dp = math.pow(step, -0.5)  # LR during decay
    wp = step * math.pow(warmup_steps, -1.5)  # LR during warmup
    lr = sf * min(dp, wp)  # Final LR

    return lr


def save_checkpoint(
    epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, file_path: str
):
    """Save model, optimizer and the model's init args (enough to rebuild it with load_checkpoint)."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "init_args": model.get_init_args(),
    }
    torch.save(state, file_path)


def load_checkpoint(
    file_path: str,
    device: torch.device,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
):
    """Rebuild the model and its Adam optimizer from a checkpoint. Returns (model, optimizer, next_epoch)."""
    checkpoint = torch.load(file_path, map_location=device)
    init_args = checkpoint["init_args"]
    model = BIALinguaNet(**init_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    model = model.to(device)
    return model, optimizer, start_epoch


def update_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def in_notebook() -> bool:
    """True when running inside a Jupyter / Colab / Kaggle kernel."""
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in getattr(shell, "config", {})


def live_plot_dual(data_dict, figsize=(12, 5), title="", save_path=None):
    """
    Plot train / validation loss per epoch. In a notebook the figure is redrawn in place;
    in a terminal it is saved to `save_path` instead (plt.show() would block training).
    """
    notebook = in_notebook()
    if notebook:
        clear_output(wait=True)
    fig, ax1 = plt.subplots(figsize=figsize)

    if data_dict["train_loss_avg"]:
        ax1.plot(data_dict["train_loss_avg"], "r-", label="Train Loss Avg")
        ax1.annotate(
            f"Avg: {data_dict['train_loss_avg'][-1]:.4f}",
            xy=(len(data_dict["train_loss_avg"]) - 1, data_dict["train_loss_avg"][-1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

    if data_dict["val_loss_avg"]:
        ax1.plot(data_dict["val_loss_avg"], "b-", label="Val Loss Avg")
        ax1.annotate(
            f"Avg: {data_dict['val_loss_avg'][-1]:.4f}",
            xy=(len(data_dict["val_loss_avg"]) - 1, data_dict["val_loss_avg"][-1]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
        )

    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    plt.suptitle(title)
    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=100)
    if notebook:
        plt.show()
    plt.close(fig)


def validator(val_loader, model, criterion, device):
    model.eval()

    with torch.no_grad():
        losses = AverageMeter()
        for sx, tx, sx_len, tx_len in val_loader:
            sx, tx, sx_len, tx_len = (
                sx.to(device),
                tx.to(device),
                sx_len.to(device),
                tx_len.to(device),
            )
            preds = model(sx, tx, sx_len, tx_len)
            loss = criterion(x=preds, y=tx[:, 1:], lens=tx_len - 1)
            losses.update(loss.item(), (tx_len - 1).sum().item())

        return losses


def trainer(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    step,
    device,
    batches_per_step,
    epochs,
    warmup_steps,
    n_emb,
    grad_clip=None,
    lr_scale=1.0,
    checkpoints_path=None,
):
    """
    Train for one epoch.

    Gradients are accumulated over `batches_per_step` batches, then the optimizer takes
    one step (so one step sees about `batches_per_step * toks_in_batch` target tokens).
    The learning rate follows get_lr(step) * lr_scale.

    Returns:
        (AverageMeter, int): the token-weighted training loss and the updated step counter.
    """
    model.train()
    data_time = AverageMeter()
    step_time = AverageMeter()
    losses = AverageMeter()

    optimizer.zero_grad()  # drop gradients left over from an incomplete accumulation
    start_data_time = time.time()
    start_step_time = time.time()

    for i, (sx, tx, sx_len, tx_len) in enumerate(train_loader):
        sx, tx, sx_len, tx_len = (
            sx.to(device),
            tx.to(device),
            sx_len.to(device),
            tx_len.to(device),
        )

        data_time.update(time.time() - start_data_time)

        preds = model(sx, tx, sx_len, tx_len)
        loss = criterion(x=preds, y=tx[:, 1:], lens=tx_len - 1)
        losses.update(loss.item(), (tx_len - 1).sum().item())

        # accumulate: each batch adds its share to the gradient of the step
        (loss / batches_per_step).backward()

        if (i + 1) % batches_per_step == 0:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step += 1
            update_lr(
                optimizer,
                new_lr=get_lr(step=step, n_emb=n_emb, warmup_steps=warmup_steps) * lr_scale,
            )

            step_time.update(time.time() - start_step_time)

            start_step_time = time.time()

            # extra snapshots during the last 2 epochs (e.g. to average checkpoints)
            if checkpoints_path and (epoch in [epochs - 1, epochs - 2]) and (step % 1500 == 0):
                folder, name = os.path.split(checkpoints_path)
                save_checkpoint(
                    epoch, model, optimizer, file_path=os.path.join(folder, f"step{step}_{name}")
                )

        start_data_time = time.time()

    return losses, step


def build_model(config) -> BIALinguaNet:
    """A fresh BIALinguaNet with the sizes in `config`."""
    return BIALinguaNet(
        ev_size=config.ev_size,
        dv_size=config.dv_size,
        n_emb=config.n_emb,
        n_head=config.n_head,
        h_size=config.h_size,
        n_block=config.n_block,
        exp_fac=config.exp_fac,
        max_seq_len=config.max_seq_len,
        d_rate=config.d_rate,
        device=config.device,
    )


def fit(
    config,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    start_epoch,
    epochs,
    lr_scale=1.0,
    save_path=None,
    plot_path=None,
    verbose=True,
    time_budget=None,
):
    """
    Train from `start_epoch` to `epochs`, validating after every epoch.
    The checkpoint at `save_path` is overwritten each time the validation loss improves,
    so it always holds the best model so far. Stops early after `time_budget` seconds
    (checked at the end of each epoch). Returns the loss history.
    """
    start_time = time.time()
    stats = {
        "train_loss_value": [],
        "train_loss_avg": [],
        "val_loss_value": [],
        "val_loss_avg": [],
    }
    best_val = math.inf
    steps_per_epoch = max(1, train_loader.n_batches // config.batches_per_step)

    for epoch in range(start_epoch, epochs):
        step = epoch * steps_per_epoch
        # set the lr for this point of the schedule (matters on resume / fine-tuning)
        update_lr(
            optimizer,
            get_lr(step=step, n_emb=config.n_emb, warmup_steps=config.warmup_steps) * lr_scale,
        )

        train_loader.create_batches()
        val_loader.create_batches()

        train_losses, step = trainer(
            train_loader=train_loader,
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
            grad_clip=config.grad_clip,
            lr_scale=lr_scale,
            checkpoints_path=save_path,
        )

        val_losses = validator(
            val_loader=val_loader, model=model, criterion=criterion, device=config.device
        )

        stats["train_loss_value"].append(train_losses.value)
        stats["train_loss_avg"].append(train_losses.avg)
        stats["val_loss_value"].append(val_losses.value)
        stats["val_loss_avg"].append(val_losses.avg)

        live_plot_dual(stats, figsize=(12, 5), title=f"Epoch {epoch + 1}/{epochs}", save_path=plot_path)

        improved = val_losses.avg < best_val
        if improved:
            best_val = val_losses.avg
            if save_path:
                save_checkpoint(epoch, model, optimizer, file_path=save_path)
        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs} | step {step} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"train loss {train_losses.avg:.4f} | val loss {val_losses.avg:.4f}"
                + (" | saved" if improved and save_path else ""),
                flush=True,
            )
        if time_budget is not None and time.time() - start_time > time_budget:
            if verbose:
                print(f"time budget reached after epoch {epoch + 1}, stopping.")
            break

    return stats
