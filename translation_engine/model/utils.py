import gc
import torch
import torch.nn as nn
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import math
from torch.nn.utils import clip_grad_norm_


def get_lr(step: int, n_emb: int, warmup_steps: int) -> float:

    sf = 2.0 * math.pow(n_emb, -0.5)  # Scale based on n_emb
    dp = math.pow(step, -0.5)  # LR during decay
    wp = step * math.pow(warmup_steps, -1.5)  # LR during warmup
    lr = sf * min(dp, wp)  # Final LR

    return lr


def save_checkpoint(
    epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, file_path: str
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "init_args": model.get_init_args(),
    }
    torch.save(state, file_path)


def load_checkpoint(
    file_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
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


def live_plot_dual(data_dict, figsize=(12, 5), title=""):
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
    plt.show()


def validator(
    val_loader, 
    model, 
    criterion, 
    device):
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
):
    model.train()
    data_time = AverageMeter()
    step_time = AverageMeter()
    losses = AverageMeter()

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
        losses.update(loss.item(), tx.size(0))

        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if (i + 1) % batches_per_step == 0:
            optimizer.step()
            step += 1
            update_lr(
                optimizer,
                new_lr=get_lr(step=step, n_emb=n_emb, warmup_steps=warmup_steps),
            )

            step_time.update(time.time() - start_step_time)

            start_step_time = time.time()

            if (epoch in [epochs - 1, epochs - 2]) and (step % 1500 == 0):
                save_checkpoint(epoch, model, optimizer, prefix=f"step{step}_")

        start_data_time = time.time()

    return losses
