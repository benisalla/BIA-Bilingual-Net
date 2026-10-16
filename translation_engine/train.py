"""
Train BIALinguaNet for one translation direction.

    python -m translation_engine.train                      # Darija -> English, default config
    python -m translation_engine.train --direction en-dr    # English -> Darija
    python -m translation_engine.train --overfit 32         # sanity check: memorise 32 pairs

The best model (lowest validation loss) is saved to
translation_engine/src/checkpoints/<src>_<tgt>_chpts.pth.tar, the file the app loads.
If that file exists, training resumes from it (use --no-resume to start over).
"""

import argparse
import os

import torch

from translation_engine.core.config import Config
from translation_engine.data.dataloader import DataLoader, build_tokenizer
from translation_engine.model.LSRCrossEntropy import LSRCrossEntropy
from translation_engine.model.utils import build_model, fit, load_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=["dr-en", "en-dr"], default="dr-en")
    parser.add_argument("--data-dir", default=None, help="folder with train.json / val.json")
    parser.add_argument("--checkpoint", default=None, help="where to save the best model")
    parser.add_argument("--n-steps", type=int, default=None, help="total optimizer steps")
    parser.add_argument("--epochs", type=int, default=None, help="overrides --n-steps")
    parser.add_argument("--toks-in-batch", type=int, default=None)
    parser.add_argument("--batches-per-step", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        metavar="N",
        help="sanity check: train and validate on the first N training pairs, "
        "no dropout, no label smoothing; the loss should go close to 0",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def config_from_args(args) -> Config:
    s_suffix, t_suffix = args.direction.split("-")
    overrides = {"s_suffix": s_suffix, "t_suffix": t_suffix}
    for name in ("data_dir", "n_steps", "epochs", "toks_in_batch", "batches_per_step", "warmup_steps"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.checkpoint:
        overrides["checkpoints_path"] = args.checkpoint
    if args.overfit:
        overrides.setdefault("checkpoints_path", f"./translation_engine/src/checkpoints/overfit_{s_suffix}_{t_suffix}.pth.tar")
        overrides.update(d_rate=0.0, label_smoothing=0.0)
        overrides.setdefault("batches_per_step", 1)
        overrides.setdefault("warmup_steps", 400)
        overrides.setdefault("epochs", 60)
    return Config(**overrides)


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    # Build configs
    config = config_from_args(args)

    # init tokenizer
    tokenizer = build_tokenizer(config.tokenizer_name)
    config.set_vocab_size(tokenizer=tokenizer)
    print(config)

    # Initialize data-loaders (validation uses the same direction as training)
    loader_args = dict(
        data_dir=config.data_dir,
        s_suffix=config.s_suffix,
        t_suffix=config.t_suffix,
        toks_in_batch=config.toks_in_batch,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
    )
    if args.overfit:
        train_loader = DataLoader(split="train", limit=args.overfit, **loader_args)
        val_loader = DataLoader(split="train", limit=args.overfit, **loader_args)
    else:
        train_loader = DataLoader(split="train", **loader_args)
        val_loader = DataLoader(split="val", **loader_args)
    print(f"train: {len(train_loader.data):,} pairs in {train_loader.n_batches:,} batches "
          f"({train_loader.n_dropped} dropped) | val: {len(val_loader.data):,} pairs")

    # Initialize model or load checkpoint
    model = build_model(config)
    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    resume = not args.no_resume and not args.overfit  # an overfit check always starts from scratch
    if resume and os.path.exists(config.checkpoints_path):
        model, optimizer, config.start_epoch = load_checkpoint(
            file_path=config.checkpoints_path,
            device=config.device,
            beta1=config.betas[0],
            beta2=config.betas[1],
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        print(f"Model loaded from checkpoint {config.checkpoints_path} (epoch {config.start_epoch}) ...")
    else:
        print("Model loaded from scratch ...")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    criterion = LSRCrossEntropy(eps=config.label_smoothing, device=config.device)

    # Move to default device
    model = model.to(config.device)
    criterion = criterion.to(config.device)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # Find total epochs to train
    steps_per_epoch = max(1, train_loader.n_batches // config.batches_per_step)
    epochs = config.epochs or (config.n_steps // steps_per_epoch) + 1

    print("training started ...")
    stats = fit(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=config.start_epoch,
        epochs=epochs,
        save_path=config.checkpoints_path,
        plot_path=os.path.splitext(config.checkpoints_path)[0].replace(".pth", "") + "_loss.png",
    )
    return stats


if __name__ == "__main__":
    main()
