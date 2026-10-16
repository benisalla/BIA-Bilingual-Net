"""
Fine-tune a trained BIALinguaNet on new (or the same) data with a smaller learning rate.

    python -m translation_engine.finetune --direction dr-en --data-dir path/to/new_data

It starts from the trained checkpoint (default translation_engine/src/checkpoints/<src>_<tgt>_chpts.pth.tar),
continues the learning-rate schedule where training stopped, scaled by `finetune_lr_scale` (0.1),
and saves the best fine-tuned model to a separate file so the original is never overwritten.
"""

import argparse
import os

import torch

from translation_engine.core.config import Config
from translation_engine.data.dataloader import DataLoader, build_tokenizer
from translation_engine.model.LSRCrossEntropy import LSRCrossEntropy
from translation_engine.model.utils import fit, load_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=["dr-en", "en-dr"], default="dr-en")
    parser.add_argument("--data-dir", default=None, help="folder with train.json / val.json")
    parser.add_argument("--pretrained", default=None, help="checkpoint to start from")
    parser.add_argument("--output", default=None, help="where to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=None, help="number of fine-tuning epochs")
    parser.add_argument("--lr-scale", type=float, default=None, help="multiplier on the learning rate")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    s_suffix, t_suffix = args.direction.split("-")

    # Build configs
    overrides = {"s_suffix": s_suffix, "t_suffix": t_suffix}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.epochs:
        overrides["finetune_epochs"] = args.epochs
    if args.lr_scale:
        overrides["finetune_lr_scale"] = args.lr_scale
    config = Config(**overrides)

    pretrained_model_path = args.pretrained or config.checkpoints_path
    output_path = args.output or pretrained_model_path.replace("_chpts", "_finetuned_chpts")
    if not os.path.exists(pretrained_model_path):
        raise SystemExit(f"No pre-trained model found at {pretrained_model_path}. Train one first.")

    # Initialize tokenizer
    tokenizer = build_tokenizer(config.tokenizer_name)
    config.set_vocab_size(tokenizer=tokenizer)

    # Load the pre-trained model; its own init args (sizes) are used, not the config's
    model, _, start_epoch = load_checkpoint(file_path=pretrained_model_path, device=config.device)
    print(f"Pre-trained model loaded from {pretrained_model_path} (trained {start_epoch} epochs) ...")
    config.max_seq_len = model.max_seq_len
    config.n_emb = model.n_emb  # the lr schedule depends on it

    # Initialize data-loaders (validation uses the same direction as training)
    loader_args = dict(
        data_dir=config.data_dir,
        s_suffix=config.s_suffix,
        t_suffix=config.t_suffix,
        toks_in_batch=config.toks_in_batch,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
    )
    train_loader = DataLoader(split="train", **loader_args)
    val_loader = DataLoader(split="val", **loader_args)

    # Fresh optimizer: fine-tuning data may differ, old Adam moments would not fit it
    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.lr * config.finetune_lr_scale,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    # Loss fun
    criterion = LSRCrossEntropy(eps=config.label_smoothing, device=config.device)

    model = model.to(config.device)
    criterion = criterion.to(config.device)

    print("Fine-tuning started ...")
    return fit(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        epochs=start_epoch + config.finetune_epochs,
        lr_scale=config.finetune_lr_scale,
        save_path=output_path,
        plot_path=os.path.splitext(output_path)[0].replace(".pth", "") + "_loss.png",
    )


if __name__ == "__main__":
    main()
