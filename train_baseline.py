"""Training entry point for the ResNet18+GRU baseline.

Standalone:
    python train_baseline.py

Metrics are logged to Weights & Biases (entity/project via CLI flags).
"""
import argparse
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import wandb
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm.auto import tqdm

from models.baseline import Baseline
from utils.vocab import GenerationVocab
from torch.utils.data import Dataset


class Flickr8kBaselineDataset(Dataset):
    _SPLITS = {"train": (0.0, 0.75), "val": (0.75, 0.875), "test": (0.875, 1.0)}

    def __init__(self, image_dir, captions_file, vocab, max_length, split="train", transform=None):
        from PIL import Image as _Image
        from torchvision.transforms import v2 as _v2
        self._Image = _Image
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform or _v2.Compose([
            _v2.Resize((224, 224)),
            _v2.ToImage(),
            _v2.ToDtype(torch.float32, scale=True),
            _v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        df = pd.read_csv(captions_file)
        df.columns = ["image", "caption"]
        unique_images = sorted(df["image"].unique())
        n = len(unique_images)
        lo, hi = self._SPLITS[split]
        selected = set(unique_images[int(n * lo):int(n * hi)])
        df = df[df["image"].isin(selected)].reset_index(drop=True)
        self.data = list(zip(df["image"].tolist(), df["caption"].tolist()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]
        image = self._Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        image = self.transform(image)
        token_ids = self.vocab.prepare(caption.lower().split(), self.max_length)
        return image, torch.tensor(token_ids, dtype=torch.long), image_name


DEFAULT_CONFIG = dict(
    epochs=20,
    batch_size=32,
    lr=0.0002,
    hidden_dim=256,
    char_dim=512,
    num_layers=2,
    dropout=0.3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    scheduler_factor=0.5,
    scheduler_patience=2,
    early_stop_patience=5,
    max_len=30,
    teacher_forcing=0.5,
    data_dir="data/flickr8k",
    checkpoint_dir="checkpoints",
    min_word_freq=2,
    num_workers=2,
    seed=42,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_vocab(captions_file, min_freq):
    df = pd.read_csv(captions_file)
    df.columns = ["image", "caption"]
    unique_images = sorted(df["image"].unique())
    train_images = set(unique_images[: int(len(unique_images) * 0.75)])
    train_df = df[df["image"].isin(train_images)]
    word_counts = Counter(
        word
        for caption in train_df["caption"].str.lower()
        for word in caption.split()
    )
    words = sorted(w for w, c in word_counts.items() if c >= min_freq)
    return GenerationVocab(words)


def build_dataloaders(config, vocab):
    captions_file = os.path.join(config.data_dir, "captions.txt")
    common_ds_kwargs = dict(
        image_dir=os.path.join(config.data_dir, "Images"),
        captions_file=captions_file,
        vocab=vocab,
        max_length=config.max_len,
    )
    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    train_dataset = Flickr8kBaselineDataset(**common_ds_kwargs, split="train")
    val_dataset   = Flickr8kBaselineDataset(**common_ds_kwargs, split="val")
    train_loader  = torch.utils.data.DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader    = torch.utils.data.DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    return train_dataset, val_dataset, train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, teacher_forcing):
    model.train()
    total_loss = 0.0
    for images, token_ids, _ in tqdm(loader, leave=False, desc="Train"):
        images, token_ids = images.to(device), token_ids.to(device)
        outputs = model(images, token_ids, teacher_forcing=teacher_forcing)
        loss = criterion(
            outputs[:, :-1].reshape(-1, outputs.size(-1)),
            token_ids[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _decode(token_ids, vocab, stop_idx, pad_idx):
    skip = {vocab.token2index[vocab.START_TAG], vocab.token2index[vocab.UNKNOWN_TAG]}
    words = []
    for tok in token_ids:
        if tok in (stop_idx, pad_idx):
            break
        if tok not in skip:
            words.append(vocab.index2token.get(tok, ""))
    return words


def _generate(model, images, vocab, device):
    pad_idx = vocab.token2index[vocab.PAD_TAG]
    unk_idx = vocab.token2index[vocab.UNKNOWN_TAG]
    dummy = torch.full((images.size(0), model.text_max_len), pad_idx, device=device)
    dummy[:, 0] = vocab.token2index[vocab.START_TAG]
    with torch.no_grad():
        logits = model(images, dummy, teacher_forcing=0.0)
        logits[..., unk_idx] = float("-inf")
        return logits.argmax(dim=-1)


def evaluate(model, loader, criterion, device, vocab):
    model.eval()
    total_loss = 0.0
    refs_by_image, hyp_by_image = {}, {}
    stop_idx = vocab.token2index[vocab.STOP_TAG]
    pad_idx  = vocab.token2index[vocab.PAD_TAG]

    with torch.no_grad():
        for images, token_ids, image_names in tqdm(loader, leave=False, desc="Val"):
            images, token_ids = images.to(device), token_ids.to(device)
            outputs = model(images, token_ids)
            loss = criterion(
                outputs[:, :-1].reshape(-1, outputs.size(-1)),
                token_ids[:, 1:].reshape(-1),
            )
            total_loss += loss.item()
            pred_ids = _generate(model, images, vocab, device).cpu().tolist()
            for i, name in enumerate(image_names):
                ref = _decode(token_ids[i].cpu().tolist(), vocab, stop_idx, pad_idx)
                refs_by_image.setdefault(name, []).append(ref)
                if name not in hyp_by_image:
                    hyp_by_image[name] = _decode(pred_ids[i], vocab, stop_idx, pad_idx)

    names = list(hyp_by_image.keys())
    refs  = [refs_by_image[n] for n in names]
    hyps  = [hyp_by_image[n]  for n in names]
    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(refs, hyps, smoothing_function=smooth)
    return total_loss / len(loader), bleu


def bleu_subset(model, loader, vocab, device, n_batches=10):
    model.eval()
    refs_by_image, hyp_by_image = {}, {}
    stop_idx = vocab.token2index[vocab.STOP_TAG]
    pad_idx  = vocab.token2index[vocab.PAD_TAG]

    with torch.no_grad():
        for i, (images, token_ids, image_names) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            pred_ids = _generate(model, images, vocab, device).cpu().tolist()
            for j, name in enumerate(image_names):
                ref = _decode(token_ids[j].tolist(), vocab, stop_idx, pad_idx)
                refs_by_image.setdefault(name, []).append(ref)
                if name not in hyp_by_image:
                    hyp_by_image[name] = _decode(pred_ids[j], vocab, stop_idx, pad_idx)

    names = list(hyp_by_image.keys())
    smooth = SmoothingFunction().method1
    return corpus_bleu([refs_by_image[n] for n in names], [hyp_by_image[n] for n in names],
                       smoothing_function=smooth)


def save_checkpoint(model, optimizer, scheduler, epoch, val_bleu, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_bleu": val_bleu,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("val_bleu", float("-inf"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", default=None)
    p.add_argument("--entity",   default="xnap-image-captioning")
    p.add_argument("--project",  default="image-captioning")
    p.add_argument("--run-name", default=None)
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    wandb.init(
        entity=args.entity, project=args.project, name=args.run_name,
        config=DEFAULT_CONFIG, resume="allow" if args.resume else None,
    )
    config = wandb.config
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    captions_file = os.path.join(config.data_dir, "captions.txt")
    vocab = build_vocab(captions_file, config.min_word_freq)
    print(f"Vocabulary size: {len(vocab)}")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config, vocab)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model = Baseline(
        num_char=len(vocab),
        hidden_dimension=config.hidden_dim,
        char_dimension=config.char_dim,
        text_max_len=config.max_len,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable")
    wandb.config.update({"num_params/total": n_total, "num_params/trainable": n_trainable})

    pad_idx   = vocab.token2index[vocab.PAD_TAG]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience,
    )
    wandb.watch(model, log="gradients", log_freq=100)

    start_epoch = 0
    best_val_bleu = float("-inf")
    epochs_since_improvement = 0
    if args.resume:
        start_epoch, best_val_bleu = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1

    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model_baseline.pt")

    for epoch in tqdm(range(start_epoch, config.epochs), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config.teacher_forcing)
        val_loss, val_bleu = evaluate(model, val_loader, criterion, device, vocab)
        train_bleu = bleu_subset(model, train_loader, vocab, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss,
                   "train/bleu": train_bleu, "val/bleu": val_bleu, "lr": current_lr})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_bleu={train_bleu:.4f}  val_bleu={val_bleu:.4f}  lr={current_lr:.2e}")

        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            epochs_since_improvement = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_bleu, checkpoint_path)
            print(f"  -> Checkpoint saved (epoch {epoch}, val_bleu={val_bleu:.4f})")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.early_stop_patience:
                print(f"Early stopping after {config.early_stop_patience} epochs without improvement.")
                break

    wandb.finish()


if __name__ == "__main__":
    main()
