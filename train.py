"""Training entry point for the ViT-GPT2 fine-tuning approach.

Standalone:
    python train.py

Under a W&B sweep agent:
    wandb sweep sweep.yaml      # one-off, prints a sweep id
    wandb agent <sweep_id>      # spawns runs, each one calling this script

The ViT encoder is frozen by default; only the GPT-2 decoder is fine-tuned.
Set `unfreeze_encoder_epoch` in config to progressively unfreeze the encoder.
"""
import argparse
import collections
import os
import random

import torch
import torch.nn as nn
import torch.utils.data
import wandb
import numpy as np
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torchvision.transforms import v2
from transformers import (
    AutoTokenizer,
    CLIPModel, CLIPProcessor,
    VisionEncoderDecoderModel, ViTImageProcessor,
)
from tqdm.auto import tqdm

from dataloaders.dataset import Flickr8kDataset


DEFAULT_CONFIG = dict(
    epochs=40,
    batch_size=32,
    lr=0.00004652,
    weight_decay=0.00181,
    label_smoothing=0.07418,
    early_stop_patience=5,
    max_len=30,
    unfreeze_encoder_epoch=5,  # set < epochs to progressively unfreeze ViT
    warmup_epochs=1,
    decoder_dropout=0.05,
    frozen_decoder_layers=8,
    llrd_factor=0.85,             # 1.0 = flat lr; <1 = lower lr for earlier layers
    accum_steps=1,               # gradient accumulation steps
    # Gender-debias: up-weight female-related tokens in CE loss (1.0 = off).
    female_token_weight=2.5,
    data_dir="data/flickr8k",
    checkpoint_dir="checkpoints",
    num_workers=2,
    seed=42,
)


_FEMALE_WORDS = [
    "woman", "women", "girl", "girls", "she", "her", "female",
    "lady", "ladies", "mother", "mom",
]


def build_gender_class_weights(tokenizer, vocab_size, female_weight):
    """Return (weights, matched_ids) for CrossEntropyLoss weighting.

    GPT-2 BPE encodes a word differently depending on whether it has a leading
    space. We try several casing/spacing variants and keep only single-token
    encodings to avoid up-weighting sub-word fragments shared with other words.
    """
    weights = torch.ones(vocab_size)
    matched = set()
    for w in _FEMALE_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                matched.add(ids[0])
    for tid in matched:
        weights[tid] = female_weight
    return weights, sorted(matched)

# ViTImageProcessor uses mean=std=0.5 (different from ImageNet)
_VIT_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_VIT_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


class _ViTTransform:
    """Wrap ViTImageProcessor as a callable compatible with Flickr8kDataset."""

    def __init__(self, augment=False):
        self.processor = ViTImageProcessor.from_pretrained(_MODEL_NAME)
        self.augment = v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.RandomRotation(degrees=15),
            v2.RandomGrayscale(p=0.1),
        ]) if augment else None
        self.erasing = v2.RandomErasing(p=0.25, scale=(0.02, 0.2)) if augment else None

    def __call__(self, img):
        if self.augment is not None:
            img = self.augment(img)
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        if self.erasing is not None:
            pixel_values = self.erasing(pixel_values)
        return pixel_values


class _CLIPScorer:
    """Batch-wise CLIP score accumulator backed by transformers directly."""
    _MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(self._MODEL_NAME).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(self._MODEL_NAME)
        self._total = 0.0
        self._n = 0

    def reset(self):
        self._total = 0.0
        self._n = 0

    @torch.no_grad()
    def update(self, images_uint8, captions):
        imgs = [img.permute(1, 2, 0).numpy() for img in images_uint8.cpu()]
        inputs = self.processor(
            text=captions, images=imgs,
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        ).to(self.device)
        out = self.model(**inputs)
        img_f = F.normalize(out.image_embeds, dim=-1)
        txt_f = F.normalize(out.text_embeds, dim=-1)
        scores = (img_f * txt_f).sum(dim=-1) * 100.0
        self._total += scores.sum().item()
        self._n += len(captions)

    def compute(self):
        return self._total / self._n if self._n > 0 else 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def build_dataloaders(config, tokenizer):
    captions_file = os.path.join(config.data_dir, "captions.txt")
    common_ds_kwargs = dict(
        image_dir=os.path.join(config.data_dir, "Images"),
        captions_file=captions_file,
        tokenizer=tokenizer,
        max_length=config.max_len,
    )
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    train_dataset = Flickr8kDataset(**common_ds_kwargs, split="train", transform=_ViTTransform(augment=True))
    val_dataset   = Flickr8kDataset(**common_ds_kwargs, split="val",   transform=_ViTTransform(augment=False))
    train_loader  = torch.utils.data.DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader    = torch.utils.data.DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    return train_dataset, val_dataset, train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, scaler, device, pad_token_id, accum_steps=1):
    model.train()
    if not next(model.encoder.parameters()).requires_grad:
        model.encoder.eval()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (images, token_ids, _) in enumerate(tqdm(loader, leave=False, desc="Train")):
        images    = images.to(device)
        token_ids = token_ids.to(device)

        dec_input     = token_ids[:, :-1]
        dec_attn_mask = (dec_input != pad_token_id).long()
        labels        = token_ids[:, 1:].clone()
        labels[labels == pad_token_id] = -100

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(
                pixel_values=images,
                decoder_input_ids=dec_input,
                decoder_attention_mask=dec_attn_mask,
            ).logits
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
    return total_loss / len(loader)


@torch.no_grad()
def _generate_captions(model, images, tokenizer, max_len, num_beams=1):
    generated_ids = model.generate(
        pixel_values=images,
        max_length=max_len,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=(num_beams > 1),
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def build_references_dict(captions_file):
    """Return {image_name: [cap1, cap2, ...]} from captions.txt (all splits)."""
    refs = collections.defaultdict(list)
    with open(captions_file) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                refs[parts[0]].append(parts[1].lower())
    return dict(refs)


def _unnormalize(images_cpu):
    return ((images_cpu * _VIT_STD + _VIT_MEAN).clamp(0, 1) * 255).to(torch.uint8)


def evaluate(model, loader, criterion, device, tokenizer, clip_metric, pad_token_id, max_len,
             clip_batches=50, references_dict=None):
    model.eval()
    total_loss = 0.0
    clip_metric.reset()
    image_captions = {}  # {image_name: generated_caption} for BLEU, deduped by image

    with torch.no_grad():
        for i, (images, token_ids, image_names) in enumerate(tqdm(loader, leave=False, desc="Val")):
            images    = images.to(device)
            token_ids = token_ids.to(device)

            dec_input     = token_ids[:, :-1]
            dec_attn_mask = (dec_input != pad_token_id).long()
            labels        = token_ids[:, 1:].clone()
            labels[labels == pad_token_id] = -100

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(
                    pixel_values=images,
                    decoder_input_ids=dec_input,
                    decoder_attention_mask=dec_attn_mask,
                ).logits
                total_loss += criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).item()

            if i < clip_batches:
                captions = _generate_captions(model, images, tokenizer, max_len)
                clip_metric.update(_unnormalize(images.cpu()), captions)
                if references_dict is not None:
                    for name, cap in zip(image_names, captions):
                        image_captions.setdefault(name, cap)

    bleu_scores = (0.0, 0.0, 0.0, 0.0)
    if references_dict is not None and image_captions:
        names = list(image_captions.keys())
        hyps  = [image_captions[n].split() for n in names]
        refs  = [[ref.split() for ref in references_dict.get(n, [])] for n in names]
        smooth = SmoothingFunction().method1
        bleu_scores = (
            corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),                smoothing_function=smooth),
            corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0),            smoothing_function=smooth),
            corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3, 0),          smoothing_function=smooth),
            corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25),    smoothing_function=smooth),
        )

    return total_loss / len(loader), clip_metric.compute(), bleu_scores


def clip_score_subset(model, loader, tokenizer, device, clip_metric, max_len, n_batches=20):
    model.eval()
    clip_metric.reset()

    with torch.no_grad():
        for i, (images, _, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            generated_ids = model.generate(
                pixel_values=images,
                max_length=max_len,
                num_beams=1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
            )
            captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            clip_metric.update(_unnormalize(images.cpu()), captions)

    return clip_metric.compute()


def save_checkpoint(model, optimizer, scheduler, epoch, val_metric, path):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    # Exclude frozen encoder to keep checkpoint size small (~500 MB vs ~1 GB)
    decoder_state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save({
        "epoch": epoch,
        "model_state_dict": decoder_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_metric": val_metric,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    # strict=False: checkpoint only has decoder keys; encoder reloaded from HuggingFace
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_metric", float("-inf"))


def build_optimizer(model, base_lr, weight_decay, llrd_factor=1.0, encoder_frozen=True):
    """AdamW with optional layer-wise learning rate decay (LLRD).

    Each GPT-2 decoder block i gets lr = base_lr * llrd_factor^(n_blocks - 1 - i),
    so earlier blocks train more slowly. llrd_factor=1.0 gives a flat lr.
    When encoder_frozen=False, encoder layers get base_lr*0.1 with the same decay.
    """
    seen = set()
    param_groups = []

    def add_group(params, lr):
        unique = [p for p in params if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in unique)
        if unique:
            param_groups.append({"params": unique, "lr": lr})

    dec = model.decoder.transformer
    n = len(dec.h)

    add_group(list(dec.wte.parameters()) + list(dec.wpe.parameters()), base_lr * (llrd_factor ** n))
    for i, block in enumerate(dec.h):
        add_group(list(block.parameters()), base_lr * (llrd_factor ** (n - 1 - i)))
    add_group(list(dec.ln_f.parameters()) + list(model.decoder.lm_head.parameters()), base_lr)

    if not encoder_frozen:
        enc = model.encoder
        enc_base = base_lr * 0.1
        n_enc = len(enc.encoder.layer)
        add_group(list(enc.embeddings.parameters()), enc_base * (llrd_factor ** n_enc))
        for i, layer in enumerate(enc.encoder.layer):
            add_group(list(layer.parameters()), enc_base * (llrd_factor ** (n_enc - 1 - i)))
        add_group(list(enc.layernorm.parameters()), enc_base)

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    p.add_argument("--entity", default="xnap-image-captioning", help="W&B entity (team)")
    p.add_argument("--project", default="image-captioning", help="W&B project name")
    p.add_argument("--run-name", default=None, help="Optional W&B run name")
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()

    wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        config=DEFAULT_CONFIG,
        resume="allow" if args.resume else None,
    )
    config = wandb.config

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = build_tokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")

    clip_metric = _CLIPScorer(device)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config, tokenizer)
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    captions_file = os.path.join(config.data_dir, "captions.txt")
    references_dict = build_references_dict(captions_file)

    model = VisionEncoderDecoderModel.from_pretrained(_MODEL_NAME)
    model.decoder.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id           = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id           = tokenizer.eos_token_id

    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    decoder_dropout = getattr(config, "decoder_dropout", 0.1)
    for module in model.decoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = decoder_dropout

    n_freeze = getattr(config, "frozen_decoder_layers", 0)
    for i in range(n_freeze):
        for p in model.decoder.transformer.h[i].parameters():
            p.requires_grad = False

    model = model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable")
    wandb.config.update({"num_params/total": n_total, "num_params/trainable": n_trainable})

    scaler    = torch.amp.GradScaler(device.type)
    pad_idx   = tokenizer.pad_token_id

    female_weight = getattr(config, "female_token_weight", 1.0)
    ce_weight = None
    if female_weight and female_weight != 1.0:
        ce_weight, matched_ids = build_gender_class_weights(tokenizer, len(tokenizer), female_weight)
        ce_weight = ce_weight.to(device)
        print(f"Gender-debias: weighting {len(matched_ids)} female-related token IDs by {female_weight}x")
        wandb.config.update({"gender_debias/n_female_tokens": len(matched_ids),
                             "gender_debias/female_weight": female_weight})
    criterion = nn.CrossEntropyLoss(
        weight=ce_weight, ignore_index=-100, label_smoothing=config.label_smoothing
    )

    llrd = getattr(config, "llrd_factor", 1.0)
    optimizer = build_optimizer(model, config.lr, config.weight_decay, llrd_factor=llrd, encoder_frozen=True)
    warmup = getattr(config, "warmup_epochs", 1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs - warmup, 1)),
        ],
        milestones=[warmup],
    )

    wandb.watch(model.decoder, log="gradients", log_freq=100)

    start_epoch = 0
    best_val_metric = float("-inf")
    epochs_since_improvement = 0
    if args.resume:
        start_epoch, best_val_metric = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, best_val_metric={best_val_metric:.4f}")

    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")

    for epoch in tqdm(range(start_epoch, config.epochs), desc="Epochs"):
        if epoch == config.unfreeze_encoder_epoch:
            for p in model.encoder.parameters():
                p.requires_grad = True
            unfreeze_lr = config.lr * 0.1
            optimizer = build_optimizer(model, unfreeze_lr, config.weight_decay, llrd_factor=llrd, encoder_frozen=False)
            remaining = max(config.epochs - epoch, 1)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=1),
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining - 1, 1)),
                ],
                milestones=[1],
            )
            print(f"Epoch {epoch}: ViT encoder unfrozen, lr={unfreeze_lr:.2e}")

        accum = getattr(config, "accum_steps", 1)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, pad_idx, accum_steps=accum)
        val_loss, val_cs, (val_bleu1, val_bleu2, val_bleu3, val_bleu4) = evaluate(
            model, val_loader, criterion, device, tokenizer, clip_metric, pad_idx, config.max_len,
            references_dict=references_dict,
        )
        train_cs = clip_score_subset(
            model, train_loader, tokenizer, device, clip_metric, config.max_len, n_batches=20
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/clip_score": train_cs,
            "val/clip_score": val_cs,
            "val/bleu1": val_bleu1,
            "val/bleu2": val_bleu2,
            "val/bleu3": val_bleu3,
            "val/bleu4": val_bleu4,
            "lr": current_lr,
        })
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_clip={train_cs:.2f}  val_clip={val_cs:.2f}  "
            f"bleu1={val_bleu1:.4f}  bleu4={val_bleu4:.4f}  lr={current_lr:.2e}"
        )

        if val_cs > best_val_metric:
            best_val_metric = val_cs
            epochs_since_improvement = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_cs, checkpoint_path)
            print(f"  -> Checkpoint saved (epoch {epoch}, val_clip_score={val_cs:.2f})")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.early_stop_patience:
                print(f"Early stopping: no val/clip_score improvement in {config.early_stop_patience} epochs.")
                break

    wandb.finish()


if __name__ == "__main__":
    main()
