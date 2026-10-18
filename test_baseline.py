"""Evaluate the ResNet18+GRU baseline with CLIP score + BLEU-1/2/3/4.

Usage:
    python test_baseline.py --checkpoint checkpoints/best_model_baseline.pt
"""
import argparse
import collections
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from models.baseline import Baseline
from utils.vocab import GenerationVocab
from torch.utils.data import Dataset


class Flickr8kBaselineDataset(Dataset):
    _SPLITS = {"train": (0.0, 0.75), "val": (0.75, 0.875), "test": (0.875, 1.0)}

    def __init__(self, image_dir, captions_file, vocab, max_length, split="train", transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform or v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        image = self.transform(image)
        token_ids = self.vocab.prepare(caption.lower().split(), self.max_length)
        return image, torch.tensor(token_ids, dtype=torch.long), image_name

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _CLIPScorer:
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


def _unnormalize(images_cpu):
    return ((images_cpu * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1) * 255).to(torch.uint8)


def build_vocab(captions_file, min_freq=2):
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


def build_references_dict(captions_file):
    """Return {image_name: [cap1, cap2, ...]} with all 5 references per image."""
    refs = collections.defaultdict(list)
    with open(captions_file) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                refs[parts[0]].append(parts[1].lower())
    return dict(refs)


def load_model(checkpoint_path, vocab, hidden_dim, char_dim, max_len, num_layers, dropout, device):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at '{checkpoint_path}'.")
        print("Train first: python train_baseline.py")
        sys.exit(1)
    model = Baseline(
        num_char=len(vocab),
        hidden_dimension=hidden_dim,
        char_dimension=char_dim,
        text_max_len=max_len,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, val_bleu={ckpt.get('val_bleu', '?')}")
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable")
    return model


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


def evaluate_test(model, test_loader, criterion, vocab, device, references_dict):
    model.eval()
    total_loss = 0.0
    clip_metric = _CLIPScorer(device)
    stop_idx = vocab.token2index[vocab.STOP_TAG]
    pad_idx  = vocab.token2index[vocab.PAD_TAG]
    image_captions = {}  # {image_name: generated_caption string}

    with torch.no_grad():
        for images, token_ids, image_names in tqdm(test_loader, desc="Evaluating"):
            images, token_ids = images.to(device), token_ids.to(device)
            outputs = model(images, token_ids)
            loss = criterion(
                outputs[:, :-1].reshape(-1, outputs.size(-1)),
                token_ids[:, 1:].reshape(-1),
            )
            total_loss += loss.item()

            pred_ids = _generate(model, images, vocab, device).cpu().tolist()
            captions = [
                " ".join(_decode(pred_ids[i], vocab, stop_idx, pad_idx))
                for i in range(len(image_names))
            ]
            clip_metric.update(_unnormalize(images.cpu()), captions)
            for name, cap in zip(image_names, captions):
                image_captions.setdefault(name, cap)

    names = list(image_captions.keys())
    hyps  = [image_captions[n].split() for n in names]
    refs  = [[ref.split() for ref in references_dict.get(n, [])] for n in names]
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),             smoothing_function=smooth)
    b2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0),         smoothing_function=smooth)
    b3 = corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3, 0),       smoothing_function=smooth)
    b4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return total_loss / len(test_loader), clip_metric.compute(), b1, b2, b3, b4


def show_samples(model, dataset, vocab, device, n):
    model.eval()
    stop_idx = vocab.token2index[vocab.STOP_TAG]
    pad_idx  = vocab.token2index[vocab.PAD_TAG]
    indices  = random.sample(range(len(dataset)), min(n, len(dataset)))

    print("\n" + "=" * 72)
    print(f"{'SAMPLE PREDICTIONS':^72}")
    print("=" * 72)
    for idx in indices:
        image_name, reference = dataset.data[idx]
        image_tensor, _ = dataset[idx]
        pred_ids = _generate(model, image_tensor.unsqueeze(0).to(device), vocab, device)
        predicted = " ".join(_decode(pred_ids[0].cpu().tolist(), vocab, stop_idx, pad_idx))
        print(f"\n  Image    : {image_name}")
        print(f"  Predicted: {predicted}")
        print(f"  Reference: {reference}")
    print("\n" + "=" * 72)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_model_baseline.pt")
    parser.add_argument("--data_dir",   default="data/flickr8k")
    parser.add_argument("--n_samples",  type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--char_dim",   type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout",    type=float, default=0.0)
    parser.add_argument("--max_len",    type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_word_freq", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    captions_file   = os.path.join(args.data_dir, "captions.txt")
    vocab           = build_vocab(captions_file, args.min_word_freq)
    references_dict = build_references_dict(captions_file)
    print(f"Vocabulary size: {len(vocab)}")

    model = load_model(args.checkpoint, vocab, args.hidden_dim, args.char_dim,
                       args.max_len, args.num_layers, args.dropout, device)

    test_dataset = Flickr8kBaselineDataset(
        image_dir=os.path.join(args.data_dir, "Images"),
        captions_file=captions_file,
        vocab=vocab,
        max_length=args.max_len,
        split="test",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"Test set: {len(test_dataset)} samples")

    pad_idx   = vocab.token2index[vocab.PAD_TAG]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    test_loss, test_cs, b1, b2, b3, b4 = evaluate_test(
        model, test_loader, criterion, vocab, device, references_dict,
    )
    print(f"\n{'Test loss':>15}: {test_loss:.4f}")
    print(f"{'Test CLIPScore':>15}: {test_cs:.2f}")
    print(f"{'Test BLEU-1':>15}: {b1:.4f}")
    print(f"{'Test BLEU-2':>15}: {b2:.4f}")
    print(f"{'Test BLEU-3':>15}: {b3:.4f}")
    print(f"{'Test BLEU-4':>15}: {b4:.4f}")

    show_samples(model, test_dataset, vocab, device, args.n_samples)


if __name__ == "__main__":
    main()
