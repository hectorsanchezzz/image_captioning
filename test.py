"""Evaluate the fine-tuned ViT-GPT2 image captioning model on Flickr8k test split.

Standalone:
    python test.py
    python test.py --seed 42 --save_predictions outputs/test_predictions.csv
    python test.py --image path/to/photo.jpg

Notebook:
    See test.ipynb (imports functions from this module).
"""
import argparse
import collections
import csv
import os
import random
import sys

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)
from PIL import Image
from tqdm import tqdm

from dataloaders.dataset import Flickr8kDataset

_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

_VIT_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_VIT_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

DEFAULT_CONFIG = dict(
    checkpoint="checkpoints/best_model.pt",
    data_dir="data/flickr8k",
    max_len=30,
    batch_size=32,
    num_beams=4,
    n_samples=5,
    seed=42,
)


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
            text=captions,
            images=imgs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        out = self.model(**inputs)
        img_f = F.normalize(out.image_embeds, dim=-1)
        txt_f = F.normalize(out.text_embeds, dim=-1)
        scores = (img_f * txt_f).sum(dim=-1) * 100.0
        self._total += scores.sum().item()
        self._n += len(captions)

    def compute(self):
        return self._total / self._n if self._n > 0 else 0.0


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def load_model(checkpoint_path, tokenizer, device, verbose=True, zero_shot=None, return_info=None):
    """Load ViT-GPT2 and optional fine-tuned decoder weights."""
    legacy_tuple_return = zero_shot is None and return_info is None
    zero_shot = False if zero_shot is None else zero_shot
    return_info = legacy_tuple_return if return_info is None else return_info

    model = VisionEncoderDecoderModel.from_pretrained(_MODEL_NAME)
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if zero_shot:
        print("Zero-shot mode: using pretrained weights without fine-tuning.")
        model = model.eval().to(device)
        if return_info:
            return model, {"epoch": None, "val_clip_score": None, "path": None}
        return model

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Train first with train.py to produce checkpoints/best_model.pt"
        )
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint at '{checkpoint_path}' is corrupted (incomplete write). "
            "Delete it and re-run train.py."
        ) from exc

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    info = {
        "epoch": ckpt.get("epoch"),
        "val_clip_score": ckpt.get("val_metric"),
        "path": checkpoint_path,
    }
    if verbose:
        epoch = info["epoch"] if info["epoch"] is not None else "?"
        if info["val_clip_score"] is not None:
            print(f"Loaded checkpoint: epoch={epoch}, val_clip_score={info['val_clip_score']:.4f}")
        else:
            print(f"Loaded checkpoint: epoch={epoch}")
    model = model.to(device)
    if return_info:
        return model, info
    return model


class ViTTransform:
    """ViTImageProcessor wrapper (shared by script, notebook, and dataset)."""

    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained(_MODEL_NAME)

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)


# Backward-compatible alias used internally
_ViTTransform = ViTTransform


def _unnormalize(images_cpu):
    return ((images_cpu * _VIT_STD + _VIT_MEAN).clamp(0, 1) * 255).to(torch.uint8)


@torch.no_grad()
def _generate_captions(model, images, tokenizer, max_len, num_beams=4,
                      top_p=None, temperature=1.0):
    """Generate captions. If top_p is given, use nucleus sampling (overrides beam)."""
    if top_p is not None:
        generated_ids = model.generate(
            pixel_values=images,
            max_length=max_len,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        generated_ids = model.generate(
            pixel_values=images,
            max_length=max_len,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def generate_captions(model, images, tokenizer, max_len, num_beams=4,
                      top_p=None, temperature=1.0):
    return _generate_captions(model, images, tokenizer, max_len, num_beams,
                              top_p=top_p, temperature=temperature)


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


def build_test_dataset(data_dir, tokenizer, max_len):
    captions_file = os.path.join(data_dir, "captions.txt")
    return Flickr8kDataset(
        image_dir=os.path.join(data_dir, "Images"),
        captions_file=captions_file,
        tokenizer=tokenizer,
        max_length=max_len,
        split="test",
        transform=ViTTransform(),
    )


def build_test_loader(data_dir, tokenizer, max_len, batch_size=32, num_workers=2):
    dataset = build_test_dataset(data_dir, tokenizer, max_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def _decode_reference(token_ids, tokenizer, pad_token_id):
    ids = token_ids.tolist()
    while ids and ids[-1] == pad_token_id:
        ids.pop()
    return tokenizer.decode(ids, skip_special_tokens=True)


def save_predictions_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {path}")


def evaluate_test(model, test_loader, criterion, tokenizer, device, max_len, num_beams=4,
                  save_predictions_path=None, references_dict=None, top_p=None,
                  temperature=1.0):
    """Return loss, CLIPScore, and BLEU-1/2/3/4; optionally save predictions."""
    model.eval()
    total_loss = 0.0
    clip_metric = _CLIPScorer(device)
    pad_token_id = tokenizer.pad_token_id
    rows = [] if save_predictions_path else None
    image_captions = {}  # {image_name: generated_caption}, deduped by image

    with torch.no_grad():
        for images, token_ids, image_names in tqdm(test_loader, desc="Evaluating test set"):
            images    = images.to(device)
            token_ids = token_ids.to(device)

            dec_input = token_ids[:, :-1]
            dec_attn_mask = (dec_input != pad_token_id).long()
            labels = token_ids[:, 1:].clone()
            labels[labels == pad_token_id] = -100

            logits = model(
                pixel_values=images,
                decoder_input_ids=dec_input,
                decoder_attention_mask=dec_attn_mask,
            ).logits
            total_loss += criterion(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            ).item()

            captions = _generate_captions(model, images, tokenizer, max_len, num_beams,
                                          top_p=top_p, temperature=temperature)
            clip_metric.update(_unnormalize(images.cpu()), captions)

            if rows is not None:
                for i, name in enumerate(image_names):
                    rows.append({
                        "image": name,
                        "reference": _decode_reference(token_ids[i], tokenizer, pad_token_id),
                        "prediction": captions[i],
                    })

            if references_dict is not None:
                for name, cap in zip(image_names, captions):
                    image_captions.setdefault(name, cap)

    bleu_scores = (0.0, 0.0, 0.0, 0.0)
    if references_dict is not None and image_captions:
        names  = list(image_captions.keys())
        hyps   = [image_captions[n].split() for n in names]
        refs   = [[ref.split() for ref in references_dict.get(n, [])] for n in names]
        smooth = SmoothingFunction().method1
        bleu_scores = (
            corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),              smoothing_function=smooth),
            corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0),          smoothing_function=smooth),
            corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3, 0),        smoothing_function=smooth),
            corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25),  smoothing_function=smooth),
        )

    if save_predictions_path:
        save_predictions_csv(rows, save_predictions_path)

    return total_loss / len(test_loader), clip_metric.compute(), *bleu_scores


def _sample_indices(dataset_len, n, seed):
    n = min(n, dataset_len)
    rng = random.Random(seed)
    return rng.sample(range(dataset_len), n)


@torch.no_grad()
def predict_sample(model, dataset, idx, tokenizer, device, max_len, num_beams=4,
                   top_p=None, temperature=1.0):
    """Return (image_name, reference, prediction, PIL image) for one dataset index."""
    image_name, reference = dataset.data[idx]
    image_path = os.path.join(dataset.image_dir, image_name)
    pil_image = Image.open(image_path).convert("RGB")
    transform = ViTTransform()
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    prediction = generate_captions(
        model, image_tensor, tokenizer, max_len, num_beams,
        top_p=top_p, temperature=temperature,
    )[0]
    return image_name, reference, prediction, pil_image


def show_samples(model, dataset, tokenizer, device, n, max_len, num_beams=4,
                 seed=42, top_p=None, temperature=1.0):
    """Print random test examples to the terminal."""
    model.eval()
    indices = _sample_indices(len(dataset), n, seed)

    print("\n" + "=" * 72)
    print(f"{'SAMPLE PREDICTIONS':^72}")
    print("=" * 72)
    for idx in indices:
        image_name, reference, prediction, _ = predict_sample(
            model, dataset, idx, tokenizer, device, max_len, num_beams,
            top_p=top_p, temperature=temperature,
        )
        print(f"\n  Image    : {image_name}")
        print(f"  Predicted: {prediction}")
        print(f"  Reference: {reference}")
    print("\n" + "=" * 72)


def plot_samples(
    model,
    dataset,
    tokenizer,
    device,
    n=4,
    max_len=30,
    num_beams=4,
    seed=42,
    cols=None,
    save_path=None,
    top_p=None,
    temperature=1.0,
):
    """Show a grid of images with predicted and reference captions."""
    import matplotlib.pyplot as plt

    model.eval()
    indices = _sample_indices(len(dataset), n, seed)
    n = len(indices)
    cols = cols or min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    axes_flat = [axes] if n == 1 else (axes.flatten() if hasattr(axes, "flatten") else [axes])

    for ax, idx in zip(axes_flat, indices):
        image_name, reference, prediction, pil_image = predict_sample(
            model, dataset, idx, tokenizer, device, max_len, num_beams,
            top_p=top_p, temperature=temperature,
        )
        ax.imshow(pil_image)
        ax.set_title(prediction, fontsize=9)
        ax.text(
            0.5, -0.12, reference, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, wrap=True,
        )
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Predicted (title) · Reference (below)", fontsize=11, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Evaluate or demo the ViT-GPT2 image captioning model.")
    parser.add_argument("--checkpoint", default=DEFAULT_CONFIG["checkpoint"])
    parser.add_argument("--data_dir", default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--image", default=None, help="Single image path (skips test-set evaluation)")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_CONFIG["n_samples"])
    parser.add_argument("--max_len", type=int, default=DEFAULT_CONFIG["max_len"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num_beams", type=int, default=DEFAULT_CONFIG["num_beams"],
                        help="Beam size (default: 4; use 1 for greedy)")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                        help="RNG seed for sample selection (default: 42)")
    parser.add_argument("--no_samples", action="store_true", help="Skip sample predictions at the end")
    parser.add_argument("--save_predictions", default=None, metavar="PATH",
                        help="CSV path for all test predictions (image, reference, prediction)")
    parser.add_argument("--output_dir", default=None,
                        help="If set, save sample grid PNG to this directory")
    parser.add_argument("--top_p", type=float, default=None,
                        help="If set, use nucleus sampling with this top_p (e.g. 0.9) instead of beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (only used with --top_p)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Zero-shot: evaluate the pretrained model without any fine-tuning checkpoint")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = build_tokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")

    try:
        model = load_model(args.checkpoint, tokenizer, device, zero_shot=args.no_checkpoint)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        sys.exit(1)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            sys.exit(1)
        transform = ViTTransform()
        image_tensor = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
        caption = generate_captions(
            model, image_tensor, tokenizer, args.max_len, args.num_beams,
            top_p=args.top_p, temperature=args.temperature,
        )[0]
        print(f"\nImage  : {args.image}")
        print(f"Caption: {caption}")
        return

    captions_file = os.path.join(args.data_dir, "captions.txt")
    references_dict = build_references_dict(captions_file)
    test_dataset, test_loader = build_test_loader(
        args.data_dir, tokenizer, args.max_len, args.batch_size
    )
    print(f"Test set: {len(test_dataset)} samples")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    test_loss, test_cs, b1, b2, b3, b4 = evaluate_test(
        model,
        test_loader,
        criterion,
        tokenizer,
        device,
        args.max_len,
        args.num_beams,
        save_predictions_path=args.save_predictions,
        references_dict=references_dict,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    print(f"\n{'Test loss':>15}: {test_loss:.4f}")
    print(f"{'Test CLIPScore':>15}: {test_cs:.2f}")
    print(f"{'Test BLEU-1':>15}: {b1:.4f}")
    print(f"{'Test BLEU-2':>15}: {b2:.4f}")
    print(f"{'Test BLEU-3':>15}: {b3:.4f}")
    print(f"{'Test BLEU-4':>15}: {b4:.4f}")

    if not args.no_samples:
        show_samples(
            model, test_dataset, tokenizer, device,
            args.n_samples, args.max_len, args.num_beams, seed=args.seed,
            top_p=args.top_p, temperature=args.temperature,
        )
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, "test_samples.png")
            plot_samples(
                model, test_dataset, tokenizer, device,
                n=args.n_samples, max_len=args.max_len, num_beams=args.num_beams,
                seed=args.seed, save_path=out_path,
                top_p=args.top_p, temperature=args.temperature,
            )


if __name__ == "__main__":
    main()
