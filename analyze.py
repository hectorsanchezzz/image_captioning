"""Per-sample analysis of the ViT-GPT2 captioning model on the Flickr8k test set.

Produces outputs/predictions.csv with one row per test image, including
predicted caption, references, per-sample BLEU-1/2/3/4, CLIPScore,
caption lengths, and simple gender-mention flags. The notebook
notebooks/bias_analysis.ipynb consumes this CSV.
"""
import argparse
import collections
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from dataloaders.dataset import Flickr8kDataset
from test import (
    build_references_dict,
    build_tokenizer,
    load_model,
    _ViTTransform,
    _generate_captions,
    _unnormalize,
)


_MALE_WORDS = {"man", "men", "boy", "boys", "guy", "guys", "male", "father", "dad", "gentleman"}
_FEMALE_WORDS = {"woman", "women", "girl", "girls", "lady", "ladies", "female", "mother", "mom"}
_CHILD_WORDS = {"child", "children", "kid", "kids", "baby", "babies", "toddler", "infant"}


def _tokenize_simple(text: str):
    return re.findall(r"[a-z']+", text.lower())


def _has_any(tokens, vocab):
    return any(t in vocab for t in tokens)


@torch.no_grad()
def _clip_scores_batch(clip_model, clip_processor, images_uint8, captions, device):
    imgs = [img.permute(1, 2, 0).numpy() for img in images_uint8.cpu()]
    inputs = clip_processor(
        text=captions, images=imgs,
        return_tensors="pt", padding=True, truncation=True, max_length=77,
    ).to(device)
    out = clip_model(**inputs)
    img_f = F.normalize(out.image_embeds, dim=-1)
    txt_f = F.normalize(out.text_embeds, dim=-1)
    return ((img_f * txt_f).sum(dim=-1) * 100.0).cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Per-sample bias/weakness analysis on the test set.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--data_dir", default="data/flickr8k")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=None,
                        help="If set, use nucleus sampling with this top_p instead of beam.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Zero-shot: analyse the pretrained model without fine-tuning.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = build_tokenizer()
    model = load_model(args.checkpoint, tokenizer, device, zero_shot=args.no_checkpoint)

    captions_file = os.path.join(args.data_dir, "captions.txt")
    references_dict = build_references_dict(captions_file)

    test_dataset = Flickr8kDataset(
        image_dir=os.path.join(args.data_dir, "Images"),
        captions_file=captions_file,
        tokenizer=tokenizer,
        max_length=args.max_len,
        split="test",
        transform=_ViTTransform(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"Test set: {len(test_dataset)} samples (rows include duplicates per caption).")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    smooth = SmoothingFunction().method1
    seen = set()
    rows = []

    for images, _token_ids, image_names in tqdm(test_loader, desc="Analyzing"):
        images = images.to(device)
        captions = _generate_captions(model, images, tokenizer, args.max_len, args.num_beams,
                                      top_p=args.top_p, temperature=args.temperature)

        # Dedupe by image to avoid 5x scoring of the same prediction.
        # Must also dedupe within the batch itself, since consecutive rows
        # of the same image fall into the same batch.
        keep_idx = []
        for i, n in enumerate(image_names):
            if n not in seen:
                seen.add(n)
                keep_idx.append(i)
        if not keep_idx:
            continue

        kept_images_u8 = _unnormalize(images.cpu())[keep_idx]
        kept_captions = [captions[i] for i in keep_idx]
        kept_names = [image_names[i] for i in keep_idx]
        clip_scores = _clip_scores_batch(clip_model, clip_processor, kept_images_u8, kept_captions, device)

        for name, pred, cs in zip(kept_names, kept_captions, clip_scores):
            refs = references_dict.get(name, [])
            pred_tokens = _tokenize_simple(pred)
            ref_token_lists = [_tokenize_simple(r) for r in refs]
            avg_ref_len = sum(len(t) for t in ref_token_lists) / max(len(ref_token_lists), 1)

            if pred_tokens and ref_token_lists:
                b1 = sentence_bleu(ref_token_lists, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
                b2 = sentence_bleu(ref_token_lists, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
                b3 = sentence_bleu(ref_token_lists, pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
                b4 = sentence_bleu(ref_token_lists, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            else:
                b1 = b2 = b3 = b4 = 0.0

            pred_set = set(pred_tokens)
            refs_tokens_flat = {t for toks in ref_token_lists for t in toks}

            rows.append({
                "image_name": name,
                "predicted_caption": pred,
                "references": " ||| ".join(refs),
                "bleu_1": b1, "bleu_2": b2, "bleu_3": b3, "bleu_4": b4,
                "clip_score": cs,
                "pred_length": len(pred_tokens),
                "ref_avg_length": avg_ref_len,
                "pred_mentions_man": _has_any(pred_set, _MALE_WORDS),
                "pred_mentions_woman": _has_any(pred_set, _FEMALE_WORDS),
                "pred_mentions_child": _has_any(pred_set, _CHILD_WORDS),
                "refs_mention_man": _has_any(refs_tokens_flat, _MALE_WORDS),
                "refs_mention_woman": _has_any(refs_tokens_flat, _FEMALE_WORDS),
                "refs_mention_child": _has_any(refs_tokens_flat, _CHILD_WORDS),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print(f"Mean BLEU-1 (sentence): {df['bleu_1'].mean():.4f}")
    print(f"Mean BLEU-4 (sentence): {df['bleu_4'].mean():.4f}")
    print(f"Mean CLIPScore:         {df['clip_score'].mean():.2f}")
    print(f"Unique predictions:     {df['predicted_caption'].nunique()} / {len(df)}")

    # Corpus BLEU — alineat amb la mètrica de validació a train.py
    from nltk.translate.bleu_score import corpus_bleu
    hyps = [_tokenize_simple(p) for p in df['predicted_caption']]
    refs = [[_tokenize_simple(r) for r in str(refs_str).split(' ||| ')]
            for refs_str in df['references']]
    cb1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),                 smoothing_function=smooth)
    cb2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0),             smoothing_function=smooth)
    cb3 = corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3, 0),           smoothing_function=smooth)
    cb4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25),     smoothing_function=smooth)
    print(f"\nCorpus BLEU-1: {cb1:.4f}  (aligned with train.py evaluate)")
    print(f"Corpus BLEU-2: {cb2:.4f}")
    print(f"Corpus BLEU-3: {cb3:.4f}")
    print(f"Corpus BLEU-4: {cb4:.4f}")


if __name__ == "__main__":
    main()
