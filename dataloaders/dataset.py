import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import PreTrainedTokenizerBase


class Flickr8kDataset(Dataset):

    _SPLITS = {"train": (0.0, 0.75), "val": (0.75, 0.875), "test": (0.875, 1.0)}

    def __init__(self, image_dir: str, captions_file: str, tokenizer: PreTrainedTokenizerBase,
                 max_length: int, split: str = "train", transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
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

        tokens = self.tokenizer.encode(caption.lower(), add_special_tokens=False)
        tokens = tokens[: self.max_length - 2]
        ids = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        ids += [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        return image, torch.tensor(ids[: self.max_length], dtype=torch.long), image_name
