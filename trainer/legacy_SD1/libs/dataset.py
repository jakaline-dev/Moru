import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import CLIPTokenizer


class Gridify(torch.nn.Module):
    def forward(
        self, img, bboxes, label
    ):  # we assume inputs are always structured like this
        print(
            f"I'm transforming an image of shape {img.shape} "
            f"with bboxes = {bboxes}\n{label = }"
        )

        # Do some transformations. Here, we're just passing though the input
        return img, bboxes, label


class MoruDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer=None,
        random_crop: bool = False,
        random_flip: bool = False,
        shuffle_tags: bool = False,
        caption_dropout: float = 0.0,
    ):
        if not tokenizer:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.data = data
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.shuffle_tags = shuffle_tags
        self.caption_dropout = caption_dropout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        available_keys = []
        if "latent_values" not in entry:
            # print(entry["image"].size, entry["grid_height"], entry["grid_width"])
            tfs = v2.Compose(
                [
                    v2.RandomCrop([entry["grid_height"], entry["grid_width"]])
                    if self.random_crop
                    else v2.CenterCrop([entry["grid_height"], entry["grid_width"]]),
                    v2.RandomHorizontalFlip()
                    if self.random_flip
                    else v2.Lambda(lambda x: x),
                    v2.ToImage(),
                    v2.ToDtype(torch.get_default_dtype(), scale=True),
                    v2.Normalize([0.5], [0.5]),
                ]
            )
            entry["pixel_values"] = tfs(entry["image"])
            entry["pixel_values"] = entry["pixel_values"].to(
                memory_format=torch.contiguous_format
            )
            available_keys.append("pixel_values")
        else:
            available_keys.append("latent_values")
        if "text_embeddings" in entry:
            available_keys.append("text_embeddings")
        elif "input_ids" in entry:
            if len(entry["input_ids"].shape) > 1:
                entry["input_ids"] = random.choice(entry["input_ids"])
            available_keys.append("input_ids")
        else:
            if len(entry["input_ids"].shape) > 1:
                entry["caption"] == random.choice(entry["caption"])
            # , == 267
            if self.shuffle_tags:
                pass
            entry["input_ids"] = self.tokenizer(
                entry["caption"],
                padding=True,
                return_tensors="pt",
                pad_to_multiple_of=self.tokenizer.model_max_length,
            ).input_ids
            available_keys.append("input_ids")
        return {key: entry[key] for key in available_keys if key in entry}
