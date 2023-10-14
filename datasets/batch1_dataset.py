import random, math
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from lib.preprocessing import image_folder_to_list
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.vae import DiagonalGaussianDistribution


class Batch1Dataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        # self.shuffle_tags = config["shuffle_tags"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        item = {}
        if "latent_values" in entry:
            item["latent_values"] = entry["latent_values"]
        else:
            tfs = v2.Compose(
                [
                    v2.RandomCrop(
                        math.floor(entry["width"] / 64) * 64,
                        math.floor(entry["height"] / 64) * 64,
                    )
                    if self.config.random_crop
                    else v2.CenterCrop(
                        math.floor(entry["width"] / 64) * 64,
                        math.floor(entry["height"] / 64) * 64,
                    ),
                    v2.RandomHorizontalFlip()
                    if self.config.random_flip
                    else v2.Lambda(lambda x: x),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.5], [0.5]),
                ]
            )
            item["pixel_values"] = tfs(entry["image"])
        if len(entry["input_ids"].shape) == 1:
            item["input_ids"] = entry["input_ids"]
        elif len(entry["input_ids"].shape) > 1:
            item["input_ids"] = random.choice(entry["input_ids"])
        else:
            pass
        return item


def get_batch1_dataset(data, config=None, tokenizer=None, vae=None, text_encoder=None):
    if vae:
        simple_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.get_default_dtype(), scale=True),
                v2.Normalize([0.5], [0.5]),
            ]
        )
        for entry in data:
            pixel_values = simple_transforms(entry["image"]).unsqueeze(dim=0)
            latent_dist = vae.encode(pixel_values).latent_dist
            entry["latent_values"] = latent_dist.parameters

    if not tokenizer:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    input_ids = tokenizer(
        [entry["caption"] for entry in data],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids

    for i, entry in enumerate(data):
        entry["input_ids"] = input_ids[i]

    return Batch1Dataset(data, config)
