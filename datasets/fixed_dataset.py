import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from lib.preprocessing import image_folder_to_list, crop_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.vae import DiagonalGaussianDistribution


class FixedDataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.shuffle_tags = config["shuffle_tags"]
        self.transforms = transforms.Compose(
            [
                transforms.RandomCrop(config["crop_width"], config["crop_height"])
                if config["random_crop"]
                else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip()
                if config["random_flip"]
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        item = {}
        if "latent_values" in entry:
            item["latent_values"] = entry["latent_values"]
        else:
            item["pixel_values"] = self.transforms(entry["image"])

        if len(entry["input_ids"].shape) == 1:
            item["input_ids"] = entry["input_ids"]
        elif len(entry["input_ids"].shape) > 1:
            item["input_ids"] = random.choice(entry["input_ids"])
        else:
            pass
        return item


def get_fixed_dataset(data, config=None, tokenizer=None, vae=None, text_encoder=None):
    data = crop_dataset(
        data,
        crop_width=config["crop_width"],
        crop_height=config["crop_height"],
    )
    if vae:
        simple_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
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

    return FixedDataset(data, config)
