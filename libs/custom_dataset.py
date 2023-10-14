import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        random_crop: bool = False,
        random_flip: bool = False,
        shuffle_tags: bool = False,
    ):
        self.data = data
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.shuffle_tags = shuffle_tags

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
                    v2.RandomCrop(entry["grid_width"], entry["grid_height"])
                    if self.random_crop
                    else v2.CenterCrop(entry["grid_width"], entry["grid_height"]),
                    v2.RandomHorizontalFlip()
                    if self.random_flip
                    else v2.Lambda(lambda x: x),
                    v2.ToImage(),
                    v2.ToDtype(torch.get_default_dtype(), scale=True),
                    v2.Normalize([0.5], [0.5]),
                ]
            )
            item["pixel_values"] = tfs(entry["image"])
            item["pixel_values"] = item["pixel_values"].to(
                memory_format=torch.contiguous_format
            )
        if len(entry["input_ids"].shape) == 1:
            item["input_ids"] = entry["input_ids"]
        elif len(entry["input_ids"].shape) > 1:
            item["input_ids"] = random.choice(entry["input_ids"])
        else:
            pass
        return {
            key: item[key]
            for key in ["pixel_values", "input_ids", "latent_values"]
            if key in item
        }
