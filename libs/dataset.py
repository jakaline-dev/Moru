import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class MoruDataset(Dataset):
    def __init__(
        self,
        data,
        random_crop: bool = False,
        random_flip: bool = False,
        shuffle_tags: bool = False,
        caption_dropout: float = 0.0,
    ):
        self.data = data
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.shuffle_tags = shuffle_tags
        self.caption_dropout = caption_dropout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        available_keys = ["input_ids"]
        if "latent_values" not in entry:
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
            entry["pixel_values"] = tfs(entry["image"])
            entry["pixel_values"] = entry["pixel_values"].to(
                memory_format=torch.contiguous_format
            )
            available_keys.append("pixel_values")
        else:
            available_keys.append("latent_values")
        if len(entry["input_ids"].shape) > 1:
            entry["input_ids"] = random.choice(entry["input_ids"])

        return {key: entry[key] for key in available_keys if key in entry}
