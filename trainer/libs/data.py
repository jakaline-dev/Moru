import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from .preprocessing import get_target_size


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


class MoruDataset(Dataset):
    def __init__(
        self,
        random_flip: bool = False,
        random_crop: bool = False,
        shuffle_tags: bool = False,
        tokenizer: CLIPTokenizer = None,
        tokenizer_2: CLIPTokenizer = None,
    ):
        self.data = None
        self.buckets = None
        self.downcast_original_sizes = True
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.shuffle_tags = shuffle_tags
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

    def load_local_folder(self, folder_path: str):
        self.data = []
        print("Load local folder")
        for filename in tqdm(os.listdir(folder_path)):
            try:
                image_path = os.path.join(folder_path, filename)
                if (
                    not os.path.isfile(image_path)
                    or os.path.splitext(filename)[1] == ".txt"
                ):
                    continue
                image = Image.open(image_path)
                # pre-resize
                target_sizes, resize_res = get_target_size(image)
                if not (image.height == resize_res[0] and image.width == resize_res[1]):
                    image = image.resize(resize_res, Image.Resampling.BICUBIC)
                # if image.mode == "RGBA":
                #     # No transparency allowed in PNGs - change alpha to white
                #     white_background = Image.new("RGB", image.size, (255, 255, 255))
                #     white_background.paste(image, (0, 0), mask=image.split()[3])
                #     image = white_background
                image = image.convert("RGB")

                base_name = os.path.splitext(filename)[0]
                text_path = os.path.join(folder_path, base_name + ".txt")
                if not os.path.isfile(text_path):
                    continue
                text = open(text_path, "r").read().strip().splitlines()
                if len(text) == 1:
                    text = text[0]

                if (
                    image.height > resize_res[0]
                    and image.width > resize_res[1]
                    and self.downcast_original_sizes
                ):
                    original_sizes = resize_res
                else:
                    original_sizes = tuple([image.height, image.width])

                self.data.append(
                    {
                        "image": image,
                        "text": text,
                        "target_sizes": target_sizes,
                        "original_sizes": original_sizes,
                    }
                )

            except Exception as e:
                print(e)
        # buckets
        buckets = {}
        for idx, item in enumerate(self.data):
            if item["target_sizes"] not in buckets:
                buckets[item["target_sizes"]] = []
            buckets[item["target_sizes"]].append(idx)
        print("---Buckets---")
        print({key: len(value) for key, value in buckets.items()})
        self.buckets = list(buckets.values())

    @torch.inference_mode()
    def cache_vae(self, vae, device):
        print("Cache VAE")
        vae.to(device)
        for idx, entry in enumerate(tqdm(self.data)):
            tfs = transforms.Compose(
                [
                    transforms.CenterCrop(entry["target_sizes"]),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            crop_top = max(
                0, int(round((entry["image"].height - entry["target_sizes"][0]) / 2.0))
            )
            crop_left = max(
                0, int(round((entry["image"].width - entry["target_sizes"][1]) / 2.0))
            )
            entry["crop_top_lefts"] = tuple([crop_top, crop_left])
            latent_dist = vae.encode(
                tfs(entry["image"]).unsqueeze(dim=0).to(device)
            ).latent_dist
            entry["latent_values"] = latent_dist.parameters
            entry["latent_values"] = entry["latent_values"].squeeze(dim=0).float().cpu()
            self.data[idx] = entry
            del entry["image"]
        vae.to("cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        if "latent_values" not in entry:
            tfs = [
                transforms.CenterCrop(entry["target_sizes"])
                if not self.random_crop
                else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip()
                if self.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
            if self.random_crop:
                transforms_random_crop = transforms.RandomCrop(entry["target_sizes"])
                crop_dict = transforms_random_crop.get_params(
                    entry["image"], entry["target_sizes"]
                )
                entry["crop_top_lefts"] = tuple([crop_dict[0], crop_dict[1]])
                tfs = [transforms_random_crop] + tfs
            else:
                crop_top = max(
                    0,
                    int(
                        round((entry["image"].height - entry["target_sizes"][0]) / 2.0)
                    ),
                )
                crop_left = max(
                    0,
                    int(round((entry["image"].width - entry["target_sizes"][1]) / 2.0)),
                )
                entry["crop_top_lefts"] = tuple([crop_top, crop_left])
            entry["pixel_values"] = transforms.Compose(tfs)(entry["image"])

        if "text_embeddings" in entry:
            pass
        elif "input_ids" in entry:
            pass
            # if len(entry["input_ids"].shape) > 1:
            #     entry["input_ids"] = random.choice(entry["input_ids"])
        else:
            if isinstance(entry["text"], str):
                caption = entry["text"]
            elif isinstance(entry["text"], (list, np.ndarray)):
                # take a random caption if there are multiple
                caption.append(random.choice(entry["text"]))
            else:
                raise ValueError(
                    "Caption should contain either strings or lists of strings."
                )
            if self.shuffle_tags:
                pass
            entry["input_ids"] = tokenize_prompt(self.tokenizer, caption)
            entry["input_ids_2"] = tokenize_prompt(self.tokenizer_2, caption)
        return entry


class MoruBatchSampler(Sampler[list[int]]):
    def __init__(
        self, buckets, batch_size: int = 1, drop_last: bool = False, generator=None
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buckets = buckets
        self.generator = generator
        self.has_subfolder = False

    def __iter__(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        for k in torch.randperm(len(self.buckets), generator=generator).tolist():
            sampler = SubsetRandomSampler(self.buckets[k], generator=generator)
            if self.drop_last:
                sampler_iter = iter(sampler)
                while True:
                    try:
                        batch = [next(sampler_iter) for _ in range(self.batch_size)]
                        yield batch
                    except StopIteration:
                        break
            else:
                batch = [0] * self.batch_size
                idx_in_batch = 0
                for idx in sampler:
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        yield batch
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                if idx_in_batch > 0:
                    yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return sum([len(x) // self.batch_size for x in self.buckets])
        else:
            return sum([np.ceil(len(x) / self.batch_size) for x in self.buckets])


def MoruCollateFn(examples):
    batch = dict()
    if "pixel_values" in examples[0]:
        batch["pixel_values"] = torch.stack(
            [example["pixel_values"] for example in examples]
        ).to(memory_format=torch.contiguous_format)  # .float()
    if "latent_values" in examples[0]:
        batch["latent_values"] = torch.stack(
            [example["latent_values"] for example in examples]
        )
    batch["original_sizes"] = [example["original_sizes"] for example in examples]
    batch["crop_top_lefts"] = [example["crop_top_lefts"] for example in examples]
    batch["target_sizes"] = [example["target_sizes"] for example in examples]
    batch["input_ids"] = torch.stack([example["input_ids"] for example in examples])
    batch["input_ids_2"] = torch.stack([example["input_ids_2"] for example in examples])
    return batch


def MoruDataLoader(
    ds: MoruDataset,
    seed=None,
    batch_size: int = 1,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
):
    if seed:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None
    return DataLoader(
        ds,
        sampler=None,
        batch_sampler=MoruBatchSampler(
            ds.buckets, batch_size=batch_size, drop_last=drop_last, generator=generator
        ),
        collate_fn=MoruCollateFn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
