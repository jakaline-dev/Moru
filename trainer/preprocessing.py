import torch
import numpy as np
import random
from datasets import Dataset
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from PIL import Image
from PIL.Image import Resampling
import os

SDXL_GRID = [
    # (512, 2048), # 4
    # (512, 1984),
    # (512, 1920),
    # (512, 1856),
    # (576, 1792),
    # (576, 1728), # 3
    # (576, 1664),
    # (640, 1600),
    # (640, 1536),
    # (704, 1472),
    (704, 1408),  # 2
    (704, 1344),
    (768, 1344),
    (768, 1280),
    (832, 1216),
    (832, 1152),
    (896, 1152),
    (896, 1088),
    (960, 1088),
    (960, 1024),
    (1024, 1024),  # 1
    (1024, 960),
    (1088, 960),
    (1088, 896),
    (1152, 896),
    (1152, 832),
    (1216, 832),
    (1280, 768),
    (1344, 768),
    (1344, 704),
    (1408, 704),  # 2
    (1472, 704),
    (1536, 640),
    (1600, 640),
    (1664, 576),
    (1728, 576),  # 3
    # (1792, 576),
    # (1856, 512),
    # (1920, 512),
    # (1984, 512),
    # (2048, 512) # 4
]
SDXL_GRID_RATIO = [x / y for (x, y) in SDXL_GRID]


def find_closest_index(sorted_list, x):
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))


def find_closest(sorted_list, x):
    return min(sorted_list, key=lambda item: abs(item - x))


def load_dataset_local_gen(folder_path: str, image_column: str, text_column: str):
    for filename in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, filename)
            if (
                not os.path.isfile(image_path)
                or os.path.splitext(filename)[1] == ".txt"
            ):
                continue
            img = Image.open(image_path)
            # pre-resize
            # _, resize_resolution = get_target_size(img)
            # img = img.resize(resize_resolution, Resampling.BICUBIC)

            base_name = os.path.splitext(filename)[0]
            text_path = os.path.join(folder_path, base_name + ".txt")
            if not os.path.isfile(text_path):
                continue
            text = open(text_path, "r").read().strip().splitlines()
            if len(text) == 1:
                text = text[0]
            item = {image_column: img, text_column: text}
            yield item
        except Exception as e:
            print(e)


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


# Do not run this in distributed systems - Make a predefined dataset beforehand
def load_dataset_local(
    folder_path, image_column: str = "image", text_column: str = "text"
):
    if not os.path.exists(folder_path):
        raise Exception("Path does not exist")
    return Dataset.from_generator(
        load_dataset_local_gen,
        gen_kwargs={
            "folder_path": folder_path,
            "image_column": image_column,
            "text_column": text_column,
        },
    )


def get_target_size(image: Image):
    image_ratio = image.width / image.height
    target_size_idx = find_closest_index(SDXL_GRID_RATIO, image_ratio)
    target_size = SDXL_GRID[target_size_idx]
    target_ratio = SDXL_GRID_RATIO[target_size_idx]
    if target_ratio <= image_ratio:
        k = target_size[1] / image.height
        resize_resolution = (int(k * image.width), target_size[1])
    else:
        k = target_size[0] / image.width
        resize_resolution = (target_size[0], int(k * image.height))
    return target_size, resize_resolution
    # w1/h1 < x/y < w2/h2
    # r1: ky=h1
    # r2: kx=w2


def fit_grid(example, image_column: str):
    image = example[image_column]
    target_size, resize_res = get_target_size(image)
    example["original_sizes"] = tuple([image.height, image.width])
    example["target_sizes"] = tuple([target_size[1], target_size[0]])
    example[image_column] = image.resize(resize_res, Image.Resampling.BICUBIC)
    return example


def collate_fn(examples):
    batch = dict()
    batch["pixel_values"] = torch.stack(
        [example["pixel_values"] for example in examples]
    ).to(memory_format=torch.contiguous_format)  # .float()
    batch["original_sizes"] = [example["original_sizes"] for example in examples]
    batch["crop_top_lefts"] = [example["crop_top_lefts"] for example in examples]
    batch["target_sizes"] = [example["target_sizes"] for example in examples]
    batch["input_ids"] = torch.stack([example["input_ids"] for example in examples])
    batch["input_ids_2"] = torch.stack([example["input_ids_2"] for example in examples])
    return batch


def setup_dataset(
    ds: Dataset,
    image_column: str = "image",
    text_column: str = "text",
    random_flip: bool = False,
    random_crop: bool = False,
    tokenizer=None,
):
    # I1. Resize (Prepare)
    # I2. Flip (None / Transform)
    # I3. Dynamic Crop (None / Transform)
    # I4. Encode (Prepare / Model)
    # T1. Tokenize (Prepare / Transform)
    # T2. Encode (Prepare / Model)

    def _transform_dataset(examples):
        default_composer = [
            # transforms.ToTensor(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5], [0.5]),
        ]
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        pixel_values = []
        original_sizes = []
        crop_top_lefts = []
        target_sizes = []
        for image in images:
            composer = default_composer
            target_size, resize_resolution = get_target_size(image)
            original_sizes.append((image.height, image.width))
            target_sizes.append((target_size[1], target_size[0]))
            # composer = [
            #     transforms.Resize(
            #         resize_resolution,
            #         interpolation=transforms.InterpolationMode.BICUBIC,
            #     )
            # ] + composer
            image = image.resize((resize_resolution), Image.BICUBIC)
            if random_flip:
                # flip
                composer = [transforms.RandomHorizontalFlip()] + composer
            if not random_crop:
                composer = [transforms.CenterCrop(target_size[::-1])] + composer
                crop_top = max(
                    0, int(round((resize_resolution[1] - target_size[1]) / 2.0))
                )
                crop_left = max(
                    0, int(round((resize_resolution[0] - target_size[0]) / 2.0))
                )
                crop_top_left = tuple([crop_top, crop_left])
            else:
                transforms_random_crop = transforms.RandomCrop(target_size[::-1])
                crop_dict = transforms_random_crop.get_params(image, target_size[::-1])
                # print(image.width, image.height, target_size)
                # print(crop_dict)
                crop_top_left = tuple([crop_dict[0], crop_dict[1]])
                composer = [transforms_random_crop] + composer
            crop_top_lefts.append(crop_top_left)
            image = transforms.Compose(composer)(image)
            pixel_values.append(image)
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["target_sizes"] = target_sizes
        examples["pixel_values"] = pixel_values
        # text
        captions = []
        for caption in examples[text_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column `{text_column}` should contain either strings or lists of strings."
                )
        examples["input_ids"] = tokenize_prompt(tokenizer, captions)
        examples["input_ids_2"] = tokenize_prompt(tokenizer, captions)
        return examples

    # ds = ds.filter(
    #     lambda item: SDXL_GRID_RATIO[0]
    #     <= item[image_column].size[0] / item[image_column].size[1]
    #     <= SDXL_GRID_RATIO[-1]
    # )
    # ds = ds.map(fit_grid, fn_kwargs={"image_column": image_column})
    ds = ds.with_transform(_transform_dataset)
    return ds


class MyBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int = 1, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self._groups()
        self.has_subfolder = False

    def _groups(self):
        groups = {}
        for idx in range(len(self.dataset)):
            target_size = self.dataset[idx]["target_sizes"]
            if target_size not in groups:
                groups[target_size] = []
            groups[target_size].append(idx)
        return list(groups.values())

    def __iter__(self):
        for k in torch.randperm(len(self.groups)).tolist():
            sampler = SubsetRandomSampler(self.groups[k])
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
            return sum([len(x) // self.batch_size for x in self.groups])  # type: ignore[arg-type]
        else:
            return sum([np.ceil(len(x) / self.batch_size) for x in self.groups])  # type: ignore[arg-type]


def setup_dataloader(
    ds,
    batch_size: int = 1,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
):
    batch_sampler = MyBatchSampler(ds, batch_size=batch_size, drop_last=drop_last)
    return DataLoader(
        ds,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


if __name__ == "__main__":
    print("Load dataset")
    ds = load_dataset_local()
    print("Setup dataset")
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )

    pipe = download_from_original_stable_diffusion_ckpt(
        # path,
        from_safetensors=True,
        scheduler_type="ddpm",
        local_files_only=True,
    )
    tokenizer = pipe.tokenizer

    ds = setup_dataset(ds, tokenizer=tokenizer)
    print("Setup dataloader")
    dl = setup_dataloader(ds, batch_size=4)
    for item in dl:
        print(item)
