import torch
import numpy as np
import random
from datasets import Dataset
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from PIL import Image
from PIL.Image import Resampling
import os

# Height, Width
SDXL_GRID = [
    # (512, 2048), # 4
    # (512, 1984),
    # (512, 1920),
    # (512, 1856),
    # (576, 1792),
    (576, 1728), # 3
    (576, 1664),
    (640, 1600),
    (640, 1536),
    (704, 1472),
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
SDXL_GRID_RATIO = [h / w for (h, w) in SDXL_GRID]


def find_closest_index(sorted_list, x):
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))


def find_closest(sorted_list, x):
    return min(sorted_list, key=lambda item: abs(item - x))


def load_dataset_local_gen(folder_path: str):
    for filename in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, filename)
            if (
                not os.path.isfile(image_path)
                or os.path.splitext(filename)[1] == ".txt"
            ):
                continue
            img = Image.open(image_path)
            img = img.convert("RGB")
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
            item = {"image": img, "text": text}
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
    folder_path
):
    if not os.path.exists(folder_path):
        raise Exception("Path does not exist")
    return Dataset.from_generator(
        load_dataset_local_gen,
        cache_dir=None,
        gen_kwargs={
            "folder_path": folder_path
        },
    )


def get_target_size(image: Image):
    image_ratio = image.height / image.width
    target_size_idx = find_closest_index(SDXL_GRID_RATIO, image_ratio)
    target_size = SDXL_GRID[target_size_idx]
    target_ratio = SDXL_GRID_RATIO[target_size_idx]
    if target_ratio <= image_ratio:
        k = target_size[1] / image.width
        resize_resolution = (int(k * image.height), target_size[1])
    else:
        k = target_size[0] / image.height
        resize_resolution = (target_size[0], int(k * image.width))
    return target_size, resize_resolution

def fit_grid(example, downcast_original_sizes=True):
    image = example['image']
    target_size, resize_res = get_target_size(image)
    example["original_sizes"] = tuple([image.height, image.width])
    if image.height > resize_res[0] and image.width > resize_res[1] and downcast_original_sizes:
        example["original_sizes"] = resize_res
    example["target_sizes"] = tuple(target_size)
    example['image'] = image.convert("RGB").resize((resize_res[1], resize_res[0]), Image.Resampling.BICUBIC)
    return example

def cache_vae_fn(example, vae, device):
    tfs = transforms.Compose(
        [
            transforms.CenterCrop(example["target_sizes"]),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    crop_top = max(
        0, int(round((example['image'].height - example["target_sizes"][0]) / 2.0))
    )
    crop_left = max(
        0, int(round((example['image'].width - example["target_sizes"][1]) / 2.0))
    )
    example["crop_top_lefts"] = [crop_top, crop_left]
    pixel_values = tfs(example["image"].convert("RGB")).unsqueeze(dim=0)
    latent_dist = vae.encode(pixel_values.to(device)).latent_dist
    example["latent_values"] = latent_dist.parameters
    example["latent_values"] = example["latent_values"].squeeze(dim=0).float().cpu()
    del example["image"]
    del pixel_values
    return example

def collate_fn(examples):
    batch = dict()
    if "pixel_values" in examples[0]:
        batch["pixel_values"] = torch.stack(
            [example["pixel_values"] for example in examples]
        ).to(memory_format=torch.contiguous_format)  # .float()
    if "latent_values" in examples[0]:
        #print(examples[0]['latent_values'])
        #print(type(examples[0]['latent_values']))
        batch["latent_values"] = torch.stack([example["latent_values"] for example in examples])
    batch["original_sizes"] = [example["original_sizes"] for example in examples]
    batch["crop_top_lefts"] = [example["crop_top_lefts"] for example in examples]
    batch["target_sizes"] = [example["target_sizes"] for example in examples]
    batch["input_ids"] = torch.stack([example["input_ids"] for example in examples])
    batch["input_ids_2"] = torch.stack([example["input_ids_2"] for example in examples])
    return batch

def setup_dataset(ds: Dataset):
    ds = ds.filter(
        lambda item: SDXL_GRID_RATIO[0]
        <= item["image"].height / item["image"].width
        <= SDXL_GRID_RATIO[-1]
    )
    ds = ds.map(fit_grid, keep_in_memory=True)
    return ds

def get_buckets(ds: Dataset):
    buckets = {}
    for idx, item in enumerate(ds):
        target_size = tuple(item["target_sizes"])
        if target_size not in buckets:
            buckets[target_size] = []
        buckets[target_size].append(idx)
    return buckets

def setup_dataset_transform(
    ds: Dataset,
    random_flip: bool = False,
    random_crop: bool = False,
    vae=None,
    accelerator=None,
    tokenizer=None,
    tokenizer_2=None
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
        # image aug
        if not vae:
            pixel_values = []
            crop_top_lefts = []
            for idx, image in enumerate(examples['image']):
                composer = []  # default_composer
                target_size = examples["target_sizes"][idx]
                if random_flip:
                    # flip
                    composer = [transforms.RandomHorizontalFlip()] + composer
                if not random_crop:
                    composer = [transforms.CenterCrop(target_size)] + composer
                    crop_top = max(
                        0, int(round((image.height - target_size[0]) / 2.0))
                    )
                    crop_left = max(
                        0, int(round((image.width - target_size[1]) / 2.0))
                    )
                    crop_top_left = [crop_top, crop_left]
                else:
                    transforms_random_crop = transforms.RandomCrop(target_size)
                    crop_dict = transforms_random_crop.get_params(image, target_size)
                    crop_top_left = [crop_dict[0], crop_dict[1]]
                    composer = [transforms_random_crop] + composer
                crop_top_lefts.append(crop_top_left)
                image = transforms.Compose(composer)(image)
                image = transforms.Compose(default_composer)(image)
                pixel_values.append(image)
            examples["crop_top_lefts"] = crop_top_lefts
            examples["pixel_values"] = pixel_values
        else:
            examples["latent_values"] = torch.Tensor(examples['latent_values'])
        # text
        captions = []
        for caption in examples["text"]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption))
            else:
                raise ValueError("Caption should contain either strings or lists of strings.")
        examples["input_ids"] = tokenize_prompt(tokenizer, captions)
        examples["input_ids_2"] = tokenize_prompt(tokenizer_2, captions)
        return examples

    if vae:
        print("Caching vae outputs...")
        with accelerator.autocast():
            ds = ds.map(cache_vae_fn, fn_kwargs={"vae": vae, "device": accelerator.device}, keep_in_memory =True, load_from_cache_file=False)
        torch.cuda.empty_cache()
    ds = ds.with_transform(_transform_dataset)
    return ds


class MyBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset, buckets, batch_size: int = 1, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buckets = buckets
        self.has_subfolder = False

    # def _groups(self):
    #     groups = {}
    #     for idx in range(len(self.dataset)):
    #         target_size = self.dataset[idx]["target_sizes"]
    #         if target_size not in groups:
    #             groups[target_size] = []
    #         groups[target_size].append(idx)
    #     return list(groups.values())

    def __iter__(self):
        for k in torch.randperm(len(self.buckets)).tolist():
            sampler = SubsetRandomSampler(self.buckets[k])
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
            return sum([len(x) // self.batch_size for x in self.buckets])  # type: ignore[arg-type]
        else:
            return sum([np.ceil(len(x) / self.batch_size) for x in self.buckets])  # type: ignore[arg-type]


def setup_dataloader(
    ds,
    buckets=None,
    batch_size: int = 1,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
):
    batch_sampler = MyBatchSampler(ds, buckets=buckets, batch_size=batch_size, drop_last=drop_last)
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
    ds = load_dataset_local("D:/Dataset/jdkd/1_jdkd")
    print("Setup dataset")
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )

    pipe = download_from_original_stable_diffusion_ckpt(
        "C:/CODE/ComfyUI_windows_portable/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors",
        from_safetensors=True,
        scheduler_type="ddpm",
        local_files_only=True,
    )
    tokenizer = pipe.tokenizer

    ds = setup_dataset(ds, tokenizer=tokenizer, random_crop=True, random_flip=True)
    for item in ds:
        print(item)

    print("Setup dataloader")
    dl = setup_dataloader(ds, batch_size=4)
    for item in dl:
        print(item)
