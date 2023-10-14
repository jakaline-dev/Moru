import math
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import v2
from collections import defaultdict
from transformers import CLIPTokenizer


def image_folder_to_list(img_folder, max_chunk=256, min_chunk=32):
    dataset = []
    folder_path = Path(img_folder)
    for image_path in folder_path.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() in [
            ".txt",
            ".json",
            ".yaml",
        ]:
            continue
        try:
            image = Image.open(image_path)
            width, height = image.size
            num_chunks = (width * height) // (64 * 64)

            # If the image is smaller than 32 chunks, skip it
            if num_chunks < min_chunk:
                print(f"Skipping '{image_path}', too small")
        except:
            continue
        image, (grid_width, grid_height) = preprocess_image(image, max_chunk, min_chunk)

        caption_path = Path(folder_path, image_path.stem + ".txt")
        if caption_path.is_file():
            with caption_path.open("r") as f:
                lines = f.readlines()
            if len(lines) == 1:
                caption = lines[0].strip()
            else:
                caption = [line.strip() for line in lines]
            # caption = [line.strip() for line in lines]
        else:
            caption = ""
        dataset.append(
            {
                "image": image,
                "image_path": image_path,
                "caption": caption,
                "grid_width": grid_width,
                "grid_height": grid_height,
            }
        )
    return dataset


def preprocess_image(image, max_chunk=256, min_chunk=32):
    # resize
    width, height = image.size
    num_chunks = (width * height) // (64 * 64)

    # If the image is larger than 96 chunks, resize it
    if num_chunks > max_chunk:
        # Calculate the ratio to scale down the image
        ratio = (
            max_chunk / num_chunks
        ) ** 0.5  # Square root because we're dealing with area

        # Calculate new dimensions
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        new_width = math.ceil(new_width / 64) * 64
        new_height = int(new_width * (height / width))

        # Resize the image
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        width, height = image.size

    # Find the nearest smaller multiples of 64 for each dimension
    grid_width = (width // 64) * 64
    grid_height = (height // 64) * 64

    # Determine which dimension is closer to its nearest smaller multiple of 64
    if abs(width - grid_width) < abs(height - grid_height):
        new_width = grid_width
        new_height = int(new_width * (height / width))
    else:
        new_height = grid_height
        new_width = int(new_height * (width / height))
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    if image.mode == "RGBA":
        # No transparency allowed in PNGs - change alpha to white
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        white_background.paste(image, (0, 0), mask=image.split()[3])
        image = white_background
    image = image.convert("RGB")
    return image, (grid_width, grid_height)


def captions_to_tokens(data, tokenizer):
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
    return data


def convert_to_buckets(data_list):
    bucket = defaultdict(list)
    for idx, entry in enumerate(data_list):
        grid_key = (entry["grid_width"], entry["grid_height"])
        bucket[grid_key].append(idx)
    return bucket


def cache_vae_outputs(data_list, vae):
    simple_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.get_default_dtype(), scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )
    for entry in data_list:
        pixel_values = simple_transforms(entry["image"]).unsqueeze(dim=0)
        latent_dist = vae.encode(pixel_values).latent_dist
        entry["latent_values"] = latent_dist.parameters
    return data_list


# LEGACY BELOW


def default_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format)  # .float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


# def crop_dataset(dataset, crop_width, crop_height):
#     tfs = transforms.CenterCrop((crop_width, crop_height))
#     for entry in dataset:
#         img_width, img_height = entry["image"].size
#         aspect_ratio = img_width / img_height

#         if crop_width == crop_height:
#             if img_width < img_height:
#                 new_width = crop_width
#                 new_height = int(new_width / aspect_ratio)
#             else:
#                 new_height = crop_height
#                 new_width = int(new_height * aspect_ratio)
#         else:
#             target_aspect_ratio = crop_width / crop_height

#             if aspect_ratio < target_aspect_ratio:
#                 new_width = crop_width
#                 new_height = int(new_width / aspect_ratio)
#             else:
#                 new_height = crop_height
#                 new_width = int(new_height * aspect_ratio)

#         entry["image"] = entry["image"].resize((new_width, new_height))
#         entry["image"] = tfs(entry["image"])
#     return dataset
