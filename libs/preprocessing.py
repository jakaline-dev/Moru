import math
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import torch
from torchvision.transforms import v2
from transformers import CLIPTokenizer


def load_data(img_folder, max_chunk=256, min_chunk=32):
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
            image, (grid_width, grid_height) = preprocess_image(
                image_path, max_chunk, min_chunk
            )
        except Exception as e:
            print(e)
            continue
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


def preprocess_image(image_path, max_chunk=256, min_chunk=32):
    image = Image.open(image_path)
    width, height = image.size
    num_chunks = (width * height) // (64 * 64)
    # If the image is smaller than 32 chunks, skip it
    if num_chunks < min_chunk:
        raise Exception(f"Skipping '{image_path}', too small")
    # If the image is larger than 96 chunks, resize it
    if num_chunks > max_chunk:
        ratio = (max_chunk / num_chunks) ** 0.5

        new_width = int(width * ratio)
        new_height = int(height * ratio)

        new_width = math.ceil(new_width / 64) * 64
        new_height = int(new_width * (height / width))

        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        width, height = image.size
    # grid time
    # width = 64 * m + a
    # height = 64 * n + b
    m: int = width // 64
    a: int = width % 64
    n: int = height // 64
    b: int = height % 64
    if a >= b and a + b <= 64:
        new_height = 64 * n
        new_width = int(width * new_height / height)
        grid_width = 64 * m
        grid_height = 64 * n
    elif a < b and a + b <= 64:
        new_width = 64 * m
        new_height = int(height * new_width / width)
        grid_width = 64 * m
        grid_height = 64 * n
    elif a >= b and a + b > 64:
        new_width = 64 * (m + 1)
        new_height = int(height * new_width / width)
        grid_width = 64 * (m + 1)
        grid_height = 64 * n
    elif a < b and a + b < 64:
        new_height = 64 * (n + 1)
        new_width = int(width * new_height / height)
        grid_width = 64 * m
        grid_height = 64 * (n + 1)

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


@torch.inference_mode()
def cache_vae_outputs(data_list, vae, device):
    vae.to(device)
    for entry in tqdm(data_list):
        tfs = v2.Compose(
            [
                v2.CenterCrop((entry["grid_width"], entry["grid_height"])),
                v2.ToImage(),
                v2.ToDtype(torch.get_default_dtype(), scale=True),
                v2.Normalize([0.5], [0.5]),
            ]
        )
        pixel_values = tfs(entry["image"]).unsqueeze(dim=0)
        latent_dist = vae.encode(pixel_values.to(device)).latent_dist
        entry["latent_values"] = latent_dist.parameters
        entry["latent_values"] = entry["latent_values"].squeeze(dim=0).to("cpu")
        del entry["image"]
        del pixel_values
    vae.to("cpu")
    torch.cuda.empty_cache()
    return data_list


@torch.inference_mode()
def cache_te_outputs(data_list, text_encoder, device, clip_skip):
    text_encoder.to(device)
    for entry in tqdm(data_list):
        # Get the text embedding for conditioning
        encoder_outputs = text_encoder(
            entry["input_ids"].to(device).unsqueeze(dim=0), output_hidden_states=True
        )
        entry["text_embeddings"] = (
            text_encoder.text_model.final_layer_norm(
                encoder_outputs.hidden_states[-clip_skip].to(dtype=text_encoder.dtype)
            )
            .squeeze(dim=0)
            .to("cpu")
        )
        del entry["input_ids"]
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    return data_list
