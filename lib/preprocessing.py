import math
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
        except:
            continue

        # resize
        width, height = image.size
        num_chunks = (width * height) // (64 * 64)

        # If the image is smaller than 32 chunks, skip it
        if num_chunks < min_chunk:
            continue

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
        width_floor = (width // 64) * 64
        height_floor = (height // 64) * 64

        # Determine which dimension is closer to its nearest smaller multiple of 64
        if abs(width - width_floor) < abs(height - height_floor):
            new_width = width_floor
            new_height = int(new_width * (height / width))
        else:
            new_height = height_floor
            new_width = int(new_height * (width / height))
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

        if image.mode == "RGBA":
            # No transparency allowed in PNGs - change alpha to white
            white_background = Image.new("RGB", image.size, (255, 255, 255))
            white_background.paste(image, (0, 0), mask=image.split()[3])
            image = white_background
        image = image.convert("RGB")

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
                "caption": caption,
                "width": image.width,
                "height": image.height,
            }
        )
    return dataset


def crop_dataset(dataset, crop_width, crop_height):
    tfs = transforms.CenterCrop((crop_width, crop_height))
    for entry in dataset:
        img_width, img_height = entry["image"].size
        aspect_ratio = img_width / img_height

        if crop_width == crop_height:
            if img_width < img_height:
                new_width = crop_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = crop_height
                new_width = int(new_height * aspect_ratio)
        else:
            target_aspect_ratio = crop_width / crop_height

            if aspect_ratio < target_aspect_ratio:
                new_width = crop_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = crop_height
                new_width = int(new_height * aspect_ratio)

        entry["image"] = entry["image"].resize((new_width, new_height))
        entry["image"] = tfs(entry["image"])
    return dataset
